import itertools
import logging
import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from unittest import result
from torch.cuda.amp import GradScaler, autocast
from utils import simclr_accuracy, save_config_file, cls_accuracy, save_checkpoint
from re import L

torch.manual_seed(0)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

classes = ['A1','A2','A3','A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'E1', 'E2', 'E3', 'E4', 'None'] 

class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args           = kwargs['args']
        self.phase          = kwargs['phase']   
        self.Process_list   = kwargs['label']
        self.model          = kwargs['model']
        self.dataloader     = kwargs['dataset']
        self.n_sample       = kwargs['n_sample']
        self.optimizer      = kwargs['optimizer']
        self.scheduler      = kwargs['scheduler']
        self.Multi          = kwargs['multi_label']
        self.writer         = kwargs['writer']
        self.model.to(self.args.device)
            
        for param in self.model.parameters():
            param.requires_grad = True

        if not self.Multi:
            normedWeights = [1 - (x / sum(self.n_sample)) for x in self.n_sample]
            normedWeights = torch.FloatTensor(normedWeights).to(self.args.device)
            logging.info(f"Loss funciont Weights: \n{normedWeights}")
            self.criterion = torch.nn.CrossEntropyLoss(normedWeights).to(self.args.device)
        else:
            self.criterion = torch.nn.MultiLabelSoftMarginLoss().to(self.args.device)
    
    # loss 계산
    def info_nce_loss(self, features, n_views):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(n_views)], dim=0) # labels.shape: (1, batch_size*2)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()                           # labels.shape: (batch*2, batch*2)
        labels = labels.to(self.args.device)                    

        features = F.normalize(features, dim=1)         
        similarity_matrix = torch.matmul(features, features.T)   # (batch*2, batch*2)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device) # 2D tensor with ones on the diagonal and zeros elsewhere (대각행렬)
        labels = labels[~mask].view(labels.shape[0], -1)    # labels.shape: (batch*2, batch*2)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def F1_score(self, output, label, threshold=0.5, beta=1):
        prob = output > threshold
        label = label > threshold

        logging.debug(f"Output:\n {prob}\n")

        TP = (prob & label).sum(1).float()
        TN = ((~prob) & (~label)).sum(1).float()
        FP = (prob & (~label)).sum(1).float()
        FN = ((~prob) & (label)).sum(1).float()

        precision = torch.mean(TP / (TP + FP + 1e-12))
        recall = torch.mean(TP / (TP +FN + 1e-12))
        F2 = (1+beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
        return F2.mean(0), precision, recall

    def plot_confusion_matrix(self, con_mat, epoch, labels, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=False):
        ex_file = f'./{self.writer.log_dir}/confusion_matrix_gray_{epoch-3}.png'
        if os.path.isfile(ex_file):
            os.remove(ex_file)
        plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
        plt.title(title)
        marks = np.arange(len(labels))
        nlabels = []
        for k in range(len(con_mat)):
            n = sum(con_mat[k])
            nlabel = '{0}(n={1})'.format(labels[k],n)
            nlabels.append(nlabel)
        plt.xticks(marks, labels)
        plt.yticks(marks, nlabels)

        thresh = con_mat.max() / 2.
        if normalize:
            for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
                plt.text(j, i, '{0}%'.format(con_mat[i, j] * 100 / n), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
        else:
            for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
                plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
        
        file = f'./{self.writer.log_dir}/confusion_matrix_gray_{epoch}.png'
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(file)
        plt.close()

    def train(self):
        scaler = torch.cuda.amp.GradScaler(enabled=True)        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        best_acc = 0
        best_epoch = 0
        
        for epoch_counter in range(self.args.epochs):
            train_acc = []
            train_losses = []
            train_precision = []
            train_recall = []
            num_correct = 0
            epoch_data = 0
            all_correct_points = 0
            all_points = 0
            wrong_class = np.zeros(self.args.num_class)
            samples_class = np.zeros(self.args.num_class)
            confusion_pred = []
            confusion_target = []

            ## train_loader 재생성
            if epoch_counter > 0:
                filepaths_new = []
                process_data = self.dataloader.dataset.Process_files
                N_files = self.dataloader.dataset.N_files
                None_data = random.choices(N_files, k=69)
                filepaths_new.extend(process_data)
                filepaths_new.extend(None_data)

                self.dataloader.dataset.All_data_list = filepaths_new

            for images, targets, _, _ in self.dataloader:
                if self.phase == 'Simclr':
                    images = [images[0]['image'], images[1]['image']]             
                    images = torch.cat(images, dim=0)                  
                elif self.phase == 'Classifier':
                    images = images['image']
                
                images = images.float().to(self.args.device)        # images.shape: (batch*2, channel, height, width)   --> transform 2 set

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)

                    if self.phase == 'Simclr':
                        logits, labels = self.info_nce_loss(features, n_views=2)
                        loss = self.criterion(logits, labels)           

                    elif self.phase == 'Classifier':
                        output = features.cuda()
                        if self.Multi:
                            targets = targets.to(self.args.device)                  
                            loss = self.criterion(output, targets)       
                        elif not self.Multi:
                            targets = targets.to(self.args.device)
                            loss = self.criterion(output, targets)         
                            
                            _, pred = torch.max(output,1)
                            results = pred == targets

                            confusion_pred.append(pred)
                            confusion_target.append(targets)

                            for i in range(results.size()[0]):
                                if not bool(results[i].cpu().data.numpy()):
                                    wrong_class[targets.cpu().data.numpy().astype('int')[i]] += 1
                                samples_class[targets.cpu().data.numpy().astype('int')[i]] += 1
                            correct_points = torch.sum(results.long())

                            all_correct_points += correct_points
                            all_points += results.size()[0]

                self.optimizer.zero_grad()                
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # Calculation accuracy
                if not self.Multi:
                    if self.phase == 'Simclr':
                        top1, top5 = simclr_accuracy(logits, labels, topk=(1, 5))
                        train_acc.append(top1[0])
                        train_losses.append(loss)

                    else:
                        iter_acc, _correct = cls_accuracy('Train', output, targets, topk=(1,))
                        train_acc.append(iter_acc)
                        train_losses.append(loss)
                        num_correct += _correct
                        epoch_data += targets.shape[0]
                else:
                    logging.debug(f"Iteration: {n_iter:4d}\nOutput:\n {output:.3f}\n")
                    F1_accuracy, precision, recall = self.F1_score(output, targets, threshold=0.5, beta=1)
                    train_acc.append(F1_accuracy)
                    train_precision.append(precision)
                    train_recall.append(recall)
                    train_losses.append(loss)

                if n_iter % self.args.log_every_n_steps == 0:
                    if self.phase == 'Simclr':
                        self.writer.add_scalar('SimCLR_loss', loss, global_step=n_iter)
                        self.writer.add_scalar('train acc/top1', top1[0], global_step=n_iter)
                        self.writer.add_scalar('train acc/top5', top5[0], global_step=n_iter)
                        self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                    elif self.phase == 'Classifier' and self.Multi == False:
                        self.writer.add_scalar('Classifier_loss', loss, global_step=n_iter)    
                        self.writer.add_scalar('classifier acc/top1', iter_acc[0], global_step=n_iter)
                        self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                    elif self.phase == 'Classifier' and self.Multi == True:
                        self.writer.add_scalar('Classifier_loss', loss, global_step=n_iter)    
                        self.writer.add_scalar('classifier F1_Score', F1_accuracy.item(), global_step=n_iter)
                        self.writer.add_scalar('classifier Precision', precision.item(), global_step=n_iter)
                        self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)

                n_iter += 1
            
            # Finish one Epoch    
            if self.phase == 'Simclr':
                Accuracy = torch.stack(train_acc).mean().item()
                Loss = torch.stack(train_losses).mean().item()
                string = f'[SimCLR learning] Epoch: {epoch_counter}, Accuracy: {round(Accuracy,2)}, Loss: {Loss}'
                print(string)

                if best_acc < Accuracy:
                    checkpoint_name = 'best_simclr_checkpoint.pth.tar'
                    save_checkpoint({
                        'epoch': self.args.epochs,
                        'arch': self.args.arch,
                        'state_dict1': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))

                    best_acc = Accuracy
                    best_epoch = epoch_counter
                    logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\t SimCLR Top1 accuracy: {Accuracy}")

            elif self.phase == 'Classifier' and self.Multi == False:
                Accuracy = num_correct.item() / epoch_data
                train_class_acc = (samples_class - wrong_class) / samples_class
                _class_acc =  list(np.array(np.round(train_class_acc,2)))

                pc_list = self.Process_list
                confusion_pred = torch.flatten(torch.stack(confusion_pred))
                confusion_target = torch.flatten(torch.stack(confusion_target))
                matric = confusion_matrix(confusion_target.cpu(), confusion_pred.cpu())
                self.plot_confusion_matrix(matric, epoch_counter, labels=pc_list, normalize=False)
                
                logging.debug(f'\n{matric}')
            
                Loss = torch.stack(train_losses).mean().item()
                string = f'[Single label Classifier] Epoch: {epoch_counter:.2f}, Accuracy: {Accuracy:.2f}, Loss: {Loss:.2f}'
                
                print(string)
                logging.debug(f'\nA1 {_class_acc[0]:.2f}, A2 {_class_acc[1]:.2f}, A3 {_class_acc[2]:.2f}, A4 {_class_acc[3]:.2f}, A5 {_class_acc[4]:.2f}')
                logging.debug(f'B1 {_class_acc[5]:.2f}, B2 {_class_acc[6]:.2f}, B3 {_class_acc[7]:.2f}, B4 {_class_acc[8]:.2f}')
                logging.debug(f'E1 {_class_acc[9]:.2f}, E2 {_class_acc[10]:.2f}, E3 {_class_acc[11]:.2f}, E4 {_class_acc[12]:.2f}')
                logging.debug(f' N {_class_acc[13]:.2f}')

                if best_acc < Accuracy:
                    SimCLR_checkpoint_name = 'best_Classifier_checkpoint.pth.tar'
                    save_checkpoint({
                        'epoch': self.args.epochs,
                        'arch': self.args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best=False, filename=os.path.join(self.writer.log_dir, SimCLR_checkpoint_name))                

                    best_acc = Accuracy
                    best_epoch = epoch_counter
                    str = f"Epoch: {epoch_counter:2d}\tLoss: {loss:.4f}\tClassifier accuracy: {Accuracy:.4f}"
                    logging.debug(str)
                    end = f"====================================================================================\n"
                    logging.debug(end)
            
            elif self.phase == 'Classifier' and self.Multi == True:
                Accuracy = torch.stack(train_acc).mean().item()
                Precision = torch.stack(train_precision).mean().item()
                Recall = torch.stack(train_recall).mean().item()
                Loss = torch.stack(train_losses).mean().item()
                string = f'[Multi label Classifier] Epoch: {epoch_counter}, F1-score: {round(Accuracy,2)}, Precision: {round(Precision,3)}, Recall: {round(Recall,3)}, Loss: {round(Loss,2)}'
                print(string)

                if best_acc < Accuracy:
                    SimCLR_checkpoint_name = 'best_Classifier_checkpoint.pth.tar'
                    save_checkpoint({
                        'epoch': self.args.epochs,
                        'arch': self.args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best=False, filename=os.path.join(self.writer.log_dir, SimCLR_checkpoint_name))                

                    best_acc = Accuracy
                    best_epoch = epoch_counter
                    logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\t Classifier accuracy: {Accuracy}")

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

        logging.info("Training has finished.")
        print('='*60)

        if self.phase == 'SimCLR':
            str = f'Best_SimCLR_accuracy {best_acc:1f} Epoch {best_epoch:02d}'
            print(str)
        else:
            str = f'Best_Classifier_accuracy {best_acc:1f} Epoch {best_epoch:02d}'
            print(str)

    def test(self, data_loader):
        print()
        test_loader = data_loader

        self.model.eval()
        self.classifier.eval()

        logging.info(f"Start SimCLR testing.")
        logging.info(f"Testing with gpu: {self.args.disable_cuda}.")

        All_test_file = 0
        correct_test_file = 0
        for images, targets, path in test_loader:        
            images = images.to(self.args.device)    
            targets = targets.to(self.args.device)

            clsf_features = self.model(images)                 
            output = self.classifier(clsf_features)  

            prob = output > 0.5
            target = targets > 0.5

            print(path[0].split('\\')[-2], path[0].split('\\')[-1])
            print(prob[0])
            print()
            continue
            answer, _, pred = cls_accuracy('Test', output, targets, topk=(1,))
            if answer == False:
                All_test_file += 1
            else:
                All_test_file += 1
                correct_test_file += 1

            f = f'{path[0]}, Predict: {pred}'
            print(f)

        ACC = round(correct_test_file/All_test_file, 2)
        print()
        print(All_test_file)
        print(correct_test_file)
        print(f'Test classification Accuracy: {ACC}')