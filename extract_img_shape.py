
# A1~ E4까지 각 데이터의 feature 추출.
# Unsupervised learning
# self supervised learning.
# downstream 을 통해 추출된 feature를 가지고 clustering(13classes)을 했을 경우,
# 분류가 되는지 확인.
# 
# 분류가 잘 된다면, label이 없는 C와 D도 적용.
# D class 이미지와 random한 연삭, clean이미지 sampling

# A1~E4, 연삭N, clean 이미지 모두 분류 
# 나머지 고속전극 (분류된 이미지 속에서 잘못된 분류 있는지 확인 해야함.) --> B와 D이미지가 들어가면 안됨.


import os
import torch
import torchvision 
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pickletools import optimize
from torch.utils.data import DataLoader

from simclr import SimCLR
from exceptions.exceptions import InvalidGUP
from models.resnet_simclr import ResNetSimCLR
from models.resnet_simclr import SimCLR_Classifier

from ImageDataset import Process_Simclr_Dataset, Process_Test_Dataset
from ImageDataset import Process_Test_Dataset

from Multi_label_classification import Multi_label_Simclr_Dataset
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

np.random.seed(0)
        
def data_show(dataset):
    while True:
        Images = next(iter(dataset))        
        img1 = Images[0]['image'].squeeze()
        img2 = Images[1]['image'].squeeze()

        image1 = img1[0].numpy()
        image2 = img2[0].numpy()
        print(image1.shape)

        img1 = image1.transpose(1,2,0)
        img2 = image2.transpose(1,2,0)

        plt.subplot(1,2,1)
        plt.imshow(img1)
        plt.subplot(1,2,2)
        plt.imshow(img2)

        plt.show()
        
model_names = 'resnet18'

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=320, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr') # 0.0003
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--base_path', default='/root/MINJU/SimCLR_Learning/DATA',
                    help='Image_directory Path')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--num_class', default=15, type=int,
                    help='number of class')
parser.add_argument('--log-every-n-steps', default=1, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu_index', default=0, type=int, help='Gpu index.')
parser.add_argument('--option', default=False, type=bool, help='Gpu index.')
parser.add_argument('--Model_pt', default='Oct09_11-40-18_57ccb9811bd5',help='Image_directory Path')
# parser.add_argument('--task', default='Simclr', type=str, help='Simclr or Classifier')
parser.add_argument('--task', default='Classifier', type=str, help='Simclr or Classifier')
parser.add_argument('--Data', default='Gray_', type=str, help='Color or Gray_')

def main():
    writer = SummaryWriter()
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        # args.gpu_index = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if not args.device:
        InvalidGUP()
        return 

    train_dataset = Process_Simclr_Dataset(args.base_path, 'Train', args.n_views, args.Data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=False, drop_last=True)

    print(f'Train_dataset: {len(train_dataset.All_data_list)}')
    print(f'Process List: {train_dataset.processes}')
    # data_show(train_loader)

    ### SimCLR, Classifier ###  
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, num_cls=args.num_class, checkpoint=args.Model_pt, phase=args.task).cuda()
    logging.info(f"Model Architecture: \n{model}")
    # model = SimCLR_Classifier(base_model=args.arch, out_dim=args.out_dim, num_cls=args.num_class, checkpoint=[args.SimCLS_pt, args.CLF_pt], task=args.task).cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.0001,
                                                                last_epoch=-1)
    with torch.cuda.device(0):
        # simclr = SimCLR(phase=Phase, dataset=train_loader, model=model, classifier=classifier, optimizer=optimizer, scheduler=scheduler, args=args, multi_label=False, label=train_dataset.processes)
        simclr = SimCLR(phase=args.task, dataset=train_loader, model=model, optimizer=optimizer, scheduler=scheduler, args=args, multi_label=False, label=train_dataset.processes, n_sample=train_dataset.n_samples_cls, writer = writer)
        simclr.train()

def test():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')

    test_dataset = Process_Test_Dataset(args.base_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True)


    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim).cuda()
    classifier = SimCLR_CLF(backbone=args.arch, input_dim=args.out_dim, num_cls=args.num_class).cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    Simclr_checkpoint = torch.load('D:\\VS_CODE\\UJU_Project\\Contrastive_learning\\runs\\Oct05_16-28-45_DESKTOP-6N5F65J\\best_simclr_checkpoint.pth.tar')
    model.load_state_dict(Simclr_checkpoint['state_dict1'])

    classifier_checkpoint = torch.load('D:\\VS_CODE\\UJU_Project\\Contrastive_learning\\runs\\Oct06_04-47-42_DESKTOP-6N5F65J\\best_Classifier_checkpoint.pth.tar')
    classifier.load_state_dict(classifier_checkpoint['state_dict2'])
    
    print()
    print(f'Test_dataset: {len(test_dataset.All_data_list)}\n')   

    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(phase='Test', model=model, classifier=classifier, optimizer=optimizer, scheduler=None, args=args, multi_label=True)
        simclr.test(test_loader)

def Multi_label_train():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.gpu_index = 0

    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if not args.device:
        InvalidGUP()
        return 

    train_dataset = Multi_label_Simclr_Dataset(args.base_path, 'Multi_label', args.n_views)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True)

    print(f'Train_dataset: {len(train_dataset.All_data_list)}')

    Phase = 'Classifier'

    if Phase == 'SimCLR':
        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim).cuda()
        classifier = None
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    
        '''
        checkpoint = torch.load('D:\\VS_CODE\\UJU_Project\\Contrastive_learning\\runs\\22_10_3_process_1\\best_simclr_checkpoint.pth.tar')
        model.load_state_dict(checkpoint['state_dict1'])
        '''

    elif Phase == 'Classifier':
        multi_label = 14
        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim).cuda()
        classifier = SimCLR_CLF(backbone=args.arch, input_dim=args.out_dim, num_cls=multi_label).cuda()
        optimizer = torch.optim.Adam(classifier.parameters(), args.lr, weight_decay=args.weight_decay)
        
        checkpoint = torch.load('D:\\VS_CODE\\UJU_Project\\Contrastive_learning\\runs\\Oct05_16-28-45_DESKTOP-6N5F65J\\best_simclr_checkpoint.pth.tar')
        model.load_state_dict(checkpoint['state_dict1'])
        
        # checkpoint = torch.load('D:\\VS_CODE\\UJU_Project\\Contrastive_learning\\runs\\Oct06_04-18-44_DESKTOP-6N5F65J\\best_Classifier_checkpoint.pth.tar')
        # classifier.load_state_dict(checkpoint['state_dict2'])
        

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                                last_epoch=-1)
    with torch.cuda.device(0):
        simclr = SimCLR(phase=Phase, model=model, classifier=classifier, optimizer=optimizer, scheduler=scheduler, args=args, multi_label=True)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
    # test()
    # Multi_label_train()
