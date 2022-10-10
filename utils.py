import os
import shutil

import torch
import yaml


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def simclr_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0) 
        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))   # true or false

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def cls_accuracy(phase, output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0) 
        _, pred = output.topk(maxk, 1, True, True)
        if phase == 'Test':
            if pred.item() in [0, 1, 2, 3, 4]:
                pred = torch.tensor([[0]]).to('cuda')
                answer = 'A'
            elif pred.item() in [5,6,7,8]:  
                pred = torch.tensor([[1]]).to('cuda')
                answer = 'B'
            else:
                pred = torch.tensor([[2]]).to('cuda')
                answer = 'E'

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))   # true or false
        correct_points = torch.sum(correct.long())
        
        k = 1
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        result = correct_k.mul_(100.0 / batch_size)
        if phase == 'Train':
            return result, correct_points
        elif phase == 'Test':
            return result, correct_points, answer

