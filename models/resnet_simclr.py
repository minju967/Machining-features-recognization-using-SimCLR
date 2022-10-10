from tabnanny import check
from xml.etree.ElementTree import QName
import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError
import torchvision.models as models

class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim, num_cls, checkpoint, phase):
        super(ResNetSimCLR, self).__init__()
        self.phase = phase
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=True, num_classes=1000),
                            "resnet50": models.resnet50(pretrained=True, num_classes=1000)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features   # resnet18: 256

        if self.phase == 'Simclr':
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))
        elif self.phase == 'Classifier':
            self.backbone.fc  = nn.Sequential(nn.Linear(dim_mlp, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64,num_cls), nn.Sigmoid())
        else:
            self.backbone.fc  = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim), nn.Linear(out_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_cls), nn.Sigmoid())

        if checkpoint:
            simcls_pt_path = os.path.join('/root/MINJU/SimCLR_Learning/runs', checkpoint, 'best_Classifier_checkpoint.pth.tar')
            simcls_checkpoint = torch.load(simcls_pt_path)
            new_simcls_chechpoint = {}
            for key in list(simcls_checkpoint['state_dict'].keys()):
                val = simcls_checkpoint['state_dict'][key]
                new_simcls_chechpoint[key.replace('backbone.','')] = val
            self.backbone.load_state_dict(new_simcls_chechpoint, strict=False)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        # if SimCLR output shape is [batch, out_dim]
        # else Classifier output shape is [batch, num_cls]
        return self.backbone(x) 


import torch
import logging
import os

class SimCLR_Classifier(nn.Module):
    def __init__(self, base_model, out_dim, num_cls, checkpoint, task):
        super(SimCLR_Classifier, self).__init__()
        self.phase = task
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=True, num_classes=1000),
                            "resnet50": models.resnet50(pretrained=True, num_classes=1000)}
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features      # backbone 모델의 fc layer의 input channel 수 추출
        
        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))

        if base_model == 'resnet18':
            self.classifier = nn.Sequential(nn.Linear(out_dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, num_cls), nn.Sigmoid())
        elif base_model == 'resnet50':
            self.classifier = nn.Sequential(nn.Linear(out_dim, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, num_cls), nn.Sigmoid())
        
        if checkpoint[0]:
            simcls_pt_path = os.path.join('/root/MINJU/SimCLR_Learning/runs', checkpoint[0], 'best_simclr_checkpoint.pth.tar')
            simcls_checkpoint = torch.load(simcls_pt_path)
            new_simcls_chechpoint = {}
            for key in list(simcls_checkpoint['state_dict1'].keys()):
                val = simcls_checkpoint['state_dict1'][key]
                new_simcls_chechpoint[key.replace('backbone.','')] = val
            self.backbone.load_state_dict(new_simcls_chechpoint)

            logging.info(f"SimCLS checkpoint file directory: {simcls_pt_path}")

        if checkpoint[1]:
            clf_pt_path = os.path.join('/root/MINJU/SimCLR_Learning/runs', checkpoint[0], 'best_Classifier_checkpoint.pth.tar')
            clf_checkpoint = torch.load(clf_pt_path)
            self.backbone.load_state_dict(clf_checkpoint['state_dict2'])
            logging.info(f"Classifier checkpoint file directory: {clf_pt_path}")

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        if self.phase == 'Simclr':
            return self.backbone(x)             # [batch, out_dim]
        else:
            features = self.backbone(x)
            y = self.classifier(features)       
            return y                            # [batch, num_cls]
