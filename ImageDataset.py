from calendar import c
import os
import glob
import numpy as np
import random
import cv2
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from albumentations.pytorch import ToTensorV2

from PIL import Image
from torchvision.transforms import transforms

classes = ['A1','A2','A3','A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'E1', 'E2', 'E3', 'E4', 'N'] 

class GaussianBlur():
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        # tensor image PIL 변환
        img = self.tensor_to_pil(img)

        return img

class ContrastiveLearningViewGenerator2(object):
    """Take two random crops of one image as the query and key."""
    # 하나의 이미지에서 두개의 transformed images 생성.
    
    def __init__(self, phase, transform):
        self.phase = phase
        self.base_transform = transforms.Compose([transforms.Resize(size=224),
                                                  transforms.ToTensor()])
        self.transform = transform

    def __call__(self, images):

        if self.phase == 'Train':
            trans_images = []
            AG = random.choices([0, 90, 180, 270], k=2)
            
            C_image = images[0]
            trans_images.append(TF.rotate(self.transform(C_image), angle=AG[0]))
            D_image = images[1]
            trans_images.append(TF.rotate(self.transform(D_image), angle=AG[0]))
            
            return trans_images
        else:
            return self.base_transform(images[0])

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""
    # 하나의 이미지에서 두개의 transformed images 생성.
    
    def __init__(self, phase, transform, n_views=2):
        self.phase = phase
        self.base_transform = transforms.Compose([transforms.Resize(size=224),
                                                  transforms.ToTensor()])
        self.transform = transform
        self.n_views = n_views

    def __call__(self, x):
        if self.phase == 'Train':
            AG = random.choices([0, 90, 180, 270], k=2)
            return [TF.rotate(self.base_transform(x), angle=AG[0]) if i==0 else TF.rotate(self.transform(x), angle=AG[1]) for i in range(self.n_views)]
        else:
            return self.base_transform(x)

def get_simclr_pipeline_transform(phase):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    if phase == 'Train':
        s = 1
        data_transforms = transforms.Compose([transforms.Resize(size=224),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomGrayscale(p=0.7),
                                            transforms.ToTensor()
                                            ])
    else:
        data_transforms = transforms.Compose([transforms.Resize(size=224),
                                            transforms.ToTensor()])

    return data_transforms

MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]

class ContrastiveLearningDataset:
    def __init__(self, root_folder, phase, n_view=None, option=False):    
        self.root_path = root_folder 
        if phase == 'Train':
            self.Base_data_dir = os.path.join(root_folder, 'Color')
        else:
            self.Base_data_dir = os.path.join(root_folder, 'Depth_test')

        self.n_view = n_view
        self.option = option
        self.phase = phase
        classes = os.listdir(self.Base_data_dir) 

        all_files = []
        for p in classes:
            cls_path = os.path.join(self.Base_data_dir, p)
            f_list = [[f.split('\\')[-3], f.split('\\')[-2], os.path.basename(f)]for f in glob.glob(cls_path+'\\*.png')] # [Color, class, name]
            all_files.extend(f_list)

        random.shuffle(all_files)
        self.All_data_list = all_files

        self.transform = A.Compose([# transforms.Pad(padding=random.choice([0,10,30,50,100])),
                                    A.HorizontalFlip(p=0.5),
                                    A.VerticalFlip(p=0.5),
                                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=1),
                                    A.Resize(224,224),
                                    ToTensorV2()
                                    ])
        # if phase == 'Train':
        #     self.transform = ContrastiveLearningViewGenerator(self.phase, get_simclr_pipeline_transform(phase),
        #                                                       self.n_view)
        # else:
        #     self.transform = ContrastiveLearningViewGenerator(self.phase, get_simclr_pipeline_transform(phase),
        #                                                       self.n_view)

    def __len__(self):
        return len(self.All_data_list)

    def __getitem__(self, idx):
        cls = self.All_data_list[idx][1]
        name = self.All_data_list[idx][2]

        img1_path = os.path.join(self.root_path, 'Color', cls, name)
        img2_path = os.path.join(self.root_path, 'gray_scale', cls, name)

        image1 = np.array(Image.open(img1_path).convert('RGB'))
        image2 = np.array(Image.open(img2_path).convert('RGB'))
        # image1 = cv2.imread(img1_path)
        # image2 = cv2.imread(img2_path)

        new_images = [self.transform(image=x) for x in [image1, image2]]
        
        if self.option:
            _images = []
            for img in new_images:
                np_img = img[1,:,:].numpy()
                idx = np.where(np_img==0)
                for h, w in zip(idx[0], idx[1]):
                    img[0][h][w] = 1
                    img[1][h][w] = 1
                    img[2][h][w] = 1
                _images.append(img)
            new_images = _images

        return new_images, 0, 0, 0

class Process_Simclr_Dataset:
    def __init__(self, root_folder, phase, n_view, Data):
        self.root_path = root_folder     

        if Data == 'Gray_':
            self.Base_data_dir = '/root/MINJU/SimCLR_Learning/DATA/Process_Gray'
        elif Data == 'Color':
            self.Base_data_dir = '/root/MINJU/SimCLR_Learning/DATA/Process_Color'

        self.n_view = n_view
        self.phase = phase
        self.processes = os.listdir(self.Base_data_dir) # A1-A5, A1-B4, E1-E4, N
        self.processes.sort()
        self.Process_files = []
        self.N_files = []
        self.Data = Data
        self.n_samples_cls = []
        for p in self.processes:
            p_path = os.path.join(self.Base_data_dir, p)    # label directory path
            self.n_samples_cls.append(len(os.listdir(p_path)))
            for f_path in glob.glob(p_path+'/*.png'):
                label = f_path.split('/')[-2]
                self.Process_files.append([label, os.path.basename(f_path)])
                
        p_path = os.path.join(self.Base_data_dir, 'N')    # label directory path
        files = glob.glob(p_path+'/*/*.png')
        for f in files:
            cls = f.split('/')[-2]
            F_name = os.path.basename(f)
            self.N_files.append(['N', cls, F_name])
            
        None_data = random.choices(self.N_files, k=69)
        
        self.All_data_list = []
        self.All_data_list.extend(self.Process_files)
        self.All_data_list.extend(None_data)

        random.shuffle(self.All_data_list)

        self.transform = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=1),
                            A.Resize(224,224),
                            ToTensorV2()])

    def __len__(self):
        return len(self.All_data_list)

    def __getitem__(self, idx):

        label = self.All_data_list[idx][0]  # A1-E4 and N
        label_idx = self.processes.index(label)
        
        img_name = self.All_data_list[idx][-1]

        if label == 'N':
            cls = self.All_data_list[idx][1]
            img_path = os.path.join(self.Base_data_dir, label, cls, img_name)
        else:
            img_path = os.path.join(self.Base_data_dir, label, img_name)   # Color images
            
        image = np.array(Image.open(img_path).convert('RGB'))
        new_image = self.transform(image=image)

        return new_image, label_idx, img_path, label

class D_class_contrastiveLearningDataset:
    def __init__(self, phase, n_view):

        self.n_view = n_view
        self.phase  = phase

        self.Base_data_dir = 'D:\\VS_CODE\\UJU_Dataset'
        
        color_files = []
        self.files  = os.listdir(os.path.join(self.Base_data_dir, 'Color', 'D'))
        for f in self.files:
            color_img = os.path.join(self.Base_data_dir, 'Color', 'D', f)
            if os.path.splitext(color_img)[1]  == '.png':
                color_files.append(os.path.basename(color_img))

        random.shuffle(color_files)
        self.All_data_list = color_files

        if self.phase == 'Train':
            self.transform = ContrastiveLearningViewGenerator2(self.phase, get_simclr_pipeline_transform(phase))
        else:
            self.transform = ContrastiveLearningViewGenerator2(self.phase, get_simclr_pipeline_transform(phase))

    def __len__(self):
        return len(self.All_data_list)

    def __getitem__(self, idx):

        C_img_path = os.path.join(self.Base_data_dir, 'Color', 'D', self.All_data_list[idx])
        D_img_path = os.path.join(self.Base_data_dir, 'RGB_D', 'D', self.All_data_list[idx])

        C_image = Image.open(C_img_path).convert('RGB')
        D_image = Image.open(D_img_path).convert('RGB')

        new_images = self.transform([C_image, D_image])
        
        return new_images, 1

def encode_label(label, classes_list = classes):
    if not isinstance(label, list):
        label = label.split(' ')

    target = torch.zeros(len(classes))
    for l in label:
        idx = classes_list.index(l)
        target[idx] = 1
    
    target[10] = 1

    return target

class DatasetforSimilarity:
    def __init__(self, root):
        self.Base_data_dir = root

        all_files = []
        files = os.listdir(self.Base_data_dir)
        for f in files:
            if os.path.splitext(f):
                all_files.append(os.path.basename(f))

        random.shuffle(all_files)
        self.All_data_list = all_files
        self.transform = transforms.Compose([transforms.Resize(size=224),
                                            # transforms.RandomGrayscale(p=1.0),
                                            transforms.ToTensor()])

    def __len__(self):
        return len(self.All_data_list)

    def __getitem__(self, idx):
        label = 1

        img_path = os.path.join(self.Base_data_dir, self.All_data_list[idx])
        image = Image.open(img_path).convert('RGB')
        new_images = self.transform(image)
        
        return new_images, label, img_path

class Process_Test_Dataset:
    def __init__(self, root_folder):
        self.root_path = root_folder     

        self.Base_data_dir = os.path.join(root_folder, 'Test_dataset', 'Process_Color')
        self.classes = ['A', 'B', 'C', 'D', 'E']

        all_files = []
        for cls in self.classes:
            p_path = os.path.join(self.Base_data_dir, cls)    # label directory path
            f_list = [[f.split('\\')[-2], os.path.basename(f)]for f in glob.glob(p_path+'\\*.png')] # [label, image_name]
            all_files.extend(f_list)

        # random.shuffle(all_files)
        self.All_data_list = all_files

        self.transform = ContrastiveLearningViewGenerator('Test', get_simclr_pipeline_transform('Test'),1)

    def __len__(self):
        return len(self.All_data_list)

    def __getitem__(self, idx):

        label = self.All_data_list[idx][0]
        label_idx = self.classes.index(label)
        
        img_name = self.All_data_list[idx][1]

        img_path = os.path.join(self.Base_data_dir, label, img_name)                                # pyvista images
        
        image = Image.open(img_path).convert('RGB')
        new_images = self.transform(image)

        return new_images, label_idx, img_path
