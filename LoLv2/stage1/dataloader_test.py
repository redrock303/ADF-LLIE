import os
import os.path as osp

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
from torchvision.transforms import Compose, ToTensor, Normalize, ConvertImageDtype
import cv2
# Code change from "https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/dataloader.py"
# By Ziteng Cui, cui@mi.t.u-tokyo.ac.jp
random.seed(1143)

def populate_train_list(images_path, mode='test'):
    # print(images_path)
    image_list_lowlight = sorted(glob.glob(images_path+'/Low/' + '*.png'))
    image_list_normlight = sorted(glob.glob(images_path+'/Normal/' + '*.png'))
    
    train_list = [[url1,url2] for (url1,url2) in zip(image_list_lowlight,image_list_normlight)]

    return train_list

class lowlight_loader(data.Dataset):

    def __init__(self, images_path, mode='train', normalize=True):
        self.train_list = populate_train_list(images_path, mode)
        #self.h, self.w = int(img_size[0]), int(img_size[1])
        # train or test
        self.mode = mode
        self.data_list = self.train_list
        self.normalize = normalize
        print("Total examples:", len(self.train_list))

    # Data Augmentation
    # TODO: more data augmentation methods
    def FLIP_LR(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_LEFT_RIGHT)
            high = high.transpose(Image.FLIP_LEFT_RIGHT)
        return low, high

    def FLIP_UD(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_TOP_BOTTOM)
            high = high.transpose(Image.FLIP_TOP_BOTTOM)
        return low, high
    
    def get_params(self, low):
        self.w, self.h = low.size
        
        self.crop_height = 64 #random.randint(self.MinCropHeight, self.MaxCropHeight)
        self.crop_width = 64 #random.randint(self.MinCropWidth,self.MaxCropWidth)
        # self.crop_height = 224 #random.randint(self.MinCropHeight, self.MaxCropHeight)
        # self.crop_width = 224 #random.randint(self.MinCropWidth,self.MaxCropWidth)

        i = random.randint(0, self.h - self.crop_height)
        j = random.randint(0, self.w - self.crop_width)
        return i,j

    def Random_Crop(self, low, high):
        self.i,self.j = self.get_params((low))

        low = low.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
        high = high.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
        return low, high
    
    
    def __getitem__(self, index):
        data_lowlight_path,data_normlight_path = self.data_list[index]
        
        
        data_lowlight = Image.open(data_lowlight_path)
        # data_highlight = Image.open(data_lowlight_path.replace('low', 'normal').replace('Low','Normal'))

        # img_name = (data_lowlight_path.split('/')[-1]).replace('low', 'normal')
        # data_normlight_path = os.path.join('/dataset/kunzhou/project/low_light_noisy/lol_dataset/LOL-v2/Real_captured/Test/Normal',img_name)
        data_highlight = Image.open(data_normlight_path)

        
        data_lowlight, data_highlight = (np.asarray(data_lowlight) / 255.0), (np.asarray(data_highlight) / 255.0)
        #data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(data_highlight).float()
        if self.normalize:
            #data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(data_highlight).float()
            transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ConvertImageDtype(torch.float), ])
            transform_gt = Compose([ToTensor(), ConvertImageDtype(torch.float), ])
            #return transform_input(data_lowlight).permute(2, 0, 1), transform_gt(data_highlight).permute(2, 0, 1)
            # return transform_input(data_lowlight), transform_gt(data_highlight)
            data_low = transform_input(data_lowlight)
            img_nf = (data_low.permute(1, 2, 0).numpy() +1.0)*0.5
            img_nf = cv2.blur(img_nf, (5, 5))
            img_nf = img_nf * 1.0 / 255.0
            img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1)
            
            return data_low, transform_gt(data_highlight),img_nf
        else:
            data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(data_highlight).float()

            img_nf =  (data_lowlight.permute(1, 2, 0).numpy() +1.0)*0.5
            img_nf = cv2.blur(img_nf, (5, 5))
            img_nf = img_nf * 1.0 / 255.0
            img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1)
            return data_lowlight.permute(2,0,1), data_highlight.permute(2,0,1),img_nf
            
    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    images_path = '/data/unagi0/cui_data/light_dataset/LOL_v2/Train/Low/'

    train_dataset = lowlight_loader(images_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4,
                                               pin_memory=True)
    for iteration, imgs in enumerate(train_loader):
        print(iteration)
        print(imgs[0].shape)
        print(imgs[1].shape)