import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2
import argparse
import os.path as osp
import sys 
sys.path.append('/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/')
sys.path.append('/dataset/kunzhou/project/package_l')
from metrics.common import calculate_ssim

# from config import Configs
from config import config 

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import logging
from datetime import datetime
import os
torch.backends.cudnn.enabled = False
import torchvision

from torchvision.models import vgg16

from ACCA import ACCA as GLNet 
# from dataset.lolv2_full import *
import math 
from dataloader_test import lowlight_loader
import random 

net = GLNet().cuda()

mpath = './lol_v2_acca_best.pt'
net.load_state_dict(torch.load(mpath),strict=False)
print('parameter \"%s\" has loaded'%mpath)


data_part = 'Test' # Train
# img_path = "/dataset/kunzhou/project/low_light_noisy/lol_dataset/LOL-v2/Real_captured/Test/"
if 'Train' in data_part:
    img_path = config.img_path             # save the training results of the stage-1
    val_dataset = lowlight_loader(images_path=img_path, mode='train', normalize=True)
else:
    img_path = config.img_val_path        # save the testing results of the stage-1
    val_dataset = lowlight_loader(images_path=img_path, mode='test', normalize=True)

w_root = './predict/{}'.format(data_part)
if not os.path.exists(w_root):
    os.mkdir(w_root)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)*255.0
    img2 = img2.astype(np.float64)*255.0
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * math.log10(255.0 / math.sqrt(mse))

@torch.no_grad()
def validate(net, root_dir=''):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    net.eval()
    N = val_dataset.__len__()
    
    # test_minmax = np.load('%s/test_minmax.npy'%'/home/redrock/data1/segmentation/monocularDepth/dataset/nyu_up')
    border = 8 
    t = tqdm(iter(val_loader), leave=True, total=len(val_loader))
    show_img = random.randint(0,N)
    for idx, imgs in enumerate(t):
        
        # minmax = test_minmax[:,idx]
        
        low_img, high_img,mask_img = imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda()
        # print(low_img.max(),low_img.min())
        # input('cc')

        # dark = low_img
        # dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        # light = mask_img
        # light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        # noise = torch.abs(dark - light)

        # mask = torch.div(light, noise + 0.0001)
        with torch.no_grad():
            out = net(low_img)[-1]

        resore_img_out_fi = out.clamp(0,1).detach().cpu().numpy()[0]
        resore_img_out_fi = np.transpose(resore_img_out_fi,(1,2,0))

        if True:
           cv2.imwrite('./predict/{}/{}.png'.format(data_part,str(idx).zfill(5)),resore_img_out_fi[:,:,::-1]*255)
        t.refresh()
    
    # return rmse.mean(),srmse.mean()

best_eval = 0

epoch = 0

        
validate(net)

    