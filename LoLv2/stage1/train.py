import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2
import argparse
import os.path as osp
import sys 
sys.path.append('../..')
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
from dataloader import *
import math 

class ColorLoss(torch.nn.Module):
    def __init__(self):
        super(ColorLoss,self).__init__()
        pass 
    def forward(self,pred,gt):
        pred = torch.nn.functional.normalize(pred,2,1)
        gt = torch.nn.functional.normalize(gt,2,1)
        loss = torch.abs((pred * gt).sum(1)-1.0)
        return loss.mean()

# Perpectual Loss
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)
        
exp_dir, cur_dir = osp.split(osp.split(osp.realpath(__file__))[0])
root_dir = osp.split(exp_dir)[0]

result_root = '{}/{}'.format('/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/LoLv2/stage1',config.model_version)
if not os.path.exists(result_root): os.mkdir(result_root)
if not os.path.exists(result_root+'/img'): os.mkdir(result_root+'/img')
if not os.path.exists(result_root+'/model'): os.mkdir(result_root+'/model')
print(result_root)

logging.basicConfig(filename='%s/train.log'%result_root,format='%(asctime)s %(message)s', level=logging.INFO)



net = GLNet().cuda()


vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()

for param in vgg_model.parameters():
    param.requires_grad = False

L_cosin = ColorLoss()
L1_smooth_loss = F.smooth_l1_loss

loss_network = LossNetwork(vgg_model)
loss_network.eval()

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
# optimizer = torch.optim.SGD(net.parameters(), lr=config.lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)



train_dataset = lowlight_loader(images_path=config.img_path, normalize=config.normalize)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                           pin_memory=True)

val_dataset = lowlight_loader(images_path=config.img_val_path, mode='test', normalize=config.normalize)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)



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
    rmse = np.zeros(N)
    srmse = np.zeros(N)
    # test_minmax = np.load('%s/test_minmax.npy'%'/home/redrock/data1/segmentation/monocularDepth/dataset/nyu_up')
    
    t = tqdm(iter(val_loader), leave=True, total=len(val_loader))
    show_img = random.randint(0,N)
    for idx, imgs in enumerate(t):
        
        # minmax = test_minmax[:,idx]
        
        low_img, high_img,mask_img = imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda()

        with torch.no_grad():
            out = net(low_img)[-1]

        resore_img_out_fi = out.clamp(0,1).detach().cpu().numpy()[0]
        resore_img_out_fi = np.transpose(resore_img_out_fi,(1,2,0))


        img_hi = high_img.detach().cpu().numpy()[0]
        img_hi = np.transpose(img_hi,(1,2,0))
        psnr = calculate_psnr(resore_img_out_fi,img_hi)
        ssim = 1.0 # calculate_ssim(resore_img_out_fi*255.0,img_hi*255.0)  faster inference 
        rmse[idx] = float(psnr)
        srmse[idx] = float(ssim)

        if idx  == show_img:
            save_tensor = torch.cat([low_img[[0]],out[[0]],high_img[[0]]],0)
            # save_tensor = save_tensor[:,[2,1,0]]
            torchvision.utils.save_image(save_tensor,'{}/img/val_{}_{}.png'.format(result_root,epoch,str(idx)))
            
        t.set_description('[validate] rmse: %f , %f' %(rmse[:idx+1].mean(),srmse[:idx+1].mean()))
        t.refresh()
    
    return rmse.mean(),srmse.mean()

best_eval = 0

for epoch in range(0,config.num_epochs):
    net.train()
    running_loss = 0.0
    
    t = tqdm(iter(train_loader), leave=True, total=len(train_loader))

    for idx, imgs in enumerate(t):
        # break
        # if idx > 2:
        #     break
        optimizer.zero_grad()
        scheduler.step()

        low_img, high_img,mask_img = imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda()

        # dark = low_img
        # dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        # light = mask_img
        # light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        # noise = torch.abs(dark - light)

        # mask = torch.div(light, noise + 0.0001)

        enhance_img = net(low_img)[-1]

        # loss =  ((out - high_img)**2).mean() #+ L_cosin(out,img_h)
        loss = L1_smooth_loss(enhance_img, high_img)+0.04*loss_network(enhance_img, high_img) # + 0.01* L_cosin(enhance_img,high_img)

        # loss =( (enhance_img - high_img)**2).mean()
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm(net.parameters(),0.1)

        optimizer.step()
        running_loss += loss.data.item()

        t.set_description('[train epoch:%d] loss: %.8f' % (epoch+1, running_loss/(idx+1)))
        t.refresh()

        
    rmse ,srmse= validate(net)
    

    if rmse>best_eval:
        best_eval = rmse 
        torch.save(net.state_dict(), "%s/model/parameter_best_c2f-single"%(result_root))
    logging.info('epoch:%d psnr:%f,ssim:%f best_psnr %f'%(epoch+1, rmse,srmse,best_eval))
    