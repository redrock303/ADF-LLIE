import cv2
import os
import sys 

import torch
import torchvision
from utils.common import tensor2img, calculate_psnr, calculate_ssim, bgr2ycbcr

import numpy as np
import random
def validate(model, val_loader, device, iteration,  save_path='.', save_img=False):
    # for batch=1
    psnr_l = []
    ssim_l = []
    viz_img = []
    random_select = random.randint(0,500)
    border = 4
    for idx, batch_data in enumerate(val_loader):
        if idx >= 500:
            break
        
        lr_img = batch_data[0].to(device)
        s1_img = batch_data[1].to(device)
        hr_img = batch_data[2].to(device)

        with torch.no_grad():
            h, w = lr_img.size()[2:]
            need_pad = False
            
            # print(lr_img.max(),lr_img.min())
            
            sr_ill,sr_vsr = model(lr_img,s1_img)
            # print('sr_vsr',sr_vsr.mean(),sr_sisr.mean())

   
        

        output_old = sr_ill.detach().cpu().numpy()[0].astype(np.float32)
        output_new = sr_vsr.detach().cpu().numpy()[0].astype(np.float32)

        
        output_old = np.transpose(output_old,(1,2,0))
        output_new = np.transpose(output_new,(1,2,0))

        lr_img_data = np.transpose(lr_img.cpu().numpy()[0].astype(np.float32),(1,2,0))

    #     # print('lr_img_data',lr_img_data.shape)
        # hr_img = model.resize_lx4(hr_img)
        gt = hr_img.cpu().numpy()[0].astype(np.float32)
        gt = np.transpose(gt,(1,2,0))
        # print(gt.shape,output_new.shape,lr_img_data.shape)
        

        if (np.random.random()>0.5 and idx %5==0)or idx == random_select:
            save_img = np.hstack([lr_img_data[:,:]*255.0,output_old[:,:]*255.0,output_new[:,:]*255.0,gt[:,:]*255.0])
            viz_img.append(save_img)
            # cv2.imwrite(ipath, save_img[:,:,::-1].copy())

        # if True:
        #     output_new = bgr2ycbcr(output_new, only_y=True)
        #     gt = bgr2ycbcr(gt, only_y=True)
            

        output_new = output_new[border:-border,border:-border]
        gt = gt[border:-border,border:-border]
        psnr = calculate_psnr(output_new*255.0, gt*255.0)
        ssim = calculate_ssim(output_new*255.0, gt*255.0)
        psnr_l.append(psnr)
        ssim_l.append(ssim)

        # print(psnr,ssim)
        # input('cc')

    avg_psnr = sum(psnr_l) / len(psnr_l)
    avg_ssim = sum(ssim_l) / len(ssim_l)

    print(avg_psnr)
    # input('check')
    ipath = os.path.join(save_path, '%d_cat_check.png' % (iteration))
    saveImg = np.vstack(viz_img)
    cv2.imwrite(ipath, saveImg)
    # input('check')
    return avg_psnr,avg_ssim


if __name__ == '__main__':
    from config import config
    from network import Network
    from dataset.lol_dataset_v2 import LOL_datset as BaseDataset
    from utils import dataloader
    from utils.model_opr import load_model
    from utils.common import *
    # from metric.metric import evaluationPSNR
    model = Network(config)
    device = torch.device('cuda')
    model = model.to(device)
    model_path = '/mnt/yfs/kunzhou/project/low_light_noisy/logs/noisy_lol_guidence/frame_align/models/200000_ft_rf.pth'
    # model.loadWeights()
    load_model(model, model_path)

    val_dataset = BaseDataset(split='val')
    val_loader = dataloader.val_loader(val_dataset, config, 0, 1)
    print(validate(model, val_loader, device, 40000, down=4, to_y=True, save_img=False))
    

    