import cv2
import os
import numpy as np
import glob 

import torch
# import lpips
import sys 
sys.path.append('../..')
from utils.common import calculate_ssim,calculate_psnr
from PIL import Image 


gt_paths = sorted(glob.glob('/dataset/kunzhou/project/low_light_noisy/lol_dataset/LOL-v2/Real_captured/Test/Normal/*.png'))
print(len(gt_paths))
# method_path = '/dataset/kunzhou/project/low_light_noisy/exp_decomposition/SNR_ei/snr_ei_lolv2'
# method_path = '/dataset/kunzhou/project/low_light_noisy/Diffusion-Low-Light/wave_lolv2_de/LOLv1'
# method_path = '/dataset/kunzhou/project/low_light_noisy/lol_dataset/LLFF_FLOW/Test'
# method_path = '/dataset/kunzhou/project/low_light_noisy/LLFormer/LLFormer_lolv2_raw'
# method_path = '/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/LoLv2/stage1/predict/Test'

method_path = '/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/LoLv2/stage2/results/mirnet_lap_lolv2/Test' # 24.19/0.882
method_path = '/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/LoLv2/stage2/results/retinexformer_lap_lolv2/Test' # 24.21/0.880
method_path = '/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/LoLv2/stage2/results/snr_lap_lolv2/Test' # 24.17/0.874
method_path = '/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/LoLv2/stage2/results/mirnet_v2_lap_lolv2/Test' # 24.31 / 0.886
method_path = '/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/LoLv2/stage2/results/unet_lap_lolv2/Test' # 24.25 / 0.880


method_path = '/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/LoLv2/mambair_lap_stage2/results/mambair_lap_lolv2/Test'  #24.34 / 0.886
method_path = '/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/LoLv2/retinexmamba_lap_stage2/results/refinexmamba_lap_lolv2/Test' #24.26 / 0.883


method_path = '/dataset/kunzhou/project/low_light_noisy/LLFlow/results/LOLv2-pc/000' # 

# method_path = '/dataset/kunzhou/project/low_light_noisy/Diffusion-Low-Light/wave_lolv2_de/predict'

# method_path = '/dataset/kunzhou/project/SD_Models/Stable_diffusion_pytorch_zk/exp_diffusion/lapdiff/diff_lap_en' # 24.12 0.872
# method_path = '/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/LoLv2/stage1/predict/Test'

method_path = '/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/LoLv2/stage2/results/unet_raw_lolv2/Test' # 21.77M 18.21 0.728


method_path = '/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/LoLv2/stage1/predict/Test'
idx = 0
pred_urls = sorted(glob.glob('{}/*.png'.format(method_path)))
psnr_l = []
ssim_l = []
crop_border = 4
for pred in pred_urls:
    print('pred',pred)
    # if 'wave_lolv2' in pred:
    #     # pred = os.path.join(method_path,'{}.png'.format(idx))
    #     pred_img = cv2.imread(pred).astype(np.float)[...,::-1]
    # else:
    pred_img = cv2.imread(pred).astype(np.float)
    gt_img = cv2.imread(gt_paths[idx]).astype(np.float)
    # print(pred,gt_paths[idx])
    # print(pred_img.shape,gt_img.shape)
    # input('cc')
    idx +=1

    # pred_img = pred_img[crop_border:-crop_border,crop_border:-crop_border]
    # gt_img = gt_img[crop_border:-crop_border,crop_border:-crop_border]

    # cv2.imwrite('tmp.png',np.hstack([pred_img,gt_img]))
    # input('c')
    
    # pred_img = np.clip(pred_img*gt_img.mean()/pred_img.mean(),0,255) # for llflow only

    ssim,psnr = calculate_ssim(pred_img,gt_img),calculate_psnr(pred_img,gt_img)

   

    psnr_l.append(psnr)
    ssim_l.append(ssim)

    print(psnr,ssim)
    # cv2.imshow('pred',pred_img/255)
    # cv2.imshow('gt',gt_img/255)
    # cv2.waitKey(20)
print(sum(psnr_l) / len(psnr_l),sum(ssim_l) / len(ssim_l))

   