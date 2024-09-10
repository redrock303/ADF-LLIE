
import cv2
import numpy as np
import os
import sys

import torch
import torch.utils.data as tData
import glob

import random
import json 
import sys 
import numpy as np
# sys.path.append('/dataset/kunzhou/project/low_light_noisy/')

import time 
# from PIL import Image
# from skimage.transform import resize
import math 
import random 
import sys 
# from utils.bsr.bsr import degradation_bsrgan_plus
def resize_padding(img):
    img = img.unsqueeze(0)
    h,w = img.size()[-2:]
    if h > w:
        pad_w = h - w 
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left 
        img = torch.nn.functional.pad(img,(pad_left,pad_right,0,0),mode='reflect')
    elif h<w:
        pad_h = w - h 
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top 
        img = torch.nn.functional.pad(img,(0,0,pad_top,pad_bottom),mode='reflect')
    size = img.size()[-1]
    # print('iomg pad',img.shape,size)
    img_resize = torch.nn.functional.interpolate(img,(128,128),mode='bilinear',align_corners=False)[0]
    return img_resize

def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    
    data[data==0] = min_nonzero
    return data

class NYU_v2_datset(tData.Dataset):
    def __init__(self, root_dir = '/dataset/kunzhou/project/low_light_noisy/lol_dataset/LOL-v2',
                    split = 'train',patch_width = 48,path_height = 48,rank=0,model_type='star'):
        self.root_dir = root_dir 
        self.rank = rank

        self.patch_width = patch_width
        self.patch_height = path_height

        self.split = split
        

        hn_root = '/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/LoLv2/stage1/predict/'
        self.data_len = 0
        self.img_list = []
        if 'train' in self.split:
            # load real
            l_path = os.path.join(root_dir,'Real_captured','Train','Low')
            h_path = os.path.join(root_dir,'Real_captured','Train','Normal')
            hn_path = sorted(glob.glob('{}/Train/*.png'.format(hn_root))) 
            idx = 0
            for img_name in  sorted(os.listdir(l_path)):
                # print(img_name,img_name.replace('low','normal'))
                # input('cc')
                img_h = os.path.join(h_path,img_name.replace('low','normal'))
                img_l = os.path.join(l_path,img_name)
                img_hn = hn_path[idx]
                if not os.path.exists(img_h) or not os.path.exists(img_l) or not os.path.exists(img_hn):
                    print(img_l,img_h,img_hn)
                    input('cc')
                idx+=1
                self.img_list.append([img_l,img_h,img_hn])
            print('1',len(self.img_list))

        else:
            l_path = os.path.join(root_dir,'Real_captured','Test','Low')
            h_path = os.path.join(root_dir,'Real_captured','Test','Normal')

            hn_root = '/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/LoLv2/stage1/predict/'
            hn_path = sorted(glob.glob('{}/Test/*.png'.format(hn_root))) 
            idx = 0
            for img_name in  sorted(os.listdir(l_path)):
                # print(img_name,img_name.replace('low','normal'))
                # input('cc')
                img_h = os.path.join(h_path,img_name.replace('low','normal'))
                img_l = os.path.join(l_path,img_name)
                img_hn = hn_path[idx]
                self.img_list.append([img_l,img_h,img_hn])
                idx+=1
        
        if 'train' in self.split and self.rank!=0:
            random.shuffle(self.img_list)
        
        self.data_len = len(self.img_list)
        # self.data_len = 5000
        print('{} sample'.format(self.data_len))

        self.patch_width = patch_width
        self.patch_height = patch_width
        # sefl.scale = 1
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):


        url_l,url_h,url_hn = self.img_list[index]
        # print('url_l,url_h',url_l,url_h,os.path.exists(url_h),os.path.exists(url_l))

        # img_l = cv2.imread(url_l)
        # img_l = np.rot90(img_l)
        # cv2.imwrite(url_l,img_l)
        # input('cc')

        img_l = cv2.imread(url_l).astype(np.float)/255.0
        img_h = cv2.imread(url_h).astype(np.float)/255.0
        img_hn = cv2.imread(url_hn).astype(np.float)/255.0
        name = url_l.split('/')[-1]
        name = name.split('.')[0]

        

        # global_ratio_r = img_h[:,:,0].mean() / (img_l[:,:,0].mean()+1e-6)
        # global_ratio_g = img_h[:,:,1].mean() / (img_l[:,:,1].mean()+1e-6)
        # global_ratio_b = img_h[:,:,2].mean() / (img_l[:,:,2].mean()+1e-6)

        # ratio_array = np.array([global_ratio_r,global_ratio_g,global_ratio_b])
        # img_l = cv2.resize(img_l,(512,512))
        # img_h = cv2.resize(img_h,(512,512))

        h,w,c = img_h.shape
        if 'train' in self.split:
            

            x = random.randint(0,w-self.patch_width-1)
            y = random.randint(0,h-self.patch_height-1)

            img_h = img_h[y:y+self.patch_height, x:x+self.patch_width]
            img_l = img_l[y:y+self.patch_height, x:x+self.patch_width]
            img_hn = img_hn[y:y+self.patch_height, x:x+self.patch_width]
            h_flip = np.random.random() > 0.5
            v_flip = np.random.random() > 0.5
            r_flip = np.random.random() > 0.5

            if h_flip:
                img_h = img_h[:, ::-1].copy()
                img_hn = img_hn[:, ::-1].copy()
                img_l = img_l[:, ::-1].copy()
                
            if v_flip:
                img_h = img_h[::-1, :].copy()
                img_hn = img_hn[::-1, :].copy()
                img_l = img_l[::-1,:].copy()
                
            if r_flip:
                img_h = img_h.transpose(1,0,2)
                img_hn = img_hn.transpose(1,0,2)
                img_l = img_l.transpose(1,0,2)
        
        # for snr only 
        img_nf = cv2.blur(img_l, (5, 5))
        img_nf = img_nf * 1.0 / 255.0
        img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1).float()


        # if 'val' in self.split:
        #     noise_img = degradation_bsrgan_plus(img_l)
        #     ration = 1
        #     img_l = ration * noise_img +(1-ration)*img_l
        # print('..',img_h.shape,img_l.shape)
        img_h = np.transpose(img_h,(2,0,1))
        img_hn = np.transpose(img_hn,(2,0,1))
        img_l = np.transpose(img_l,(2,0,1))

        
        


        # img_l = replaceZeroes(img_l)
        
        c,h,w = img_l.shape 
        # h = int(h//64)*64
        # w = int(w//64)*64
        img_h = torch.from_numpy(img_h[:,:h,:w]).float()
        img_hn = torch.from_numpy(img_hn[:,:h,:w]).float()
        img_l = torch.from_numpy(img_l[:,:h,:w]).float()
        # ratio = torch.from_numpy(ratio_array).float()


        return img_l,img_hn,img_h,img_nf[:,:h,:w].float()

if __name__ == '__main__':
    import torchvision

    nyu_v2_train = NYU_v2_datset(split='test',patch_width = 256,path_height = 256)

    print(nyu_v2_train.__len__())
    for idx,data in enumerate(nyu_v2_train):
        img_l = data[0].numpy()
        img_l = np.transpose(img_l,(1,2,0))*255.0
        img_l = np.clip(img_l,0,255).astype(np.uint8)
        # print('img_l',img_l.max(),img_l.min(),img_l.shape)

        # ratio = data['ratio']
        

        img_h = data[2].numpy()
        img_h = np.transpose(img_h,(1,2,0))*255.0
        img_h = np.clip(img_h,0,255).astype(np.uint8)

        img_hn = data[1].numpy()
        img_hn = np.transpose(img_hn,(1,2,0))*255.0
        img_hn = np.clip(img_hn,0,255).astype(np.uint8)

        if img_h.shape[0] != img_l.shape[0] or img_h.shape[1] != img_l.shape[1]:
            print('check',idx,img_h.shape,img_l.shape)
            input('check')
        print('pass',idx,img_h.shape,img_l.shape)

        # continue 
        print()
        # depth = depth / 10.0
        cv2.imwrite('img_l.png',np.hstack([img_l,img_hn,img_h]))
       
       
        input('c')