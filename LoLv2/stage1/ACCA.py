import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import math

from timm.models.layers import trunc_normal_
from blocks import CBlock_ln, SwinTransformerBlock
from global_net import Global_pred


class SelfIGBlock(torch.nn.Module):
    def __init__(self,feat0 = 16,feat1=8,window_size = 4):
        super(SelfIGBlock,self).__init__()
        self.window_size = window_size

        self.conv_first = nn.Conv2d(feat0,feat0,3,1,1)

        self.conv_x = nn.Conv2d(feat0,window_size*2,window_size+1,padding = (window_size)//2,stride=window_size,groups=window_size)
        # self.conv_y = nn.Conv2d(nf,window_size,window_size+1,padding = (window_size)//2,stride=window_size,groups=window_size)
        
        self.conv_c = nn.Conv2d(feat0,feat0,window_size+1,padding = (window_size)//2,stride=window_size,groups=feat0)
        

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU()

        # self.softmax = nn.Softmax(-1)
        self.norm1 = nn.LayerNorm(feat0)
        # self.relu = nn.ReLU()
    def forward(self,x):
        b,c,h,w = x.size()
        x = self.lrelu(self.conv_first(x))
        # y_in = nn.functional.interpolate(y, size=(h,w), mode='bilinear', align_corners=False)
        # print(x.shape,y_in.shape)

        w_xy = self.conv_x(x).permute(0,2,3,1).contiguous().view(-1,self.window_size*2)
        w_x,w_y =w_xy[:,:self.window_size],w_xy[:,self.window_size:]  # b window_size**2, nh nw

        nh,nw = h // self.window_size,w//self.window_size
        x_reshape = x.view(b,c,nh,self.window_size,nw,self.window_size) .permute(0,2,4,3,5,1).contiguous().view(-1,self.window_size**2,c)
        # print('x_reshape',x_reshape.shape)
        x_reshape = self.norm1(x_reshape)

        atten = torch.einsum('bm,bn->bmn',F.normalize(w_x,p=2,dim=-1),F.normalize(w_y,p=2,dim=-1)).view(-1,self.window_size**2)
        # print('atten',atten.shape)
        atten = self.relu(atten)

        w_c = self.conv_c(x).permute(0,2,3,1).contiguous().view(-1,self.window_size*2)
        # print(atten.shape,w_c.shape)
        atten = torch.einsum('bn,bc->bnc',atten,F.normalize(w_c,p=2,dim=-1))


        out = torch.einsum('bnc,bnc->bnc',atten,x_reshape).view(b,nh,nw,self.window_size,self.window_size,c).contiguous().permute(0,5,1,3,2,4)
        out = out.contiguous().view(b,c,h,w)
        # print('out',out.shape)
        return out 

class Local_pred(nn.Module):
    def __init__(self, dim=16, number=4, type='ccc'):
        super(Local_pred, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(3, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type =='ccc':  
            #blocks1, blocks2 = [block for _ in range(number)], [block for _ in range(number)]
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type =='ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type =='cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        #    block1 = [CBlock_ln(16), nn.Conv2d(16,24,3,1,1)]

        blocks1 = [SelfIGBlock(dim),SelfIGBlock(dim),SelfIGBlock(dim)]
        blocks2 = [SelfIGBlock(dim),SelfIGBlock(dim),SelfIGBlock(dim)]

        self.mul_blocks = nn.Sequential(*blocks1, nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_blocks = nn.Sequential(*blocks2, nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())


    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        mul = self.mul_blocks(img1)
        add = self.add_blocks(img1)

        return mul, add

# Short Cut Connection on Final Layer
class Local_pred_S(nn.Module):
    def __init__(self, in_dim=3, dim=8, number=4, type='ccc'):
        super(Local_pred_S, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type =='ccc':
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type =='ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type =='cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        
        blocks1 = [SelfIGBlock(dim)]
        blocks2 = [SelfIGBlock(dim)]
        
        #    block1 = [CBlock_ln(16), nn.Conv2d(16,24,3,1,1)]
        self.mul_blocks = nn.Sequential(*blocks1)
        self.add_blocks = nn.Sequential(*blocks2)

        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
            
            

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        # short cut connection
        mul = self.mul_blocks(img1) + img1
        add = self.add_blocks(img1) + img1
        mul = self.mul_end(mul)
        add = self.add_end(add)

        return mul, add

class ACCA(nn.Module):
    def __init__(self, in_dim=3, with_global=True, type='lol'):
        super(ACCA, self).__init__()
        #self.local_net = Local_pred()
        
        self.local_net = Local_pred_S(in_dim=in_dim)

        self.with_global = with_global
        if self.with_global:
            self.global_net = Global_pred(in_channels=in_dim,out_channels=8, type=type)
        self.window_size = 8
    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def forward(self, img_low):
        H,W = img_low.size()[-2:]

        img_low = self.check_image_size(img_low)
        #print(self.with_global)
        mul, add = self.local_net(img_low)
        img_high = (img_low.mul(mul)).add(add)

        if not self.with_global:
            return mul, add, img_high
        
        else:
            gamma, color = self.global_net(img_low)
            b = img_high.shape[0]
            img_high = img_high.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
            img_high = torch.stack([self.apply_color(img_high[i,:,:,:], color[i,:,:])**gamma[i,:] for i in range(b)], dim=0)
            img_high = img_high.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)
            return mul, add, img_high[...,:H,:W]


if __name__ == "__main__":
    import time 
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    img = torch.Tensor(1, 3, 512, 512).cuda()
    net = IAT().cuda()
    print(net)
    print('total parameters:', sum(param.numel() for param in net.parameters()))

    N = 200 
    st = time.time()
    for _ in range(N):
        _, _, high = net(img)
    print((time.time() - st)/N)