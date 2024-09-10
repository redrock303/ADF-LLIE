import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2




import sys
sys.path.append('/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM')
from utils.modules.pyramid_lap import PyLap

import utils.modules.SNR.arch_util as arch_util
from utils.modules.SNR.transformer.Models import Encoder_patch66
###############################
class low_light_transformer(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(low_light_transformer, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(24, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)

        self.upconv1 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf*2, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64*2, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 12, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.transformer = Encoder_patch66(d_model=1024, d_inner=2048, n_layers=6)
        self.recon_trunk_light = arch_util.make_layer(ResidualBlock_noBN_f, 6)
        self.window_size = 32

        self.spy_lap = PyLap(3)

        self.criterion = nn.L1Loss(reduction='mean')

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x, inp_s1_img,mask=None,gt=None):

        if gt is None:
            H,W = x.shape[2:]
            x = self.check_image_size(x)
            inp_s1_img = self.check_image_size(inp_s1_img)
            mask = self.check_image_size(mask)


        b,c,h,w = x.size()
        decomposition_ins = self.spy_lap(x)
        decomposition_hn  = self.spy_lap(inp_s1_img)

        high_frequency = []
        for i in range(len(decomposition_ins)):

            if i != 0:
                data_1 = F.interpolate(decomposition_ins[i], size = (h,w), mode='bilinear', align_corners=False)
                data_2 = F.interpolate(decomposition_hn[i], size = (h,w), mode='bilinear', align_corners=False)
            else:
                data_1 = decomposition_ins[i]
                data_2 = decomposition_hn[i]
            
            high_frequency += [data_1,data_2]

        base = decomposition_hn[-1]
        
        ins = torch.cat(high_frequency,1)


        x_center = x
        if mask is None:
            dark = x.clone()
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

        L1_fea_1 = self.lrelu(self.conv_first_1(ins))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))

        fea = self.feature_extraction(L1_fea_3)
        fea_light = self.recon_trunk_light(fea)

        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode='nearest')

        xs = np.linspace(-1, 1, fea.size(3) // 4)
        ys = np.linspace(-1, 1, fea.size(2) // 4)
        xs = np.meshgrid(xs, ys)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(fea.size(0), 1, 1, 1).cuda()
        xs = xs.view(fea.size(0), -1, 2)

        height = fea.shape[2]
        width = fea.shape[3]
        fea_unfold = F.unfold(fea, kernel_size=4, dilation=1, stride=4, padding=0)
        fea_unfold = fea_unfold.permute(0, 2, 1)
        # print('fea_unfold',fea_unfold.shape)

        mask_unfold = F.unfold(mask, kernel_size=4, dilation=1, stride=4, padding=0)
        mask_unfold = mask_unfold.permute(0, 2, 1)
        mask_unfold = torch.mean(mask_unfold, dim=2).unsqueeze(dim=-2)
        mask_unfold[mask_unfold <= 0.5] = 0.0

        fea_unfold = self.transformer(fea_unfold, xs, src_mask=mask_unfold)
        fea_unfold = fea_unfold.permute(0, 2, 1)
        fea_unfold = nn.Fold(output_size=(height, width), kernel_size=(4, 4), stride=4, padding=0, dilation=1)(fea_unfold)

        channel = fea_light.shape[1]
        mask = mask.repeat(1, channel, 1, 1)
        
        # print('fea_unfold',fea_unfold.shape,fea_light.shape,mask.shape)

        fea = fea_unfold * (1 - mask) + fea_light * mask

        out_noise = self.recon_trunk(fea)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))
        out_dec_level1 = self.conv_last(out_noise)


        spy_pred = []
        for i in range(len(decomposition_ins)):
            data = out_dec_level1[:,i*3:(i+1)*3]
            # print(data.shape,decomposition_hn[i].shape)
            if i != 0:
                data = F.interpolate(data, size = (decomposition_hn[i].size()[-2],decomposition_hn[i].size()[-1]), mode='bilinear', align_corners=False) 
                # if gt is None:
                    # print(data.shape,decomposition_hn[i].shape)
                data = data + decomposition_hn[i]
            spy_pred.append(data)
        # input('cc')
        if gt is not None:
            spy_gt = self.spy_lap(gt)
            loss = 0
            for i in range(len(decomposition_ins)):
                loss +=  self.criterion(spy_pred[i], spy_gt[i])

            # loss_ic = self.criterion(spy_pred[-1], decomposition_hn[-1])

            loss_dict = dict(L1=loss) 
            #     print(decomposition_ins[i].shape,spy_gt[i].shape,spy_pred[i].shape)
            # input('cc')

            # ensure illunumator_consistency
            
            # loss_ill0 = self.illu_consistency_loss(gx, gt.detach()) + self.illu_consistency_loss(lx, gt.detach())
            
            # loss_ill1 = self.illu_consistency_loss(out, gt.detach())

            # loss_ic = 1e2 * self.illu_consistency_loss(out_dec_level1, inp_s1_img)
            # loss_ic = self.criterion(self.adaptiveavgpooling(out_dec_level1) , self.adaptiveavgpooling(gt))
            # loss_l1 = self.criterion(out_dec_level1, gt)
            # loss_dict = dict(L1=loss_l1,Lic=loss_ic) # 
            return loss_dict
        else:
            pred = self.spy_lap(spy_pred,inverse=True)
            inp_s1_img_re = self.spy_lap(decomposition_hn,inverse=True)
            return inp_s1_img_re[:, :, :H, :W] ,pred[:, :, :H, :W] 

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    img = torch.Tensor(1, 3, 600, 400).cuda()
    net = low_light_transformer(nf=64, nframes=5,groups=8, front_RBs=1,back_RBs=1, center=None,
                                                           predeblur=True, HR_in=True,
                                                           w_TSA=True).cuda()
    net.eval()
    print('total parameters:', sum(param.numel() for param in net.parameters())/1e6)
    with torch.no_grad():
        _, high = net(img,img,img[:,[0]])


# net = 
# print("backbone have {:.3f}M paramerters in total".format(sum(x.numel() for x in filter(lambda p: p.requires_grad,net.parameters()))/1000000.0))
# ins = torch.randn(1, 3, 512, 512).cuda()
# mask = torch.randn(1,1, 512, 512).cuda()
# output = net(ins,mask)
# print(output.size())
# input('check')