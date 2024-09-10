import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM')
from utils.modules.pyramid_lap import PyLap
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=24, out_channels=12,nf = 256):
        super(UNet, self).__init__()

        self.conv_first = nn.Conv2d(in_channels, nf, kernel_size=3, padding=1)

        self.down1 = DoubleConv(nf, nf)
        self.down2 = DoubleConv(nf, nf*2)
        self.down3 = DoubleConv(nf*2, nf*4)
        
        self.up3 = nn.ConvTranspose2d(nf*4, nf*2, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(nf*2, nf, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(nf, nf, kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(nf, out_channels, kernel_size=1)
        self.criterion = nn.L1Loss(reduction='mean')
        self.window_size = 16

        self.spy_lap = PyLap(3)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, inp_img,inp_s1_img,gt=None):
        if gt is None:
            H,W = inp_img.shape[2:]
            inp_img = self.check_image_size(inp_img)
            inp_s1_img = self.check_image_size(inp_s1_img)

        # add lap decomposion 
        b,c,h,w = inp_img.size()
        decomposition_ins = self.spy_lap(inp_img)
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

        # Downsample path
        x = self.conv_first(ins)

        x1 = self.down1(nn.functional.max_pool2d(x, 2))
        x2 = self.down2(nn.functional.max_pool2d(x1, 2))
        x3 = self.down3(nn.functional.max_pool2d(x2, 2))

        # Upsample path
        x = self.up3(x3)
        x = self.up2(x + x2)  # Skip connection
        x = self.up1(x + x1)  # Skip connection
        
        # Final convolution
        out_dec_level1 = self.final_conv(x)

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

        if gt is not None:
            spy_gt = self.spy_lap(gt)
            loss = 0
            for i in range(len(decomposition_ins)):
                loss +=  self.criterion(spy_pred[i], spy_gt[i])

            loss_ic = self.criterion(spy_pred[-1], decomposition_hn[-1]) 
                #+ self.criterion(spy_pred[-2], decomposition_hn[-2]) + self.criterion(spy_pred[-3], decomposition_hn[-3])  

            # pred = self.spy_lap(spy_pred,inverse=True)
            # loss_ic = self.fft_loss(spy_pred[-1], decomposition_hn[-1])*1e2

            loss_dict = dict(L1=loss,loss_ic=loss_ic) 

            return loss_dict
        else:
            pred = self.spy_lap(spy_pred,inverse=True)
            inp_s1_img_re = self.spy_lap(decomposition_hn,inverse=True)
            return inp_s1_img_re[:, :, :H, :W] ,pred[:, :, :H, :W] 
if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    img = torch.Tensor(1, 3, 600, 400).cuda()
    net = UNet().cuda()
    net.eval()
    print('total parameters:', sum(param.numel() for param in net.parameters())/1e6)
    with torch.no_grad():
        _, high = net(img,img)
    print(high.shape)


