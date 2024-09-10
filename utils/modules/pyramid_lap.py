import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

import cv2
import numpy as np

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        self.kernel_size = kernel_size[0]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        pad = self.kernel_size  // 2
        input = F.pad(input,(pad,pad,pad,pad),mode='reflect')
        return self.conv(input, weight=self.weight, groups=self.groups)


class IllunimationConsistency(torch.nn.Module):
    def __init__(self,kernel_size=51,type='l2'):
        super(IllunimationConsistency,self).__init__()
        self.kernel_size = kernel_size
        # print('kernel_size',self.kernel_size)
        self.gaussian = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=0.1*kernel_size)
    def down_smooth(self,x,scale=0.25):
        b,c,h,w = x.size()
        x_pack = x.contiguous().view(-1,1,h,w)

        x_pack_blur = self.gaussian(x_pack)
        x_pack_blur = F.interpolate(x_pack_blur, scale_factor=scale, mode='bilinear', align_corners=False)
        return x_pack_blur
    def forward(self,x,y,scale=0.25):
        x_ds = self.down_smooth(x,scale)
        y_ds = self.down_smooth(y,scale)
        # print(x.shape,y.shape,x_ds.shape,y_ds.shape)
        return ((x_ds - y_ds)**2).mean()

class PyLap(torch.nn.Module):
    def __init__(self,num_levels=4):
        super(PyLap,self).__init__()
        self.num_levels = num_levels 
        self.gaussian = GaussianSmoothing(1,11,1)

        nograd = [self.gaussian]
        for module in nograd:
            for param in module.parameters():
                param.requires_grad=False

    def _smooth(self,x):
        b,c,h,w = x.size()
        x_pack = x.view(-1,1,h,w)

        x_pack_blur = self.gaussian(x_pack) 
        return x_pack_blur.view(b,-1,h,w)
    def forward(self,data,inverse=False):
        if not inverse:
            pyramid = []
            current_level = data 
            for level in range(self.num_levels):
                
                blurred = F.interpolate(current_level, scale_factor=0.5, mode='bilinear', align_corners=False)
                blurred = self._smooth(blurred)
                upsampled = F.interpolate(blurred, size=current_level.shape[2:], mode='bilinear', align_corners=False)
                residual = current_level - upsampled
                pyramid.append(residual)
                current_level = blurred

            pyramid.append(current_level)  # Add the lowest resolution image to the pyramid
            return pyramid
        else:
            restorer_x = data[-1]
            for level in range(len(data)-2,-1,-1):
                # print(restorer_x.shape)


                restorer_x = F.interpolate(restorer_x, size=data[level].shape[2:], mode='bilinear', align_corners=False)

                # if level >0:
                restorer_x += data[level] 
            return restorer_x

class MSRLayer(torch.nn.Module):
    def __init__(self,kernel_size):
        super(MSRLayer,self).__init__()
        self.kernel_size = kernel_size
        # print('kernel_size',self.kernel_size)
        self.gaussian = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=0.1*kernel_size)
    def forward(self,x):
        b,c,h,w = x.size()
        x = x.masked_fill(x <1e-3, 1e-3)

        x_pack = x.view(-1,1,h,w)

        

        x_pack_blur = self.gaussian(x_pack)
        # print('x_pack',x_pack.shape,x_pack.max(),x_pack.mean(),x_pack.min())
        # print('x_pack_blur',x_pack_blur.shape,x_pack_blur.max(),x_pack_blur.mean(),x_pack_blur.min())
        dst_Img = torch.log(x_pack)
        # print('dst_Img ',dst_Img.shape,dst_Img.max(),dst_Img.mean(),dst_Img.min())
        
        dst_lblur = torch.log(x_pack_blur)
        # print('dst_lblur ',dst_lblur.shape,dst_lblur.max(),dst_lblur.mean(),dst_lblur.min())
        dst_Ixl = x_pack * x_pack_blur
        delta_i = dst_Img - dst_Ixl


        # input('cc')
        delta_i = delta_i.view(b*c,-1)

        outmap_min,_ = torch.min(delta_i, dim=1, keepdim=True)
        outmap_max,_ = torch.max(delta_i, dim=1, keepdim=True)
        # # print('outmap_min',outmap_min)
        delta_i = (delta_i - outmap_min) / (outmap_max - outmap_min)  #normalization 

        return delta_i.view(b,c,h,w)
        
class GradientLoss(nn.Module):
    def __init__(self,weight=0.1):
        super(GradientLoss,self).__init__()
        self.weight = weight
        self.criterion = nn.L1Loss(reduction='mean')
    def cal_gradient(self,data):

        pad_y = torch.nn.functional.pad(data,(0,0,1,0),mode='reflect')
        pad_x = torch.nn.functional.pad(data,(0,1,0,0),mode='reflect')

        data_gy = pad_y[:,:,1:] - pad_y[:,:,:-1]
        data_gx = pad_x[:,:,:,1:] -  pad_x[:,:,:,:-1]

        # print(data_gx.shape,data_gy.shape)
        return data_gx,data_gy

    def forward(self,pred,gt):
        # shape b 1 h w 
        pred_gx,pred_gy = self.cal_gradient(pred)
        gt_gx,gt_gy = self.cal_gradient(gt)

        gx_loss = self.criterion(pred_gx,gt_gx) + self.criterion(pred_gy,gt_gy)
        return self.weight * gx_loss