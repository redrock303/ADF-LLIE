import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, in_channels=3, out_channels=3,nf = 256):
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

        # Downsample path
        x = self.conv_first(inp_img)

        x1 = self.down1(nn.functional.max_pool2d(x, 2))
        x2 = self.down2(nn.functional.max_pool2d(x1, 2))
        x3 = self.down3(nn.functional.max_pool2d(x2, 2))

        # Upsample path
        x = self.up3(x3)
        x = self.up2(x + x2)  # Skip connection
        x = self.up1(x + x1)  # Skip connection
        
        # Final convolution
        out = self.final_conv(x)

        if gt is not None:
            # print(out.shape,gt.shape)
            return dict(L1=self.criterion(out,gt)) 
        else:
            out = out[...,:H,:W]
        return out,out
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


