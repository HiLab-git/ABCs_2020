from __future__ import print_function, division
from configparser import BasicInterpolation
# from nnUNet.nnunet.network_architecture.HUNet_SSE_sig_new import HUnet_SSE_sig_new
from pickle import FRAME
from numpy.core.multiarray import concatenate
from numpy.lib.utils import deprecate

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from nnunet.network_architecture.neural_network import SegmentationNetwork
import SimpleITK as sitk
import numpy as np
import os
from matplotlib import pyplot as plt 

norm_op_kwargs = {'eps': 1e-5, 'affine': True}
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self,in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=[3, 3,1],padding=[1,1,0]),
            nn.InstanceNorm3d(out_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),

            nn.Conv3d(out_channels, out_channels, kernel_size=[1, 1,3],padding=[0,0,1]),
            nn.InstanceNorm3d(out_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),

            nn.Conv3d(out_channels, out_channels, kernel_size=[3, 3,1],padding=[1,1,0]),
            nn.InstanceNorm3d(out_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),

            nn.Conv3d(out_channels, out_channels, kernel_size=[3, 3,1],padding=[1,1,0]),
            nn.InstanceNorm3d(out_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),            
          #   nn.Dropout(**dropout_op_kwargs),
            nn.Conv3d(out_channels, out_channels, kernel_size=[1, 1,3],padding=[0,0,1]),
            nn.InstanceNorm3d(out_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),

            nn.Conv3d(out_channels, out_channels, kernel_size=[3, 3,1], padding=[1,1,0]),
            nn.InstanceNorm3d(out_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs)            
        )
       
    def forward(self, x):
        x = self.conv_conv(x)
        return  x


class Basic1 (nn.Module):
    """ Basic conv Block"""
    def __init__(self,in_channels,out_channels):
          super(Basic1, self).__init__()
          self.conv_hr = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=[3, 3,1], padding=[1,1,0]),
            nn.InstanceNorm3d(out_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),

            nn.Conv3d(out_channels, out_channels, kernel_size=[1, 1,3], padding=[0,0,1]),
            nn.InstanceNorm3d(out_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),

            nn.Conv3d(out_channels, out_channels, kernel_size=[3, 3,1], padding=[1,1,0]),
            nn.InstanceNorm3d(out_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),

            nn.Conv3d(out_channels, out_channels, kernel_size=[3, 3,1], padding=[1,1,0]),
            nn.InstanceNorm3d(out_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),

            nn.Conv3d(out_channels, out_channels, kernel_size=[1, 1,3],padding=[0,0,1]),
            nn.InstanceNorm3d(out_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),

            nn.Conv3d(out_channels, out_channels, kernel_size=[3, 3,1], padding=[1,1,0]),
            nn.InstanceNorm3d(out_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs)
          )
    def forward(self, x):
          x = self.conv_hr(x)
          return x


class Basic (nn.Module):
    """ Basic conv Block"""
    def __init__(self,in_channels):
          super(Basic, self).__init__()
          self.conv_hr = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=[3, 3,1], padding=[1,1,0]),
            nn.InstanceNorm3d(in_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),

            nn.Conv3d(in_channels, in_channels, kernel_size=[1, 1,3], padding=[0,0,1]),
            nn.InstanceNorm3d(in_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),
            nn.Conv3d(in_channels, in_channels, kernel_size=[3, 3,1], padding=[1,1,0]),
            nn.InstanceNorm3d(in_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),

            nn.Conv3d(in_channels, in_channels, kernel_size=[3, 3,1], padding=[1,1,0]),
            nn.InstanceNorm3d(in_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),

            nn.Conv3d(in_channels, in_channels, kernel_size=[1, 1,3],padding=[0,0,1]),
            nn.InstanceNorm3d(in_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),

            nn.Conv3d(in_channels, in_channels, kernel_size=[3, 3,1], padding=[1,1,0]),
            nn.InstanceNorm3d(in_channels,**norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs)
          )

       
    def forward(self, x):
          x = self.conv_hr(x)
          return x


class DownBlock(nn.Module):
    """Downsampling before ConvBlock"""
    def __init__(self, in_channels, out_channels,pool_num=[2,2,2]):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(pool_num),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class DownBlock_Con(nn.Module):
    """Downsampling """
    def __init__(self, pooling_p):
        super(DownBlock_Con, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(pooling_p),
            # nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.maxpool_conv(x)     

class UpBlock_Con(nn.Module):
    """Upampling """
    def __init__(self,scale_factor=2):
        super(UpBlock_Con, self).__init__()
        self.uppool_conv = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True),
        )

    def forward(self, x):
        return self.uppool_conv(x) 

class Concatenate(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(Concatenate, self).__init__()
        self.conv = ConvBlock(in_channels , out_channels)
    def forward(self, x1, x2):
          x = torch.cat([x2, x1], dim=1)
          return self.conv(x)


class SpatialSELayer3D_0(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
        
    """
    def __init__(self, in_channels ):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D_0, self).__init__()
        self.conv = nn.Conv3d(in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, input_tensor,input_tensor_concat, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor_concat.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)
        
        # spatial excitation
        output_tensor = torch.mul(
            input_tensor_concat, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor

class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
        
    """

    def __init__(self, in_channels ):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

        

    def forward(self, input_tensor,input_tensor_concat, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor_concat.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)
        
        # spatial excitation
        output_tensor = torch.mul(
            input_tensor_concat, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor



class Concatenate_Threechs(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(Concatenate_Threechs, self).__init__()
        self.conv = ConvBlock(in_channels , out_channels)
    def forward(self, x1, x2,x3):
          x = torch.cat([x3,x2, x1], dim=1)
          return self.conv(x)

class UpBlock(nn.Module):
    """Upssampling before ConvBlock"""
    def __init__(self, in_channels, out_channels,scale_factor=(2,2,2),
                 trilinear=True):
        super(UpBlock, self).__init__()
        self.trilinear = trilinear
        if trilinear:
            self.uppool_conv = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='trilinear',align_corners=True),
            ConvBlock(in_channels, out_channels)
            )

    def forward(self, x):
        x = self.uppool_conv(x)
        return x

class HMRNet_s(SegmentationNetwork):
    """ HMRNet-s is designed for slice-like anatomical structures. 
    we replace a standard 3D convolution layer witha spatially se-
    parable convolution that contains two intra-slice convolutions 
    (3×3×1 kernels) and one inter-slice convolution (1×1×3 kernel)."""
    def __init__(self, params):
        super(HMRNet_s, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.fr_chs = self.params['fr_feature_chns']
        self.num_classes   = self.params['class_num']
        self.trilinear = self.params['trilinear']
        self.conv_op = self.params['con_op']
        self._deep_supervision = self.params['_deep_supervision']
        self.do_ds = self.params['do_ds']

        assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

        self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0])
        self.fr_in_conv =ConvBlock(self.in_chns,self.fr_chs)
        self.sse0 = SpatialSELayer3D_0(self.ft_chns[0])

        self.basic_conv1 = Basic1(self.fr_chs,self.fr_chs)    #full _resolution conv
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1])   # Unet  feature down sample
        self.basic_down1 = DownBlock_Con( pooling_p=2)          # full_resolution down sample
        self.basic_up1 = UpBlock_Con(scale_factor=2)            #Unet  feature up sample
        self.concat1 =  Concatenate(self.ft_chns[1]+self.fr_chs,self.ft_chns[1]) 
        """ BFC block"""
        self.fr_concat1 =Concatenate(self.ft_chns[1]+self.fr_chs,self.fr_chs)
        self.sse_up1 = SpatialSELayer3D(self.ft_chns[1])   
        self.sse_down1 = SpatialSELayer3D(self.fr_chs)


        self.basic_conv2 = Basic(self.fr_chs)
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2])
        self.basic_down2 = DownBlock_Con( pooling_p=4)         
        self.basic_up2 = UpBlock_Con( scale_factor=4)            
        self.concat2 =  Concatenate(self.ft_chns[2]+self.fr_chs,self.ft_chns[2])
        """ BFC block"""
        self.fr_concat2 =Concatenate(self.ft_chns[2]+self.fr_chs,self.fr_chs)
        self.sse_up2 = SpatialSELayer3D(self.ft_chns[2])
        self.sse_down2 = SpatialSELayer3D(self.fr_chs)
        


        self.basic_conv3 = Basic(self.fr_chs)
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3],pool_num=[2,2,1])
        self.basic_down3 = DownBlock_Con( pooling_p=[8,8,4])          
        self.basic_up3 = UpBlock_Con(scale_factor=(8,8,4))        
        self.dpvision_con3 = nn.Conv3d(self.ft_chns[3], self.num_classes, kernel_size=1)    #deep supervison 
        self.concat3 =  Concatenate(self.ft_chns[3]+self.fr_chs,self.ft_chns[3])
        """ BFC block"""
        self.fr_concat3 =Concatenate(self.ft_chns[3]+self.fr_chs,self.fr_chs)
        self.sse_up3 = SpatialSELayer3D(self.ft_chns[3])
        self.sse_down3 = SpatialSELayer3D(self.fr_chs)




        if(len(self.ft_chns) == 5):
          self.basic_conv4 = Basic(self.fr_chs)  
          self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4])
          self.basic_down4 = DownBlock_Con( self.fr_chs,self.ft_chns[4],pooling_p=16)        
          self.basic_up4 = UpBlock_Con(self.ft_chns[4],self.fr_chs, scale_factor=16)            
          self.concat4 =  Concatenate(self.ft_chns[4])
          self.fr_concat4 =Concatenate(self.fr_chs)
          self.sse_up4 = SpatialSELayer3D(self.ft_chns[4])
          self.sse_down4 = SpatialSELayer3D(self.fr_chs)


          self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], 
            trilinear=self.trilinear) 

        
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], scale_factor =(2,2,1),
            trilinear=self.trilinear) 
        self.basic_conv5 = Basic(self.fr_chs)
        self.basic_down5 = DownBlock_Con( pooling_p=4)         
        self.basic_up5 = UpBlock_Con(scale_factor=4)           
        self.dpvision_con5 = nn.Conv3d(self.ft_chns[2], self.num_classes, kernel_size=1)    #deep supervison
        self.concat5 =  Concatenate_Threechs(2*self.ft_chns[2]+self.fr_chs,self.ft_chns[2])
        self.fr_concat5 =Concatenate(self.ft_chns[2]+self.fr_chs,self.fr_chs)
        """ BFC block"""
        self.sse_up5 = SpatialSELayer3D(self.ft_chns[2])
        self.sse_down5 = SpatialSELayer3D(self.fr_chs)

        
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], 
            trilinear=self.trilinear) 
        self.basic_conv6 = Basic(self.fr_chs)
        self.basic_down6 = DownBlock_Con(pooling_p=2)       
        self.basic_up6 = UpBlock_Con(scale_factor=2)         
        self.dpvision_con6 = nn.Conv3d(self.ft_chns[1], self.num_classes, kernel_size=1)    #deep supervison
        self.concat6 =  Concatenate_Threechs(2*self.ft_chns[1]+self.fr_chs,self.ft_chns[1])
        self.fr_concat6 =Concatenate(self.ft_chns[1]+self.fr_chs,self.fr_chs)
        """ BFC block"""
        self.sse_up6 = SpatialSELayer3D(self.ft_chns[1])
        self.sse_down6 = SpatialSELayer3D(self.fr_chs)


        self.up4 = UpBlock(self.ft_chns[1],self.ft_chns[0], 
            trilinear=self.trilinear) 
        self.basic_conv7 = Basic(self.fr_chs)   
        self.concat7 =  Concatenate_Threechs(2*self.ft_chns[0]+self.fr_chs,self.ft_chns[0]) 
        self.fr_concat7 =Concatenate(self.ft_chns[0]+self.fr_chs,self.fr_chs)
        self.final_concat = Concatenate(self.ft_chns[0]+self.fr_chs,self.ft_chns[0])
        """ BFC block"""
        self.sse7 = SpatialSELayer3D_0(self.fr_chs)
        self.fr_sse7 = SpatialSELayer3D_0(self.ft_chns[0])

    
        self.out_conv = nn.Conv3d(self.ft_chns[0], self.num_classes,  
            kernel_size = 3, padding = 1)
        self.softmax = lambda x: F.softmax(x, 1)
        self.i =1

    def forward(self, x):
        segout = []
        upscale_logits_ops = []
        x0 = self.in_conv(x)
        fr_x0 = self.fr_in_conv(x)

        x1 = self.down1(x0) # conv  + down sample
        x11 =  self.basic_up1(x1) #conv  + up sample
        fr_x1 = self.basic_conv1(fr_x0)  #conv
        fr_x11 = self.basic_down1(fr_x1)  #conv  + down sample
        x1 = self.concat1(x1,fr_x11) 
        
        x1 = self.sse_down1(fr_x11,x1) # BFC  in  multi-resolution branch
        fr_x1 = self.fr_concat1(fr_x1,x11)
        fr_x1 =  self.sse_up1(x11,fr_x1) # BFC in  full resolution branch


        x2 = self.down2(x1)
        x22 =  self.basic_up2(x2)
        fr_x2 = self.basic_conv2(fr_x1)
        fr_x22 = self.basic_down2(fr_x2)
        x2 = self.concat2(x2,fr_x22)
        x2 = self.sse_down2(fr_x22,x2)
        fr_x2 = self.fr_concat2(fr_x2,x22)
        fr_x2 =  self.sse_up2(x22,fr_x2)
        


        x3 = self.down3(x2)
        x33 =  self.basic_up3(x3)
        fr_x3 = self.basic_conv3(fr_x2)
        fr_x33 = self.basic_down3(fr_x3)
        x3 = self.concat3(x3,fr_x33)
        x3 = self.sse_down3(fr_x33,x3)
        deep_x3 = self.dpvision_con3(x3)
        segout.append(deep_x3)
        fr_x3 = self.fr_concat3(fr_x3,x33)                    
        fr_x3 =  self.sse_up3(x33,fr_x3)


        if(len(self.ft_chns) == 5):
          x4 = self.down4(x3)
          x44 =  self.basic_up4(x4)
          fr_x4 = self.basic_conv4(fr_x3)
          fr_x44 = self.basic_down4(fr_x4)
          x4 = self.concat4(x4,fr_x44)
          fr_x4 = self.fr_concat4(fr_x4,x44)

          x = self.up1(x4, x3)

        else:
          x = x3
          fr_x = fr_x3


        x5= self.up2(x)
        x55 = self.basic_up5(x5)
        fr_x5 = self.basic_conv5(fr_x)
        fr_x55 = self.basic_down5(fr_x5)
        x5 = self.concat5(x5,fr_x55,x2)
        x5 = self.sse_down5(fr_x55,x5)
        deep_x5 = self.dpvision_con5(x5)
        segout.append(deep_x5)        
        fr_x5 = self.fr_concat5(fr_x5,x55)
        fr_x5 =  self.sse_up5(x55,fr_x5)        

        x6 = self.up3(x5)
        x66 = self.basic_up6(x6)
        fr_x6 = self.basic_conv6(fr_x5)
        fr_x66 = self.basic_down6(fr_x6)
        x6 = self.concat6(x6,fr_x66,x1)
        x6 = self.sse_down6(fr_x66,x6)
        deep_x6 = self.dpvision_con6(x6)
        segout.append(deep_x6)        
        fr_x6= self.fr_concat6(fr_x6,x66) 
        fr_x6 =  self.sse_up6(x66,fr_x6)


        x7 = self.up4(x6)
        x7_ =x7
        fr_x7 = self.basic_conv7(fr_x6)
        x7 = self.concat7(x7,fr_x7,x0)
        x7 = self.sse7(fr_x7,x7)
        fr_x7 = self.fr_concat7(fr_x7,x7)
        fr_x7 =  self.fr_sse7(x7_,fr_x7)

        x7 = self.final_concat(x7,fr_x7)
        
        output = self.out_conv(x7)
        segout.append(output)
        for i in range(len(segout)):
             upscale_logits_ops.append(lambda x: x)
        
        if self._deep_supervision and self.do_ds:
               return tuple([segout[-1]] + [i(j) for i, j in
                                              zip(list(upscale_logits_ops)[::-1], segout[:-1][::-1])])
          
        else:
          return segout[-1]


        return output

if __name__ == "__main__":
    params = {'in_chns':3,
              'class_num': 2,
              'feature_chns':[16, 32, 64, 128],
              'fr_feature_chns':8,
              'trilinear': True,
                            '_deep_supervision':True,
              'do_ds':True,
              'con_op':nn.Conv3d}
    Net = HMRNet_s(params)
#     Net = Net.double()
    Net = Net.cuda()

    x  = np.random.rand(2, 3, 40, 40, 40)
    xt = torch.from_numpy(x).float()
    xt = torch.tensor(xt).cuda()
    
    y = Net(xt)
#     y = y.cpu()
    print('11',y[0].shape,'done')
    # print(Net)
