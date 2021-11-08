"""Using SEAM approach (normalize CAM, use different features and add second loss).
"""

import torch
import torch.nn as nn
from .resnet import Resnet
from .utils import norm_batch
import torch.nn.functional as F
from .sem_cams import DeeplabDecoder
from torch.nn.functional import cosine_similarity
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.deeplabv3.decoder import DeepLabV3PlusDecoder

class PCMDeeplab(nn.Module):
    """PCM module adapted to work with DeepLab.
    
    Reference: 
    Wang, Y. et al. (2020). Self-supervised equivariant attention 
    mechanism for weakly supervised semantic segmentation. 
    """
    
    def __init__(self, infeat, n_class):
        super().__init__()
        self.infeat = infeat
        self.n_class = n_class
        self.f8_aspp = torch.nn.Conv2d(128, 64, 1, bias = False)
        self.f8_3 = torch.nn.Conv2d(128, 64, 1, bias = False)
        self.f9 = torch.nn.Conv2d(128+4, 128, 1, bias = False)


    def forward(self, aspp, f3, inp, cam):
        # get dims of largest map
        n,c,h,w = cam.size()
        
        # apply 1x1 conv to input features
        f3 = self.f8_3(f3)
        aspp = self.f8_aspp(aspp)
        # upsample aspp to cam size
        aspp = F.interpolate(aspp,(h,w),mode='bilinear',align_corners=True)
        # downsample input to cam size
        x_s = F.interpolate(inp,(h,w),mode='bilinear',align_corners=True)
        # upsample the f3 to the cam size
        f3 = F.interpolate(f3, (h,w), mode='bilinear', align_corners=True)
        # concatenate the different features
        #print('xs', x_s.shape)
        #print('aspp', aspp.shape)
        #print('f3', f3.shape)
        f = torch.cat([x_s, aspp, f3], dim=1)
        # run through a 1x1 conv
        f = self.f9(f)
        
        # find the affinity matrix
        f = f.view(n,-1,h*w)
        f = f/(torch.norm(f,dim=1,keepdim=True)+1e-5)
        aff = F.relu(torch.matmul(f.transpose(1,2), f),inplace=True)
        aff = aff/(torch.sum(aff,dim=1,keepdim=True)+1e-5)
        # apply affinity to the CAM
        cam = cam.view(n,-1,h*w)
        cam_rv = torch.matmul(cam, aff).view(n,-1,h,w)
        
        return cam_rv

class DeepLabForward(SegmentationModel):
    """Determines the forward pass of Deeplab.
    Edited for use with PCM.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Sequentially pass `x` through model`s encoder, decoder and heads"""
        features = self.encoder(x)
        final_feats, second_final_feats = self.decoder(*features)

        return final_feats, second_final_feats, features

class DeeplabDecoderPCM(DeeplabDecoder):
    """Patch of the normal decoder so that it returns multiple 
    levels of feature maps instead of only the final one.
    """
    
    def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-3])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        
        return fused_features, aspp_features

class PCMBackDeepLab(DeepLabForward):
    """Defines the backbone for PCM using DeepLab
    """
    def __init__(self,
                 encoder_name,
                 decoder_channels,
                 encoder_depth  = 5,
                 encoder_weights = None,
                 encoder_output_stride = 16,
                 decoder_atrous_rates = (12, 24, 36),
                 in_channels = 4,
                 classes = 5,
                 upsampling = 4,
                 p_dropout = 0.3):
        
        super().__init__()
        
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )
        
        self.decoder = DeeplabDecoderPCM(
              encoder_channels=self.encoder.out_channels,
              out_channels=decoder_channels,
              atrous_rates=decoder_atrous_rates,
              output_stride=encoder_output_stride)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
class DeepLabModel(nn.Module):
    """Full model of PCM + DeepLab
    """
    def __init__(self, backbone=PCMBackDeepLab, n_class=5, infeat = 128, debug = False):
        """
        Parameters
        ----------
        backbone : nn.Module
            Any network which produces CAMs.
        infeat : int
            The number of feature maps as input to PCM.
            Should be the number of output feature maps
            from the final layer of the backbone network.
        
        """
        super().__init__()
        
        self.backbone = backbone
        self.debug = debug
        self.attention_head = PCMDeeplab(infeat, n_class)
        self.cam_attention = nn.Conv2d(infeat, n_class, 1)
        
    def forward(self, x):
        
        # get feature maps from different layers
        f, aspp, encoder_feats = self.backbone(x)
        
        # use a 1x1 conv to get 5 class maps
        cam = self.cam_attention(f)
        
        # use relu and normalize cam
        n, c, h, w = cam.size()
        cam_d = F.relu(cam.detach())
        cam_d_max = torch.max(cam_d.view(n,c,-1), dim=-1)[0].view(n,c,1,1)+1e-5
        cam_d_norm = F.relu(cam_d-1e-5)/cam_d_max
        
        # use pcm on CAM
        cam_p = self.attention_head(aspp, encoder_feats[-3], x, cam_d_norm)
        
        # normalize the values
        cam_p = norm_batch(cam_p)
        
        # get final logits from the cam
        logits = self.backbone.avgpool(cam)
        logits = logits.squeeze()
        
        # get logits from the PCM cam
        logits_pcm = self.backbone.avgpool(cam_p)
        logits_pcm = logits_pcm.squeeze()
        
        if not self.training:
            return cam, logits, cam_p, f
        else:
            return [logits, logits_pcm]