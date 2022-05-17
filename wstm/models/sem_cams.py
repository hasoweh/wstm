"""Using built in eSEM approach (cosine similarity between feature map pixels).
"""

import torch
import torch.nn as nn
from .utils import esem
from .utils import norm_batch
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.deeplabv3.decoder import ASPP, SeparableConv2d

class DeepLabForward(SegmentationModel):
    """Determines the forward pass of Deeplab.
    Edited for use in SEM/eSEM.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Sequentially pass `x` through model`s encoder, decoder and heads"""
        x = self.encoder(x) # produces list of features
        x = self.decoder(*x) # pass list of features to decoder

        return x

class DeeplabDecoder(nn.Module):
    """Patch of the normal decoder so that it only upsamples by a
    rate of 2. This is to keep the feature maps from being too
    large when using the pixel correlation module for the CAMs.
    """
    
    def __init__(
        self,
        encoder_channels,
        out_channels=256,
        atrous_rates=(12, 24, 36),
        output_stride=16
        ):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))
        
        self.out_channels = out_channels
        self.output_stride = output_stride
        
        self.aspp = nn.Sequential(
            ASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
        # change scale factor to 2 so we get smaller feature map sizes
        scale_factor = 2# if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        highres_in_channels = encoder_channels[-3]
        highres_out_channels = 48   # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-3])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return fused_features
    
class DeepLabBackbone(DeepLabForward):
    
    def __init__(self,
                 encoder_name = 'resnet34',
                 decoder_channels = 128,
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

        self.decoder = DeeplabDecoder(
              encoder_channels=self.encoder.out_channels,
              out_channels=decoder_channels,
              atrous_rates=decoder_atrous_rates,
              output_stride=encoder_output_stride)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
class SEM_DeepLab(nn.Module):
    
    def __init__(self, backbone=DeepLabBackbone, n_class=5, infeat = 128, eval_ = False):
        """
        Parameters
        ----------
        backbone : nn.Module
            Any network which produces CAMs.
        n_class : int
            Number of classes to predict.
        infeat : int
            The number of feature maps as input to PCM.
            Should be the number of output feature maps
            from the final layer of the backbone network.
        eval_ : bool
            Whether we are in evaluation mode. If True, 
            then the network will return CLMs, logits,
            and the final feature maps as an output. 
            If False, then the network will only return logits.
        
        """
        super().__init__()
        
        self.backbone = backbone
        self.eval_ = eval_
        self.esem = esem
        self.cam_attention = nn.Conv2d(infeat, n_class, 1)
        self.n_classes = n_class
        self._calc_esem = True
        
    @property
    def calc_esem(self):
        return self._calc_esem
    
    @calc_esem.setter
    def calc_esem(self, boolean):
        self._calc_esem = boolean
        
    def forward(self, x):
        
        # get feature maps from different layers
        f = self.backbone(x)
        
        # use a 1x1 conv to get 5 class maps
        cam = self.cam_attention(f)
        
        # use sem on CAM only for inference
        if self.eval_ and self.calc_esem:
            cam_p = self.esem(cam, f, self.n_classes)
        
        # get final logits from the cam
        logits = self.backbone.avgpool(cam)
        logits = logits.squeeze()
        
        if self.eval_:
            if self.calc_esem:
                # return orig cam, logits, improved cam (esem), and features
                return norm_batch(cam), logits, cam_p, f
            else:
                # return orig cam, logits, empty tensor, and features
                return norm_batch(cam), logits, torch.zeros_like(cam), f

        else:
            # return preds
            return logits