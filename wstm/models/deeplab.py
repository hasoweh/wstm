import torch.nn as nn

from .resnet import Resnet
from typing import Optional
from wstm.models.utils import sem
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.deeplabv3.decoder import DeepLabV3Decoder, DeepLabV3PlusDecoder
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead

class SegModelClassif(SegmentationModel):
    """Determines the forward pass of Deeplab.
    Edited to use only a classifier head for Deeplab
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Sequentially pass `x` through model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        labels, cam = self.classification_head(decoder_output)
        if not self.training:
            sem_cam = sem(cam.detach().cpu(), decoder_output.detach().cpu(), self.classes)
        else:
            sem_cam = sem(cam, decoder_output, self.classes)
        
        if self.training:
            return labels
        else:
            return labels, sem_cam
        
class DeepLabClassif(SegModelClassif):
    """DeepLabV3+ with only a classification head.
    
    Reference:
        https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/deeplabv3/model.py
    """
    def __init__(
            self,
            encoder_name,
            decoder_channels: int = 256,
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = None,
            encoder_output_stride: int = 16,
            decoder_atrous_rates: tuple = (12, 24, 36),
            in_channels: int = 4,
            classes: int = 5,
            upsampling: int = 4,
            p_dropout: float = 0.3
            ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels= self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride= encoder_output_stride
        )

        self.classification_head = ClassificationHead(
            in_channels=self.decoder.out_channels, 
            classes = classes, 
            dropout = p_dropout, 
            activation = None
        )

class DeepLabACoL(DeepLabClassif):
    """Switches the classification head from using FC layer to
    using instead a set of 1x1 convs to get class activation maps.
    """
    def __init__(self,
                 encoder_name,
                 decoder_channels: int = 256,
                 encoder_depth: int = 5,
                 encoder_weights: Optional[str] = None,
                 encoder_output_stride: int = 16,
                 decoder_atrous_rates: tuple = (12, 24, 36),
                 in_channels: int = 4,
                 classes: int = 5,
                 upsampling: int = 4,
                 p_dropout: float = 0.3):
        
        super().__init__(encoder_name,
                         decoder_channels,
                         encoder_depth,
                         encoder_weights,
                         encoder_output_stride,
                         decoder_atrous_rates,
                         in_channels,
                         classes,
                         upsampling,
                         p_dropout)
        
        self.classes = classes
        
        self.classification_head = ACoLHead(
            in_channels=self.decoder.out_channels, 
            classes = classes, 
            dropout = p_dropout
        )
        
        
class ACoLHead(nn.Module):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2):
        super().__init__()
        
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
            
        self.attention = nn.Conv2d(in_channels=in_channels, 
                                   out_channels=classes, 
                                   kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        
        
        
    def forward(self, x):
        
        cam = self.attention(x)
        out = self.pool(cam).squeeze()

        return out,cam
        