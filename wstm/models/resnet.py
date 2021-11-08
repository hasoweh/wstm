import re
import torch
import torch.nn as nn
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

def set_parameter_requires_grad(model, unfreeze):
    # helper function when unfreezing pre-trained weights
    for name, param in model.named_parameters():
        if name not in unfreeze:
            param.requires_grad = False

class Resnet(nn.Module):
    """Slightly modified version of Resnet adding dropout and
    the ability to learn class specific feature maps.
    
    Parameters
    ----------
    base_model : torchvision.models.resnet
        Some variant of Resnet from the torchvision library.
    n_classes : int
        Number of classes for classification.
    n_bands : int
        Number of image bands for input.
    p_dropout : float
        The probability of dropout.
    bias : optional, torch.tensor
        Can provide a tensor as a bias for the fully connected classifier.
    unfreeze : bool
        If we have pre-trained weights, we can choose to unfreeze.
    headless : bool
        If in headless mode the model's forward method will output
        only the final extracted features. Thus, acts as a feature 
        extractor rather than a classifier.
    use_attention : bool
        If True, the model will not use any fully connected layers in 
        the classifier head. Rather, it will use a 1x1 convolution
        in order to learn a class specific feature map. For more 
        information, see: 
        Zhang, X., et al. (2018). Adversarial complementary 
        learning for weakly supervised object localization.
        
    """
    def __init__(self, 
                 base_model, 
                 n_classes, 
                 n_bands = 4, 
                 p_dropout = 0.25, 
                 bias = None,
                 unfreeze = False,
                 headless = False,
                 use_attention = True):

        super().__init__()
        
        self.headless = headless
        self.use_attention = use_attention
        resnet = base_model
        
        # unfreeze pretrained parameters
        if unfreeze:
            set_parameter_requires_grad(resnet, unfreeze)
            
        self.dropout = nn.Dropout(p = p_dropout)

        if n_bands != 3:
            resnet.conv1 = nn.Conv2d(n_bands, 
                                     64, 
                                     kernel_size=7, 
                                     stride=2, 
                                     padding=3, 
                                     bias=False)
        
        
        if self.use_attention:
            self.attention = nn.Conv2d(in_channels=resnet.fc.in_features, 
                                       out_channels=n_classes, 
                                       kernel_size=1)
            del resnet.fc
        else:
            resnet.fc = nn.Linear(
                      in_features = resnet.fc.in_features, 
                      out_features = n_classes
            )
        # provide bias init
        if bias is not None:
            assert isinstance(bias, torch.Tensor), 'bias must be tensor'
            resnet.fc.bias = nn.Parameter(bias)
            
        # add some attributes for compatibility w/ segm_model_pytorch pkg
        self.in_channels = n_bands
            
        self.model = resnet
        
    
    @property
    def depth(self):
        '''Count the number of times we downsample the image'''
        names = []
        count = 0
        r = re.compile('^.*downsample$')
        # look for modules named downsample or maxpool
        for name, layer in self.model.named_modules():
            names.append(name)
            if r.match(name) or name == 'maxpool':
                count += 1
        # check if first conv uses stride of 2
        if self.model.conv1.stride[0] == 2:
            count += 1
            
        return count
        
    
    @property
    def out_channels(self):
        channels = [int(self.model.inplanes / 2**j) for j in range(self.depth-1)]
        channels.reverse()
        return channels
        
    
    def forward(self, x):
        res = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.dropout(x)
        
        x = self.model.layer2(x)
        x = self.dropout(x)
        
        x = self.model.layer3(x)
        x = self.dropout(x)
        
        x = self.model.layer4(x)
        x = self.dropout(x)

        # headless mode uses resnet as feature extractor
        if not self.headless:
            if self.use_attention:
                x = self.attention(x)
                # gives the per-class logit scores
                x = self.model.avgpool(x).squeeze()
            else:
                x = self.model.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.model.fc(x)
                
        return x
