import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from skimage.segmentation import slic


class CamGenerationUnit(nn.Module):
    def __init__(self, n_feat, n_class):
        super().__init__()

        self.conv = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(n_feat)
        self.relu = nn.ReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(n_feat, n_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x

class MSGSRNet(nn.Module):
    def __init__(self, backbone, n_class, n_bands, return_branch=None):
        super().__init__()
        self.cg1 = CamGenerationUnit(64, n_class) # TODO CHANGE TO cg0, then also can use self.return_branch = return_branch without -1
        self.cg2 = CamGenerationUnit(128, n_class)
        self.cg3 = CamGenerationUnit(256, n_class)
        self.cg4 = CamGenerationUnit(512, n_class)
        self.model = backbone
        
        # determines if we want the output from a specific CGU
        # this is relevant when we do the GradCAM++ calc
        self.return_branch = return_branch - 1
        
        if n_bands != 3:
            self.model.conv1 = nn.Conv2d(n_bands, 64, kernel_size=7, 
                                         stride=2, padding=3,
                                         bias=False)

    def forward(self, x, inference=False):
        logits = []
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        logits.append(self.cg1(x))
        x = self.model.layer2(x)
        logits.append(self.cg2(x))
        x = self.model.layer3(x)
        logits.append(self.cg3(x))
        x = self.model.layer4(x)
        logits.append(self.cg4(x))

        if self.return_branch:
            return logits[self.return_branch]
        else:
            return logits



# get the cam for a single level in the model
def gradcampp(model, target_layer, img, target_class):
    target_layers = [target_layer]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    targets = [ClassifierOutputTarget(target_class)]
    # get CAM for the input image and the target class
    mask = cam(img, targets)

    return mask

def superpixel_cam(cam, segments):
    # need to give the average value of the CAM in each segment
    # super pixels are segmented by having a unique label in each superpixel
    # so get the value of each superpixel label
    labels = np.unique(segments)
    updated_cam = np.zeros(segments.shape)
    for label in labels:
        mask = np.where(segments == label, 1, 0)
        mean_val = mean_nonzero(cam, mask)
        updated_cam[np.where(segments == label)] = mean_val
    return updated_cam

def mean_nonzero(array, mask):
    return np.mean(array[np.where(mask != 0)])