import os
import re
import sys
import glob
import json
import time
import torch
import skimage
import numpy as np
import skimage.io as imgio
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

def crf_inference(probs, img, n_iters=10,
                  sxy_gaussian=(3, 3), compat_gaussian=3,
                  kernel_gaussian=dcrf.DIAG_KERNEL,
                  normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
                  sxy_bilateral=(49, 49), compat_bilateral=4,
                  srgb_bilateral=(5, 5, 5),
                  kernel_bilateral=dcrf.DIAG_KERNEL,
                  normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC,
                  n_classes=11):
    """
    Parameters
    ----------
    probs: tensor, shape = (channel,h,w)
    img: tensor, shape = (h,w,channel)
    """
    # From DSRG original paper, https://github.com/speedinghzl/DSRG/blob/master/training/tools/utils.py
    n_class, h, w = probs.shape

    img = img.contiguous().numpy()
    
    d = dcrf.DenseCRF2D(w, h, n_classes) # Define DenseCRF model.
    U = -np.log(probs) # Unary potential.
    #U = np.reshape(U, (n_classes, -1))
    U = U.reshape((n_classes, -1)) # Needs to be flat.
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        assert(img.shape[0:2] == (h, w)), "The image height and width must come first, channels last."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img)
    Q = d.inference(n_iters)
    Q = np.array(Q, dtype=np.float32)
    preds = Q.reshape((n_classes, h, w))

    return preds
