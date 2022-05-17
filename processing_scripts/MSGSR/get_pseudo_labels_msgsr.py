import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from wstm.models.MSGSRNet import *
from torchvision.models import resnet18
from wstm.utils.camutils import save, crs
from wstm.models import get_classification_model
from wstm.utils.dataloader import get_dataloader
from sklearn.metrics import (f1_score, jaccard_score, precision_score, 
                             recall_score, confusion_matrix, accuracy_score)

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from skimage.segmentation import slic

def get_cg_unit_from_model(layer_num, classes, n_bands, dev, weights):
    """Because MSGSR Net uses 4 different CG Units, we need to
    select the gradient for the specific one.
    """
    model = MSGSRNet(resnet18(), len(classes), n_bands, layer_num)

    # load model weights and send to device
    device = torch.device(dev)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    model.to(device)

    layer_name = "cg%d.relu" % layer_num
    layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            layer = module
            break
            
    if not layer:
        m = "Couldn't find any layer with the name %s in the model"
        raise AssertionError(m % layer_name)
        
    return model, layer

def get_multi_scale_cams(img, 
                         target, 
                         layer_num, 
                         classes,
                         n_bands, 
                         device, 
                         weights):
    model, layer = get_cg_unit_from_model(layer_num, 
                                          classes, 
                                          n_bands, 
                                          device, 
                                          weights)

    cam = gradcampp(model, layer, img, target)
    return cam

def main(ap):
    # load config
    config = ap["config"]

    with open(config) as file:
        config = json.load(file)

    # load model weights
    weights = ap["model_weights"]

    # get class names
    classes = config['classes']

    # number of image bands
    n_bands = len(config['means'])

    # dataloader parameters   
    params = {'batch_size': ap["batch_size"],
              'shuffle': True,
              'num_workers': 4,
              'drop_last': False
             }

    # define the base arguments used by all dataloaders
    base_args = {'band_means': tuple(config['means']),
                 'band_stds': tuple(config['stds']),
                 'target_class' : None,
                 'classes': classes,
                 'return_name': True
                }

    # get dataloaders
    validation = get_dataloader(config, ap["subset"], base_args)

    validation_generator = torch.utils.data.DataLoader(validation, **params)

    for i, (img_batch, lbl_batch, areas, fname, xform) in enumerate(validation_generator):

        size = img_batch.shape[-1]
        if img_batch.shape[0] > 1:
            m = "Currently only batch sizes of 1 supported"
            raise AssertionError(m)
        
        # NEED TO ADAPT THE METHOD TO WORK WITH MULTI LABEL
        # thus I will use the ground truth labels to determine the correct class to select
        present_labels = torch.where(lbl_batch[0] == 1)[0]
        all_class_cams = np.zeros((len(classes), size, size))

        # choose 3 bands for img segmentation into superpixels
        new_img = np.array(
            [img_batch[0, 0].numpy(),
             img_batch[0, 1].numpy(),
             img_batch[0, 3].numpy()
            ]
        )
        segments = slic(new_img.transpose(1,2,0), 
                        n_segments=200, compactness=10, 
                        sigma=1, multichannel=1)

        for target in range(len(classes)):
            # only calculate cams for classes that are present
            if target not in present_labels:
                # put empty array in label axes that are not present
                all_class_cams[target,:,:] = np.zeros((size, size))
            else:
                final_cams = [get_multi_scale_cams(img_batch.to(ap["device"]), 
                                                   target, 
                                                   layer_num, 
                                                   classes, 
                                                   n_bands, 
                                                   ap["device"], 
                                                   weights)
                              for layer_num in range(1, 5)]
                    
                final_cams = np.concatenate(final_cams, axis = 0)
                mean_cam = np.mean(final_cams[:3], axis=0) + final_cams[-1]

                sp_cam = superpixel_cam(mean_cam, segments)
                # Cams need to be thresholded with 0.5 as per the paper
                sp_cam = np.where(sp_cam >= 0.5,
                                  sp_cam,
                                  0).astype(np.float32)

                all_class_cams[target,:,:] = sp_cam

        # aggregates cams into a single mask of class labels, or 255 where no class
        output_mask = np.where(np.max(all_class_cams, axis = 0) == 0, 
                               255,
                               np.argmax(all_class_cams, axis = 0))
        
        # save to file
        save(output_mask.reshape(1, size, size), 
             [Path(fname[0]).name], 
             [xform], 
             ap["out_dir"], 
             crs, 
             dtype = np.uint8, 
             shape = (size, size))
        
def add_arguments():
    ap = argparse.ArgumentParser(prog='PseudoLabel Extractor', description='PseudoLabel Extractor')
    ap.add_argument('-w', '--model_weights', type=str, required = True,
                   help='Path to the file containing trained model weights.')
    ap.add_argument('-c', '--config', type=str, required = True,
            help='Give the path to the training config file.')
    ap.add_argument('-v', '--device',  type=str, required = True,
            help='Name of the desired device for training (e.g. cpu or cuda:1)')
    ap.add_argument('-o', '--out_dir',  type=str, required = True,
            help='Directory where the CAM masks should be saved to.')
    ap.add_argument('-b', '--subset', type=str, required = True, 
            help='Data subset to process CAMs on (e.g. train, val, or test.')
    ap.add_argument('-a', '--batch_size', default = 1, type = int,
            help='Number of images per batch.')
    
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    
    args = add_arguments()
    main(args)