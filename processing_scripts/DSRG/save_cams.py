import cv2
import json
import torch
import argparse
import numpy as np
from pydoc import locate
from pathlib import Path
from wstm.utils.image_utils import upsample
from wstm.utils.camutils import PseudoCreator
from wstm.models import get_classification_model
from wstm.utils.dataloader import get_dataloader

import warnings
warnings.filterwarnings('ignore')

def main(ap): 

    # load config file
    with open(ap['config']) as file:
        config = json.load(file)
    
    # image bands
    n_bands = len(config['means'])
    
    # get class names
    classes = config['classes']

    # define GPU or CPU device
    device = torch.device(ap['device'])
    
    # dataloader parameters
    params = {'batch_size': ap['batch'],
              'shuffle': False,
              'num_workers': config['num_workers'],
              'drop_last': False}

    # define the base arguments used by all dataloaders
    base_args = {'classes': classes,
                 'band_means': tuple(config['means']),
                 'band_stds': tuple(config['stds']),
                 'per_image_area_weights' : None,
                 'return_name' : True,
                 'target_class' : None,
                 'augmenter' : None}
    
    # get dataloader and generator
    loader = get_dataloader(config, ap['subset'], base_args)
    generator = torch.utils.data.DataLoader(loader, **params)

    # get full model
    model = get_classification_model(ap['model'], classes, config, debug = True)
    
    # load model weights
    weights_file = ap['model_weights']
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.to(device)
    model.train(False)

    for loaded in generator:
        img_batch = loaded[0].to(device)
        filenames = loaded[3]
        
        out = model(img_batch.float())
        
        logits = out[1]
        features = out[-1]
        clms = out[0].detach().cpu().numpy()
        batch, n_class, _, _ = clms.shape
        
        for clm, file in zip(clms, filenames):
        
            clm = upsample(clm,
                           shape = (n_class, 76, 76), # needs to be the same size as the output from the segmentation model. We edit DeepLabv3+ to not do the upsample of the final feature maps until later so that we can first work with smaller maps for the DSRG algorithm, otherwise it is too slow.
                           interpolation = cv2.INTER_CUBIC, 
                           dtype = np.float32)

            # save to npy files
            out = ap["outdir"] + "/%s.npy"
            np.save(out % Path(file).stem, clm)
                

def add_arguments():
    ap = argparse.ArgumentParser(prog='CAM Extractor', description='CAM Extractor')
    ap.add_argument('-w', '--model_weights', type=str, required = True,
                   help='Path to the file containing trained model weights.')
    ap.add_argument('-c', '--config', type=str, required = True,
            help='Give the path to the training config file.')
    ap.add_argument('-v', '--device',  type=str, required = True,
            help='Name of the desired device for training (e.g. cpu or cuda:1)')
    ap.add_argument('-o', '--outdir',  type=str, required = True,
            help='Directory where the CAM masks should be saved to.')
    ap.add_argument('-b', '--subset', type=str, required = True, 
            help='Data subset to process CAMs on (e.g. train, val, or test.')
    ap.add_argument('-m', '--model', type=str,
            help='Model to load.')
    ap.add_argument('-a', '--batch', default = 1, type = int,
            help='Number of images per batch.')
    
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    
    args = add_arguments()
    main(args)