"""Tests the performance of CLM based methods for image segmentation. Thus, we can test
CAM, PCM, SEM and eSEM with this script.
"""

import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from pydoc import locate
from rasterio.crs import CRS
from wstm.utils.camutils import PseudoCreator
from wstm.models import get_classification_model
from wstm.utils.dataloader import get_dataloader
from sklearn.metrics import (f1_score, jaccard_score, precision_score, 
                             recall_score, confusion_matrix)


def main(ap):
    
    # load config
    with open(ap['config']) as file:
        config = json.load(file)

    # load model weights
    weights = ap['weights_file']

    # get class names
    classes = config['classes']
    
    # number of image bands
    n_bands = len(config['means'])

    # dataloader parameters   
    params = {'batch_size': ap['batch_size'],
              'shuffle': True,
              'num_workers': 4,
              'drop_last': False
             }

    # define the base arguments used by all dataloaders
    base_args = {'band_means': tuple(config['means']),
                 'band_stds': tuple(config['stds']),
                 'target_class' : None
                }
    
    # get dataloaders
    validation = get_dataloader(config, ap['split'], base_args)
    
    validation_generator = torch.utils.data.DataLoader(validation, **params)
    
    sig = nn.Softmax(dim=1)
        
    # create the model
    model = get_classification_model(ap['model'], classes, config)
    
    # load model weights and send to device
    device = torch.device(ap['device'])
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    model.to(device)
    print('Using:', device)

    all_preds = []
    all_truth = []
    
    # setup cam creator object
    creator = PseudoCreator(model, 
                            (n_bands, 304, 304),
                            ap['cam_threshold'],
                            ap['prediction_threshold'],
                            use_enhanced = ap['enhanced'],
                            manual_sem = ap['n_seed'])


    for i, (img_batch, lbl_batch, files, _) in enumerate(validation_generator):
        # load image and mask
        img_batch, lbl_batch = img_batch.to(device), lbl_batch.to(device)

        mask = creator(img_batch, 
                       filename = files, 
                       remove_shadow = ap["shadow"],
                       aggregate = True)
        
        all_preds.extend(np.array(mask).flatten())
        all_truth.extend(lbl_batch.cpu().numpy().flatten())
    
    # calc metrics for the batches
    cl = [0,1,2,3,4]
    print('Macro Precision:', precision_score(all_truth, all_preds, average = 'macro', labels = cl))
    print('Macro Recall:', recall_score(all_truth, all_preds, average = 'macro', labels = cl))
    print('Macro F1:', f1_score(all_truth, all_preds, average = 'macro', labels = cl))
    print('Macro mIoU:', jaccard_score(all_truth, all_preds, average = 'macro', labels = cl))

    #print(confusion_matrix(all_truth, all_preds, labels = cl))
    
def add_arguments():
    ap = argparse.ArgumentParser(prog='Test CLM', description='Tests CLM area predictions')
    ap.add_argument('-c', '--config', type=str, required = True,
            help='Give the path to the config file.')
    ap.add_argument('-w', '--weights_file',  type=str, required = True,
            help='Name of the pre-trained weights file.')
    ap.add_argument('-v', '--device',  type=str, required = True,
            help='Name of the desired device to use (e.g. cpu or cuda:1)')
    ap.add_argument('-r', '--cam_threshold', default=0.7, type=float,
            help='Threshold value for the region of CAM to keep.')
    ap.add_argument('-e', '--enhanced', action='store_true',
            help='Whether to apply enhancement method (True) or just use original CAM (False).')
    ap.add_argument('-z', '--batch_size', type=int, default=32,
            help='Size of batch.')
    ap.add_argument('-p', '--prediction_threshold',  type=float, default=0.9,
            help='Minimum required confidence in the network class prediction.')
    ap.add_argument('-m', '--model',  type=str,
            help='Name of model to use.')
    ap.add_argument('-s', '--n_seed', default = 0, type=int,
            help='Number of seeds to use for the SEM function. If left as zero the SEM function will not be applied.')
    ap.add_argument('-t', '--split', default='test', type=str,
            help='Dataset split to use. Should be either train or test.')
    ap.add_argument('-a', '--shadow', action='store_true',
            help='Whether to remove shadows.')
    
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    
    
    args = add_arguments()
    main(args)