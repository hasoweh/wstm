"""Tests the segmentation performance of the final stage of the WSTM method.
"""
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from rasterio.crs import CRS
from wstm.utils.camutils import save
from wstm.utils.dataloader import get_dataloader
from segmentation_models_pytorch import DeepLabV3Plus
from wstm.trainers.pixelwise_trainer import default
from sklearn.metrics import (f1_score, 
                             jaccard_score, 
                             precision_score, 
                             recall_score, 
                             confusion_matrix)

def main(ap):
    # load config
    with open(ap['config']) as file:
        config = json.load(file)

    # load model weights
    weights = ap['weights_file']

    # determine the coordinate reference system
    crs = default

    classes = config['classes']

    # dataloader parameters   
    params = {'batch_size': ap['batch_size'],
              'shuffle': True,
              'num_workers': 1,
              'drop_last': False}

    # define the base arguments used by all dataloaders
    base_args = {'band_means': tuple(config['means']),
                 'band_stds': tuple(config['stds']),
                 'target_class' : None
                }
    
    # get dataloaders
    validation = get_dataloader(config, 'test', base_args)
    
    validation_generator = torch.utils.data.DataLoader(validation, **params)
    sig = nn.Softmax(dim=1)

    # create the model
    model = DeepLabV3Plus(encoder_name= config['model'], encoder_weights = None, 
                          in_channels = len(config['means']), classes = len(classes))
    device = torch.device(ap['device'])
    model.load_state_dict(torch.load(weights, map_location = device))
    model.eval()
    model.to(device)
    print('Using:', device)

    all_preds = []
    all_truth = []
    
    with torch.no_grad():
        for i, (img_batch, lbl_batch, files, xform) in enumerate(validation_generator):

            # load image and mask
            img_batch, lbl_batch = img_batch.to(device), lbl_batch.to(device)

            # process batch through network
            out = model(img_batch.float())

            # get the prediction and probability maps
            preds = sig(out).cpu().numpy()

            pred_out = np.argmax(preds, axis = 1)
            
            all_preds.append(pred_out)
            all_truth.append(lbl_batch)
            
    # prepare batches for using in metric functions 
    all_preds = np.concatenate(all_preds)
    all_preds = all_preds.flatten()
    all_truth = torch.cat(all_truth).cpu().numpy().flatten()
    
    # calc metrics for the batches
    cl = [0,1,2,3,4]
    print('Macro Precision:', precision_score(all_truth, all_preds, average = 'macro', labels = cl))
    print('Macro Recall:', recall_score(all_truth, all_preds, average = 'macro', labels = cl))
    print('Macro F1:', f1_score(all_truth, all_preds, average = 'macro', labels = cl))
    print('Macro mIoU:', jaccard_score(all_truth, all_preds, average = 'macro', labels = cl))
    
    print(confusion_matrix(all_truth, all_preds, labels = cl))
    
def add_arguments():
    ap = argparse.ArgumentParser(prog='Segmentation predictor', description='Segmentation predictor')
    ap.add_argument('-c', '--config', type=str, required = True,
            help='Give the path to the config file.')
    ap.add_argument('-w', '--weights_file',  type=str, required = True,
            help='Name of the pre-trained weights file.')
    ap.add_argument('-v', '--device',  type=str, required = True,
            help='Name of the desired device to use (e.g. cpu or cuda:1)')
    ap.add_argument('-z', '--batch_size', type=int, default=64,
            help='Size of batch to process.')
    
    
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    
    args = add_arguments()
    main(args)