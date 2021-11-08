import os
import glob
import json
import torch
import warnings
import argparse
import numpy as np
import torchvision
import torch.nn as nn
from wstm.monitoring.metrics import *
from wstm.utils.dataloader import get_dataloader
from wstm.models import get_classification_model

def main(ap):

    weights_file = ap['weights']
    
    with open(ap['config']) as file:
        config = json.load(file)
        
    classes = config['classes']
    
    device = torch.device(ap['device'])
    
    model = get_classification_model(ap['model'], classes, config)
    model.load_state_dict(torch.load(weights_file, map_location = device))
    model.to(device)
    model.eval()
    
    params = {'batch_size': ap['batch'],
              'shuffle': False,
              'num_workers': 16,
              'drop_last' : False}

    base_args = {'classes': classes,
                 'band_means': tuple(config['means']),
                 'band_stds': tuple(config['stds']),
                 'per_image_area_weights' : None,
                 'target_class' : None,
                 'augmenter' : None}

    # get dataloaders
    test = get_dataloader(config, 'test', base_args)

    generator = torch.utils.data.DataLoader(test, **params)
    
    sig = nn.Sigmoid()

    all_labels = []
    all_preds = []
    all_sigmoids = []

    # loop through all batches
    with torch.no_grad():
        for img_batch, lbl_batch, _ in generator:

            # load image and mask
            img_batch, lbl_batch = img_batch.to(device), lbl_batch.to(device)

            # process batch through network
            out = model(img_batch.float())
            
            pred_out = np.array(sig(out).cpu() > 0.5, dtype=float)

            all_sigmoids.append(sig(out).cpu().numpy())
            all_labels.append(lbl_batch.cpu().numpy())
            all_preds.append(pred_out)
            
    metss = EpochMetrics(all_labels, all_preds, all_sigmoids, classes)()
    
def add_arguments():
    ap = argparse.ArgumentParser(prog='Classification Tester', description='Classification Tester')
    ap.add_argument('-c', '--config', type=str, required = True,
            help='Give the path to the training config file.')
    ap.add_argument('-m', '--model', default='Resnet',
            help='Name of the model to use for training.')
    ap.add_argument('-b', '--batch',  type=int, required = True,
            help='Number of images per batch.')
    ap.add_argument('-v', '--device',  type=str, required = True,
            help='Name of the desired device for training (e.g. cpu or cuda:1)')
    ap.add_argument('-w', '--weights',  type=str, required = True,
            help='Path to the weights file of the trained model.')
    
    args = vars(ap.parse_args())
    return args
    
if __name__ == '__main__':

    args = add_arguments()
    main(args)