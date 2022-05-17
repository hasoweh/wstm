"""Tests the segmentation performance of the final stage of the WSTM method.
"""
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from wstm.utils.dataloader import get_dataloader
from segmentation_models_pytorch import DeepLabV3Plus
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

    classes = config['classes']

    # dataloader parameters
    params = {'batch_size': ap['batch_size'],
              'shuffle': True,
              'num_workers': 1,
              'drop_last': False}

    # define the base arguments used by all dataloaders
    base_args = {'band_means': tuple(config['means']),
                 'band_stds': tuple(config['stds']),
                 'target_class' : targets
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
            
            all_preds.extend(pred_out.flatten())
            all_truth.extend(lbl_batch.cpu().numpy().flatten().tolist())


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
    
    targets = [
        "Cleared_0_1006_WEFL_NLF",
        "Cleared_0_1034_WEFL_NLF",
        "Cleared_0_1082_WEFL_NLF",
        "Cleared_0_112_WEFL_NLF",
        "Cleared_0_170_WEFL_NLF",
        "Cleared_0_1296_WEFL_NLF", 
        "Cleared_0_1339_WEFL_NLF",
        "Cleared_0_1663_WEFL_NLF",
        "Cleared_0_1785_WEFL_NLF",
        "Fagus_sylvatica_9_6388_WEFL_NLF",
        "Fagus_sylvatica_9_7201_WEFL_NLF", 
        "Fagus_sylvatica_7_6792_WEFL_NLF", 
        "Fagus_sylvatica_8_5500_WEFL_NLF",
        "Fagus_sylvatica_7_5549_WEFL_NLF",
        "Fagus_sylvatica_3_5311_WEFL_NLF", 
        "Fagus_sylvatica_3_8037_WEFL_NLF",
        "Fagus_sylvatica_3_8976_WEFL_NLF", 
        "Fagus_sylvatica_4_9918_WEFL_NLF",
        "Picea_abies_2_14926_WEFL_NLF",
        "Picea_abies_3_10077_WEFL_NLF",
        "Picea_abies_3_12659_WEFL_NLF",
        "Picea_abies_3_10161_WEFL_NLF",
        "Picea_abies_3_10270_WEFL_NLF",
        "Picea_abies_3_10476_WEFL_NLF",
        "Picea_abies_3_10742_WEFL_NLF",
        "Picea_abies_2_12498_WEFL_NLF",
        "Picea_abies_3_10915_WEFL_NLF",
        "Pinus_sylvestris_2_17451_WEFL_NLF",
        "Pinus_sylvestris_2_18499_WEFL_NLF",
        "Pinus_sylvestris_3_15213_WEFL_NLF",
        "Pinus_sylvestris_3_16716_WEFL_NLF", 
        "Pinus_sylvestris_3_17383_WEFL_NLF",
        "Pinus_sylvestris_3_17956_WEFL_NLF",
        "Pinus_sylvestris_3_18171_WEFL_NLF",
        "Pinus_sylvestris_3_19102_WEFL_NLF",
        "Pinus_sylvestris_4_18747_WEFL_NLF",
        "Pinus_sylvestris_4_18775_WEFL_NLF",
        "Quercus_petraea_8_22319_WEFL_NLF",
        "Quercus_petraea_9_22282_WEFL_NLF",
        "Quercus_robur_7_29902_WEFL_NLF",
        "Quercus_robur_8_27749_WEFL_NLF",
        "Quercus_robur_8_29916_WEFL_NLF",
        "Quercus_robur_9_26479_WEFL_NLF",
        "Quercus_robur_9_27951_WEFL_NLF",
        "Quercus_rubra_2_32622_WEFL_NLF",
        "Quercus_rubra_3_30218_WEFL_NLF",
        "Quercus_rubra_3_34305_WEFL_NLF",
        "Quercus_rubra_4_33400_WEFL_NLF"
    
    ]
    
    args = add_arguments()
    main(args)
