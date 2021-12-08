import json
import torch
import argparse
from pydoc import locate
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
    
    creator = PseudoCreator(model, 
                            (n_bands, ap['up_size'], ap['up_size']),
                            threshold_cam = ap['cam_thresh'],
                            threshold_pred = ap['pred_threshold'],
                            use_enhanced = ap['enhanced'],
                            manual_sem = ap['n_seed'])
    
    creator.save_cams(generator, 
                      ap['outdir'],
                      device,
                      ap['r_shadow'], 
                      False)
                
def add_arguments():
    ap = argparse.ArgumentParser(prog='CAM Extractor', description='CAM Extractor')
    ap.add_argument('-w', '--model_weights', type=str, required = True,
                   help='Path to the file containing trained model weights.')
    ap.add_argument('-c', '--config', type=str, required = True,
            help='Give the path to the training config file.')
    ap.add_argument('-t', '--pred_threshold',  type=float, default = 0.80,
            help='Float value between 0.0 - 1.0 which determines the minimum confidence required for a classifaction prediction to be accepted.')
    ap.add_argument('-v', '--device',  type=str, required = True,
            help='Name of the desired device for training (e.g. cpu or cuda:1)')
    ap.add_argument('-o', '--outdir',  type=str, required = True,
            help='Directory where the CAM masks should be saved to.')
    ap.add_argument('-b', '--subset', type=str, required = True, 
            help='Data subset to process CAMs on (e.g. train, val, or test.')
    ap.add_argument('-r', '--cam_thresh', type=float, default = 0.9, 
            help='Threshold for the areas to use from the CAM heatmap.')
    ap.add_argument('-f', '--r_shadow', action='store_true',
            help='Whether to remove shadow pixels from CAMs.')
    ap.add_argument('-m', '--model', type=str,
            help='Model to load.')
    ap.add_argument('-e', '--enhanced', action='store_true',
            help='Whether to apply enhancement method (True) or just use original CAM (False).')
    ap.add_argument('-s', '--n_seed', default = 0, type = int,
            help='Number of seeds to use for manual SEM.')
    ap.add_argument('-a', '--batch', default = 1, type = int,
            help='Number of images per batch.')
    ap.add_argument('-u', '--up_size', default = 304, type = int,
            help='Size to upsample CAM to. Used for both height and width dims.')
    
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    
    args = add_arguments()
    main(args)