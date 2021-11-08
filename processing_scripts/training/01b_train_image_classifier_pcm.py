"""Alternative script to train the first part of WSTM. Script uses PCM as the CAM method and
thus requires an alternative loss function from the usual image classifier training.
"""

import json
import torch
import warnings
import rasterio
import argparse
import numpy as np
from wstm.monitoring.metrics import *
from wstm.utils.augmenter import Augmenter
from wstm.utils.dataloader import get_dataloader
from wstm.models import get_classification_model
from wstm.trainers.basetrainer import ModelTrainer
from wstm.utils.trainUtils import get_class_weights
from wstm.monitoring.losses import AreaAwareMulti

def main(ap):

    # load config file
    with open(ap['config']) as file:
        config = json.load(file)
    
    # define GPU or CPU device
    device = torch.device(ap['device'])

    # get class names
    classes = config['classes']
    if config['aggregate']:
        classes = list(np.unique([c.split(' ')[0] for c in classes]))
    # remove last class if we want to ignore (only applies to deepglobe)
    if 'ignore_class' in config:
        if config['ignore_class'] == 255:
            classes = classes[:len(classes)-1]
    
    # define augmentations
    augs = {'hflip': {'prob': 0.5},
            'vflip': {'prob': 0.5},
            'rotate': {'degrees': [90, 90],
                       'prob': 0.5}
    }
    augs = Augmenter(augs)

    # load area based loss function weights
    if 'area_weights' in config and config['area_weights']:
        with open(config['area_weights']) as file:
            area_weights = json.load(file)
    else: area_weights = None

    # load class imbalance information
    if 'class_imbal_weights' in config:
        class_weights = get_class_weights(config).to(device)
    else: class_weights = None
    
    # dataloader parameters
    params = {'batch_size': config['batch_size'],
              'shuffle': True,
              'num_workers': config['num_workers'],
              'drop_last': config['drop_last']}

    # define the base arguments used by all dataloaders
    base_args = {'classes': classes,
                 'band_means': tuple(config['means']),
                 'band_stds': tuple(config['stds']),
                 'per_image_area_weights' : area_weights,
                 'return_name' : False,
                 'target_class' : None,
                 'augmenter' : augs}
    
    # get dataloaders
    training = get_dataloader(config, 'train', base_args)
    validation = get_dataloader(config, 'val', base_args) 
    
    # create generators
    training_generator = torch.utils.data.DataLoader(training, **params)
    validation_generator = torch.utils.data.DataLoader(validation, **params)
    
    generators = {'training': training_generator,
                  'validation': validation_generator
                  }
    
    # loss function
    criterion = AreaAwareMulti()
    
    # select model
    model = get_classification_model(ap['model'], classes, config)
    model.to(device)
    print('Using:', device)
    
    # pass all params to optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, 
                                  weight_decay=config['decay'], amsgrad=False)
    
    # define LR scheduler
    if config['scheduler']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                               factor=config['lr_decay'], 
                                                               patience=1, threshold=0.001, 
                                                               threshold_mode='rel', cooldown=0, 
                                                               min_lr=0, eps=1e-08, verbose=False)
    else:
        scheduler = None
    
    # train the model
    t = ModelTrainer(config['epochs'], 
                     classes, 
                     model, 
                     device, 
                     generators, 
                     criterion, 
                     optimizer, 
                     ap['save_name'], 
                     scheduler, 
                     class_weights, 
                     config['weights_path'] 
                     )
    model = t.run()
    
def add_arguments():
    ap = argparse.ArgumentParser(prog='Classification Trainer', description='Classification Trainer for PCM')
    ap.add_argument('-c', '--config', type=str, required = True,
            help='Give the path to the training config file.')
    ap.add_argument('-m', '--model', default='Sem_deeplab',
            help='Name of the model to use for training.')
    ap.add_argument('-s', '--save_name',  type=str, required = True,
            help='Name of the output weights file.')
    ap.add_argument('-v', '--device',  type=str, required = True,
            help='Name of the desired device for training (e.g. cpu or cuda:1)')
    
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    args = add_arguments()
    main(args)