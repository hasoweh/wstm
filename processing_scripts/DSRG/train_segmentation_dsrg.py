"""Original paper uses DeepLabv2 which outputs low resolution feature maps. 
However, in our paper we use Deeplabv3+ which has full resolution feature maps (same size 
as input), and thus the DSRG algorithm is extremely slow (8 hours per epoch roughly). 
Thus we here edit the DeepLabv3+ model to return the feature maps before the final 
upsampling so that we can use the DSRG on these smaller feature maps. Given that the final 
upsample is just a bilinear operation, it shouldn't matter if we perform DSRG on the maps 
before or after the upsampling. 
"""

import json
import torch
import argparse
import numpy as np
import torchvision
from wstm.utils.augmenter import Augmenter
from wstm.models.dsrg.losses import DSRG_Loss
from wstm.utils.dataloader import get_dataloader
from wstm.models.dsrg.dsrg_layer import DSRGLayer
from wstm.utils.trainUtils import get_class_weights
from segmentation_models_pytorch import DeepLabV3Plus
from wstm.trainers.pixelwise_trainer import DSRGTrainer

def main(ap):

    # load config file
    with open(ap['config']) as file:
        config = json.load(file)
    
    # define GPU or CPU device
    device = torch.device(ap['device'])
    
    # get class names
    classes = config['classes']
    
    # define augmentations
    augs = {'hflip': {'prob': 0.5},
        'vflip': {'prob': 0.5},
        'rotate': {'degrees': [25, 335],
                   'prob': 0.5}
    }
    augs = Augmenter(augs)
    
    # load class imbalance information
    if 'class_imbal_weights' in config and config['class_imbal_weights']:
        class_weights = get_class_weights(config)
    else: class_weights = None
    
    # dataloader parameters
    params = {'batch_size': config['batch_size'],
              'shuffle': True,
              'num_workers': config['num_workers'],
              'drop_last': config['drop_last']}

    # define the base arguments used by all dataloaders
    base_args = {'band_means': tuple(config['means']),
                 'band_stds': tuple(config['stds']),
                 'target_class' : None,
                 'augmenter' : augs}
    
    # get dataloaders
    training = get_dataloader(config, 'train', base_args)
    validation = get_dataloader(config, 'val', base_args)

    training_generator = torch.utils.data.DataLoader(training, **params)
    validation_generator = torch.utils.data.DataLoader(validation, **params)

    generators = {'training': training_generator,
                  'testing': validation_generator
                  }
    
    # loss function
    criterion = DSRG_Loss(ap["ignore"], device = ap["device"])
    
    # create model
    model = DeepLabV3Plus(encoder_name= config['model'], encoder_weights = None, 
                          in_channels = len(config['means']), classes = len(classes),
                          upsampling = 1) # set upsample to 1 and upsample manually later
    model.to(device)
    print('Using device:', device)
    
    # define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, 
                                  weight_decay=config['decay'], amsgrad=False)
    if config['scheduler']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                               factor=config['lr_decay'], 
                                                               patience=1, threshold=0.001, 
                                                               threshold_mode='rel', cooldown=0, 
                                                               min_lr=0, eps=1e-08, verbose=False)
    else:
        scheduler = None
    
    # train the model
    t = DSRGTrainer(config['epochs'], 
                    generators, 
                    model, 
                    device, 
                    criterion.to(device), 
                    optimizer, 
                    ap['save_name'], 
                    scheduler, 
                    config['weights_path']
                    )
    model = t.run()
    
def add_arguments():
    ap = argparse.ArgumentParser(prog='Semantic segmentation Trainer', description='Semantic segmentation Trainer')
    ap.add_argument('-i', '--ignore', type=int, default=255,
                   help='Pixel value in masks to ignore when determining the loss.')
    ap.add_argument('-c', '--config', type=str, required = True,
            help='Give the path to the training config file.')
    ap.add_argument('-s', '--save_name',  type=str, required = True,
            help='Name of the output weights file.')
    ap.add_argument('-v', '--device',  type=str, required = True,
            help='Name of the desired device for training (e.g. cpu or cuda:1)')

    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    
    args = add_arguments()
    main(args)