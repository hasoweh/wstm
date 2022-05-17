#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 09:49:30 2021

@author: steve
"""

import os
import re
import glob
import json
import torch
import rasterio
import cv2 as cv
import numpy as np
import torch.nn as nn
from .camutils import *
from pathlib import Path
from numpy import genfromtxt
from .inputAssertions import *
from .image_utils import upsample
from .augmenter import Augmenter
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

def filter_target_files(targets, full_list):
    """Filters a list of file names and returns only those
    present in the 'targets'.
    
    Parameters:
    -----------
    targets : list
        List of the names which are to be kept from the full_list.
    full_list : list
        List to be filtered.
    """
    if isinstance(targets, str):
        rasterData = [img for img in full_list if targets in Path(img).stem]
    elif isinstance(targets, list):
        rasterData = [img for img in full_list if Path(img).stem in targets]
    
    return rasterData

def convert_one_hot(classes, labels):
    """Converts a list of class indexes to a one-hot
    encoded vector.
    
    Examples:
    ---------
    >>> all_classes = ['grass', 'tree', 'road', 'cloud']
    >>> image_labels = [1, 2]
    >>> print(convert_one_hot(all_classes, image_labels))
    [0, 1, 1, 0]
    
    """
    # get the one hot encoding vector for the target labels
    one_hot = torch.zeros(len(classes), dtype = torch.float32)
    for lab in labels:
        # get index from original class list
        idx = classes.index(lab)
        one_hot[idx] = 1
    return one_hot

def load_image_and_label(img_path, labels, classes):
    '''
    Parameters
    ----------
    img_path : str
        Full path to an image.
    labels: json
        A JSON formatted dictionary mapping image names to labels.
    '''
    # load image file
    with rasterio.open(img_path) as f:
        img = f.read()
        meta = f.meta.copy()

    # Take the filename from the file path
    filen = Path(img_path).name
    
    # get the label
    label = labels[filen]
    
    if len(label) == 0:
        raise AssertionError('No valid label for %s' % filen)
        
    one_hot = convert_one_hot(classes, label)
    
    return img.astype(np.uint8), one_hot, meta

def get_img_paths(json_path, img_folder, target_class, phase, img_type = "tif"): # need to make img type flag for using npy files
    # load image target labels from json file
    with open(json_path) as file:
        print('Opening', json_path)
        labels = json.load(file)

    # get all full image paths
    pattern = os.path.join(img_folder, '%s.%s')
    file_paths = [pattern % (Path(img).stem, img_type) for img in labels[phase]]
    if target_class is not None:
        file_paths = filter_target_files(target_class, 
                                         file_paths
                                         )
    return file_paths, labels

class ForestDataset(Dataset):
    """Forest dataset class.
    """
    
    def __init__(self, 
                 phase,
                 img_folder, 
                 json_file,
                 classes, 
                 band_means, 
                 band_stds, 
                 per_image_area_weights = None,
                 return_name = False,
                 aggregate = False,
                 target_class = None,
                 augmenter = None):
        """
        Parameters
        ----------
        phase : str
            Tells which phase of training we are in. Can be 'train', 
            'val', or 'test'.
        img_folder : str
            Folder containing image files.
        json_file : str
            Path to the desired label file.
        classes : list
            List of target classes.
        band_means : tuple
            List of mean pixel values for each image band. Means
            are calculated over the entire image dataset.
        band_stds : tuple
            List of pixel standard deviation values for each image 
            band. Standard deviations are calculated over the entire 
            image dataset.
        per_image_area_weights : optional, dict
            A dictionary where the keys are image names and the values are
            vectors giving class specific weights for the loss function. 
            The weight vectors should thus be the length of the number of
            classes we are training and each value should be 1 + % pixels 
            which the label covers.
            If left as None, the function will return weights of 1 (so it will
            have no effect on the normal loss).
        return_name : bool
            Whether the filename should be returned by the dataloader.
        aggregate : bool
            Whether tree species classes should be aggregated to their 
            family level.
            (e.g. Quercus robur and Quercus rubra becomes Quercus)
        target_class : str
            String giving a keyword. Only images with this keyword will be
            loaded by the dataloader. Can also be used to target one specific
            image by giving the entire filename of the desired image.
        augmenter : optional, wstm.utils.augmenter.Augmenter
            An Augmenter object which can be used to apply augmentations
            to images.
        """
        # assert data types
        assert_list(classes, 'classes')
        assert_tuple(band_stds, 'band_stds')
        assert_tuple(band_means, 'band_means')
        assert_str(img_folder, 'img_folder')
        assert_str(json_file, 'json_file')
        
        self.return_fn = return_name
        self.phase = phase
        self.classes = classes
        self.augmenter = augmenter
    
        # init base transforms
        self.transforms = [T.ToTensor(), T.Normalize(band_means, band_stds)]

        if per_image_area_weights is not None:
            assert_dict(per_image_area_weights, 'per_image_area_weights')
        self.weights = per_image_area_weights

        self.rasterData, self.labels = get_img_paths(json_file, 
                                                     img_folder, 
                                                     target_class,
                                                     phase)
        

    def __len__(self):
        return len(self.rasterData)
        
    def __getitem__(self, idx):
        # ensure the idx is in a list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get full file path
        img_name = self.rasterData[idx]
        image, label, meta = load_image_and_label(img_name, 
                                                  self.labels[self.phase],
                                                  self.classes)
        
        transform = T.Compose(self.transforms)
        
        # change the order of the bands, as ToTensor will flip them
        image = image.transpose(1, 2, 0)
        image = transform(image)
        
        # apply extra augmentations
        if self.augmenter is not None:
            image = self.augmenter([image])[0]
        
        # determine outputs
        if self.weights is None:
            weight = torch.ones(len(self.classes))
        else:
            # return with area weights
            weight = torch.tensor(self.weights[Path(img_name).name])
            
        if not self.return_fn:
            return image, label, weight, np.array(meta["transform"])
        else:
            return image, label, weight, img_name, np.array(meta["transform"])
        
class CamDataset(Dataset):
    """Loads weakly supervised pixel labels based on class localization 
    maps (CLMs).
    """

    def __init__(self, 
                 json_file,
                 phase,
                 img_folder, 
                 mask_folder,
                 band_means, 
                 band_stds, 
                 augmentations = None,
                 target_class = None,
                 upsample = False,
                 augmenter = None,
                 img_type = 'tif',
                 mask_type = 'tif'):
        
        mask_paths = get_img_paths(json_file, mask_folder, target_class, phase, mask_type)[0]
        if target_class is not None:
            mask_paths = filter_target_files(target_class, 
                                        mask_paths)
            
        # assert we found files
        if len(mask_paths) == 0:
            patt = mask_folder + '/*.%s' % mask_type
            m = "No files found using  pattern: %s" % patt
            raise AssertionError(m)
        self.rasterData = mask_paths
        
        # make a dictionary of {img: mask}
        patt = os.path.join(img_folder, '%s.%s')
        self.files_dict = {p: patt % (Path(p).stem, img_type) for p in mask_paths}

        self.img_folder = img_folder
        self.img_type = img_type
        self.upsample = upsample
        self.augmenter = augmenter
        
        # initialize the base transformations that should always be applied
        self.img_preprocessing = [T.ToTensor(),
                                  T.Normalize(band_means, band_stds)]
        
    def __len__(self):
        return len(self.rasterData) 
        
    def load_mask_and_img(self, mask_path, img_path):
        # load files
        with rasterio.open(img_path) as f:
            img_arr = f.read()
            meta = f.meta.copy()
        with rasterio.open(mask_path) as f:
            mask_arr = f.read()
        mask_arr = np.squeeze(mask_arr)
        
        return img_arr, torch.tensor(mask_arr), meta
    
    def upsample(self, img_arr, mask_arr):
        
        img_arr = upsample(img_arr,
                           shape = (4, 304, 304), 
                           interpolation = cv.INTER_CUBIC, 
                           dtype = np.uint8)

        mask_arr = upsample(mask_arr,
                            shape = (304, 304), 
                            interpolation = cv.INTER_NEAREST, 
                            dtype = np.uint8)
        return img_arr, mask_arr

    def __getitem__(self, idx):
        # ensure the idx is in a list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get full file path
        mask_path = self.rasterData[idx]
        img_path = self.files_dict[mask_path]
        
        # load data
        image, mask, meta = self.load_mask_and_img(mask_path, img_path)
        if self.upsample:
            img_arr, mask_arr = upsample(img_arr, mask_arr)

        # set up transformations
        pre_process = T.Compose(self.img_preprocessing)
        
        # change the order of the bands, as ToTensor will flip them
        image = image.transpose(1, 2, 0)
        image = pre_process(image)
        
        # apply extra augmentations
        if self.augmenter is not None:
            image, mask = self.augmenter([image, mask])

        return image, mask, img_path, np.array(meta['transform'])
    
class DSRGDataset(CamDataset):
    
    def __init__(self, 
                 json_file,
                 phase,
                 img_folder, 
                 mask_folder,
                 band_means, 
                 band_stds, 
                 augmentations = None,
                 target_class = None,
                 upsample = False,
                 augmenter = None,
                 img_type = 'tif',
                 mask_type = 'npy',
                 classes = []):
        
        super().__init__(json_file,
                         phase,
                         img_folder, 
                         mask_folder,
                         band_means, 
                         band_stds, 
                         augmentations = augmentations,
                         target_class = target_class,
                         upsample = upsample,
                         augmenter = augmenter,
                         img_type = img_type,
                         mask_type = mask_type)
        
        
        # load image target labels from json file
        with open(json_file) as file:
            self.img_lvl_labels = json.load(file)[phase]
        self.classes = classes
        
    def load_mask_and_img(self, mask_path, img_path):
        
        # load files
        with rasterio.open(img_path) as f:
            img_arr = f.read()
            meta = f.meta.copy()
        mask_arr = np.load(mask_path)
        
        return img_arr, torch.tensor(mask_arr), meta

        
    def __getitem__(self, idx):
        # ensure the idx is in a list
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # get full file path
        mask_path = self.rasterData[idx]
        img_path = self.files_dict[mask_path]
        
        # load image, img meta, and masks
        image_orig, mask, meta = self.load_mask_and_img(mask_path, img_path)
        
        # get image level labels
        img_lvl_label = self.img_lvl_labels[Path(img_path).name]
        img_lvl_label = convert_one_hot(self.classes, img_lvl_label)
        
        # set up transformations
        pre_process = T.Compose(self.img_preprocessing)
        
        # change the order of the bands, as ToTensor will flip them
        image = image_orig.transpose(1, 2, 0)
        # for DSRG, need 1 image preprocessed (for the network) and 1 unchanged (for CRF)
        image = pre_process(image)
        
        # apply extra augmentations
        if self.augmenter is not None:
            image, mask = self.augmenter([image, mask])
            
        # need to downsample the original image for parts of DSRG
        image_orig = upsample(image_orig, (image_orig.shape[0],
                                           mask.shape[-1],
                                           mask.shape[-1])
                             )

        return image, mask, img_lvl_label, image_orig, np.array(meta['transform'])
        
        
class DSRG_OnlineDataset(ForestDataset):
    """Doesn't try to load CAM masks because these will be generated live.
    Will need more time to process because need to pass an image first through
    the 1st stage CNN, but will remove the need for more hard drive space.
    """
    
    def __init__(self, 
                 phase,
                 img_folder, 
                 json_file,
                 classes, 
                 band_means, 
                 band_stds, 
                 per_image_area_weights = None,
                 return_name = False,
                 aggregate = False,
                 target_class = None,
                 augmenter = None):
        
        super().__init__(phase,
                         img_folder, 
                         json_file,
                         classes, 
                         band_means, 
                         band_stds, 
                         per_image_area_weights = per_image_area_weights,
                         return_name = return_name,
                         aggregate = aggregate,
                         target_class = target_class,
                         augmenter = augmenter)
        
    def __getitem__(self, idx):
        # ensure the idx is in a list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get full file path
        img_name = self.rasterData[idx]
        image_orig, img_lvl_label, meta = load_image_and_label(img_name, 
                                                              self.labels[self.phase],
                                                              self.classes)
        
        transform = T.Compose(self.transforms)
        
        # change the order of the bands, as ToTensor will flip them
        image = image_orig.transpose(1, 2, 0)
        image = transform(image)
        
        # apply extra augmentations
        if self.augmenter is not None:
            image = self.augmenter([image])[0]
        
        # determine outputs
        if self.weights is None:
            weight = torch.ones(len(self.classes))
        else:
            # return with area weights
            weight = torch.tensor(self.weights[Path(img_name).name])
            

        return image, img_lvl_label, image_orig, np.array(meta["transform"])
        
        
def get_dataloader(config, phase, base_args):

    # select the function to create the desired dataloader
    loader_dict = {'ForestDataset': create_forest,
                   'CamDataset': create_cam,
                   "DSRGDataset": create_dsrg,
                   "DSRG_OnlineDataset": create_dsrg_online
                  }
    loader = config['loader']
    
    # create the dataloader
    dataloader = loader_dict[loader](config, phase, base_args)
    
    return dataloader
    
def create_forest(config, phase, base_args):
    
    base_args['phase'] = phase
    base_args['json_file'] = config['json_file']
    base_args['img_folder'] = config['img_folder']
        
    return ForestDataset(**base_args)

def create_cam(config, phase, base_args):
    
    base_args['phase'] = phase
    base_args['json_file'] = config['json_file']
    base_args['img_folder'] = config['img_folder']
    if phase == "train" or phase == "val":
        base_args['mask_folder'] = config['cam_folder'] # should be psuedolabels in the folder
    elif phase == "test":
        base_args['mask_folder'] = config['test_folder']
    base_args['img_type'] = config['img_type']
    base_args['mask_type'] = config['mask_type']
    
    return CamDataset(**base_args)

def create_dsrg(config, phase, base_args):
    
    base_args['phase'] = phase
    base_args['json_file'] = config['json_file']
    base_args['img_folder'] = config['img_folder']
    if phase == "train" or phase == "val":
        base_args['mask_folder'] = config['cam_folder'] # should be raw cams in the folder
        base_args['mask_type'] = config['mask_type']
        base_args['classes'] = config['classes']
    elif phase == "test":
        base_args['mask_folder'] = config['test_folder']
        base_args['mask_type'] = config['img_type']
    base_args['img_type'] = config['img_type']
    
    if phase == "train" or phase == "val":
        return DSRGDataset(**base_args)
    elif phase == "test":
        return CamDataset(**base_args)


def create_dsrg_online(config, phase, base_args):
    base_args['phase'] = phase
    base_args['json_file'] = config['json_file']
    base_args['img_folder'] = config['img_folder']
    base_args['classes'] = config["classes"]
    
    return DSRG_OnlineDataset(**base_args)

