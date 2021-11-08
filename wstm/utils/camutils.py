import os
import glob
import torch
import rasterio
import cv2 as cv
import numpy as np
import torch.nn as nn
from pathlib import Path
from .inputAssertions import *
from .image_utils import upsample
from ..models.utils import sem_seeds
from torch.nn.functional import cosine_similarity
from .file_utils import delete_files_from_dir
from torchvision.transforms.functional import to_pil_image
from .image_utils import load_image, threshold_shadows, get_img_meta

def find_values_same_across_lists(nested_list):
    """Finds values which are contained in all nested lists.
    """
        
    class_max = list(set.intersection(*map(set, nested_list)))
    return class_max

def compare_arrays(array_list):
    """Generator which returns where each array is higher 
    than all other arrays in a given list.
    """
    def find_indexes(target_array, other_array):
        return np.where(target_array.flatten() > other_array.flatten())[0]
    
    for i, arr in enumerate(array_list):
        # get the list of array_list without the current one
        subset_arrays = array_list.copy()
        del subset_arrays[i]
        subset_arrays = np.array(subset_arrays)

        idxs = [find_indexes(arr, arr2) for arr2 in subset_arrays]
        class_max = find_values_same_across_lists(idxs)
        
        yield class_max, i 
    
def save(mask_batch, filenames, batch_xform, outd, crs, 
         dtype = np.uint8, shape = (304, 304), clear = False):
    '''Takes a batch of arrays and saves them to file.
    
    Parameters
    ----------
    mask_batch : np.array(shape = (n, w, h))
        Array containing a batch of 'n' segmentation maps of
        size width (w) by height (h).
    filenames : np.array(shape = n)
        Array containing 'n' filenames which correspond to
        the segmentation maps of 'pred_batch'.
    batch_xform : np.array(shape = n)
        Array containing 'n' transformation parameters which 
        correspond to the segmentation maps of 'pred_batch'.
        Used to assign proper georeferencing information to
        each saved segmentation mask.
    outd : str
        Output directory to save masks.
    dtype : np.dtype
        Desired datatype of the output file.
    shape : tuple
        Shape (height, width) of the segmentation masks.
    clear : bool
        Whether to delete all files existing in the output
        directory (outd).
    '''
    
    # delete all files in the folder
    if clear:
        delete_files_from_dir(outd)
        
    n_masks_batch = mask_batch.shape[0]
    
    for b in range(n_masks_batch):
        fn = Path(filenames[b]).stem
        
        # get img metadata
        meta = get_img_meta(crs, shape, dtype, batch_xform[b])  
            
        outpath = os.path.join(outd, fn + '.tif')
        out_arr = mask_batch[b, :, :].astype(dtype)
        with rasterio.open(outpath, 'w', **meta) as dest:
            dest.write(np.expand_dims(out_arr, 0))     
                
class CamCreator():
    """Generates pseudolabels from class activation maps.
    """

    def __init__(self, 
                 model, 
                 input_shape = (4, 304, 304),
                 threshold_cam = 0.9,
                 threshold_pred = 0.9,
                 use_enhanced = True,
                 manual_sem = False
                 ):
        
        self.model = model
        self.thresh = threshold_cam
        self.pred_thres = threshold_pred
        self.use_enhanced = use_enhanced
        self.input_shape = input_shape
        self.width = input_shape[2]
        self.height = input_shape[1]
        self.sig = nn.Sigmoid()
        self.manual_sem = manual_sem
        
        if self.manual_sem and self.use_enhanced:
            m = "'manual_sem' and 'use_enhanced' cannot both be True"
            raise AssertionError(m)
       
    def upsample_cam(self, cam):
        # if cam already the size desired no upsample
        if cam.shape[-1] == self.width:
            return cam
        # if not then upsample
        else:
            out_shape = (self.height, self.width)
            
            return np.array(upsample(cam.numpy(),
                                     shape = out_shape, 
                                     interpolation = cv.INTER_CUBIC, 
                                     dtype = np.uint8))
    
    @property
    def minus(self):
        return self._minus
    
    @minus.setter
    def minus(self, tuple_):
        """Determines if we need to subtract 1 from the length
        of a list. If we include shadow as a class then we remove
        one index from the length of the list containing class 
        specific pixel labels.
        """
        shadow_class = tuple_[0]
        filename = tuple_[1]
        
        if shadow_class is not None and 'Cleared' not in filename:
            self._minus = 1
        else: self._minus = 0

    def get_multiclass_output(self, output_mask, class_masks, idx):
        
        # create a copy of the class_masks list from which we can delete
        compare_masks = class_masks.copy()

        if self.minus:
            # remove the shadow mask from the class mask comparisons
            del compare_masks[-1]

        # find all pixels where each class has the highest probability
        for index_maxes, i in compare_arrays(compare_masks):
            # assign the class index value to all pixels in index_maxes
            output_mask[index_maxes] = idx[i]

        # reshape the mask to the original image shape
        output_mask = output_mask.reshape((self.height, self.width))
        
        return output_mask
        
    def aggregate_cams(self, shadow_class, filename, class_masks, idx):
        """Takes individual class CAMs and overlays them to 
        form one single mask. In areas where two CAMs overlap,
        the class with the higher probability will be assigned
        to the pixel.
        """
        
        nodata = 255
        
        # start with nodata and fill in class IDs after
        output_mask = np.full((self.height * self.width), nodata, dtype = np.uint8)
        
        # subtract one from the length of the class_masks to get
        # correct number of non-shadow classes
        self.minus = (shadow_class, filename)

        # if we only have 1 CAM (aside from shadow) then we don't need to compare
        if len(class_masks)-self.minus == 1:
            output_mask = np.where(class_masks[0] >= self.thresh, 
                                   idx[0], 
                                   255).astype(np.uint8)
            
        # if multiple CAMs we compare to find highest class at each pixel
        else:
            output_mask = self.get_multiclass_output(output_mask, 
                                                     class_masks, 
                                                     idx)

        # add the shadow class pixels (if necessary)
        if self.minus:
            output_mask = np.where(class_masks[-1] == shadow_class, 
                                   shadow_class, 
                                   output_mask)
        return output_mask
        
    def process_batch(self, img_batch):
        """Runs inference on a batch of images.
        """
        out = self.model(img_batch.float())
        
        logits = out[1]
        features = out[-1]
        
        if self.use_enhanced:
            cams = out[2]
        else: cams = out[0]

        return logits, cams, features
        
    def get_confident_predictions(self, current_logits):
        """Finds the class indexes for classes predicted above 
        given threshold.
        """
        idx = torch.where(self.sig(current_logits).flatten() > self.pred_thres)
        idx = idx[0].cpu().numpy().tolist()
        
        return idx
    
    def decide_shadow_mask(self, cam, shadow_class, current_file):
        """Determines whether to create a pixel label mask for the
        shadow class or not.
        """
        # we don't take shadow class from Cleared images because its
        # possible the image will be completely field and no shadow
        if shadow_class is not None and 'Cleared' not in current_file:
            # makes a mask for a shadow class
            cam, self.shadows = threshold_shadows(cam, 
                                                  current_file,
                                                  shadow_class = shadow_class)
        else:
            # just removes shadows but doesn't include as a class
            cam, self.shadows = threshold_shadows(cam, current_file)
        
        return cam
    
    def prepare_cam(self, 
                    cam_, 
                    current_features, 
                    remove_shadow, 
                    shadow_class, 
                    current_file
                   ):
        """Gets a single CAM ready for being used as mask."""
        
        if self.manual_sem:
            cam_ = sem_seeds(cam_, current_features, self.manual_sem)

        cam_ = self.upsample_cam(cam_.detach().cpu())

        # threshold pixels with high confidence
        if self.thresh is not None:
            cam_ = np.where(cam_ >= self.thresh,
                            cam_,
                            0).astype(np.float32)

        # threshold shadows
        if remove_shadow:
            # determines whether to make a shadow mask or just remove
            cam_ = self.decide_shadow_mask(cam_,
                                           shadow_class, 
                                           current_file)
            
        return cam_
    
    def get_pseudolabel(self, 
                        current_logits,
                        current_cams,
                        current_img,
                        current_file,
                        current_features,
                        remove_shadow = True,
                        aggregate = True,
                        shadow_class = None):
        try:
            current_cams = np.squeeze(current_cams)
        except:
            current_cams = current_cams.squeeze()
            
        # get indexes where we have prediction above threshold
        idx = self.get_confident_predictions(current_logits)

        if len(idx) == 0:
            # if no preds then we return empty mask
            cam_ = np.full(current_cams[0].shape, 255, dtype = np.float32)
            cam_ = self.upsample_cam(cam_)
            return cam_

        self.shadows = None
        
        # if p == cleared then  leave shadows
        class_cams = [self.prepare_cam(current_cams[p],
                                       current_features, 
                                       remove_shadow if p != 4 else False, 
                                       shadow_class, 
                                       current_file) 
                      for p in idx]
        
        if self.shadows:
            class_cams.append(shadows)

        # combine all class masks into one singular output mask
        if aggregate:
            output_mask = self.aggregate_cams(shadow_class, 
                                              current_file, 
                                              class_cams, 
                                              idx
                                              )

        else:
            # dictionary where the key gives the class index
            # and the value is the class mask
            output_mask = {cl : m for cl, m in zip(idx, class_cams)}
        
        return output_mask
    
    def __call__(self, 
                 img_batch, 
                 filename = None, 
                 remove_shadow = False,
                 shadow_class = None,
                 aggregate = True,
                 n_class = 5):
        """
        Runs the main processing steps to generate an SEM CAM (SCAM).
        
        Parameters
        ----------
        img_batch : torch.tensor(shape = (b, c, w, h))
            Batch of Tensors which have been pre-processed for prediction by 
            the model.
        filename : optional, array-like
            Only needed if the remove_shadow parameter is not None so we can
            check if Cleared is in the filename (since it is better not to 
            take shadow pixels from these types of images).
        remove_shadow : bool
            Whether to remove shadow areas from the SCAM areas.
        take_confident : optional, float
            If provided as a float, then only SCAM pixels with a confidence 
            higher than the float value will be kept.
            
        """ 
        # process the img_batch through the model
        logits, cams, features = self.process_batch(img_batch)
        
        # check if we have a batch of size of 1
        # if so we need to unsqueeze, otherwise we loop on
        # the wrong dimension
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        
        batch_size = logits.shape[0]
        cam_batch = [self.get_pseudolabel(logits[i],
                                          cams[i],
                                          img_batch[i],
                                          filename[i],
                                          features[i],
                                          remove_shadow,
                                          aggregate,
                                          shadow_class) 
                     for i in range(batch_size)]
        
        return cam_batch
            
    def save_cams(self,
                  loader,
                  generator,
                  out_dir,
                  device,
                  shadow_class = None,
                  remove_shadow = True,
                  verbose = True):
        """Takes a generator and extracts CAMs from each image loaded.
        """
        counter = 0
        
        for loaded in get_loaded(generator):
            
            img_batch = loaded[0]
            lbl_batch = loaded[1]
            area_btch = loaded[2]
            name_ = loaded[3]
            img_batch = img_batch.to(device)

            counter += img_batch.shape[0]
            print('Images processed:', counter)
            
            # process batch through network
            mask_batch = self.__call__(img_batch, 
                                       name_, 
                                       remove_shadow = remove_shadow, 
                                       shadow_class = shadow_class)

            for i, mask in enumerate(mask_batch):
                with rasterio.open(name_[i]) as src:
                    profile = src.meta
                    profile['height'] = self.height
                    profile['width'] = self.width
                    profile['count'] = 1
                    profile['driver'] = 'GTiff'

                outfile = out_dir + '/%s.%s' % (Path(name_[i]).stem, 'tif')
                with rasterio.open(outfile, 'w', **profile) as dst:
                    dst.write(mask.astype(rasterio.uint8), 1) 
                    