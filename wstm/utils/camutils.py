import os
import glob
import torch
import rasterio
import cv2 as cv
import numpy as np
import torch.nn as nn
from pathlib import Path
from rasterio.crs import CRS
from .inputAssertions import *
from .image_utils import upsample
from ..models.utils import sem_seeds
from .file_utils import delete_files_from_dir
from torch.nn.functional import cosine_similarity
from torchvision.transforms.functional import to_pil_image
from .image_utils import load_image, threshold_shadows, get_img_meta

# default coordinate reference system for the dataset
crs = CRS.from_wkt('PROJCS["unknown",GEOGCS["unknown",DATUM["Unknown_based_on_GRS80_ellipsoid",SPHEROID["GRS 1980",6378137,298.257222101004,AUTHORITY["EPSG","7019"]]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]')
    
    
def save(mask_batch, filenames, batch_xform, outd, crs, 
         dtype = np.uint8, shape = (304, 304)):
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
    '''
    if not isinstance(mask_batch, np.ndarray):
        mask_batch = np.array(mask_batch)
        
    n_masks_batch = mask_batch.shape[0]
    
    for b in range(n_masks_batch):
        fn = Path(filenames[b]).stem
        
        # get img metadata
        meta = get_img_meta(crs, shape, dtype, batch_xform[b])  
            
        outpath = os.path.join(outd, fn + '.tif')
        out_arr = mask_batch[b, :, :].astype(dtype)
        with rasterio.open(outpath, 'w', **meta) as dest:
            dest.write(np.expand_dims(out_arr, 0))     
                
class PseudoCreator():
    """Generates pseudolabels from class localization maps.
    
    Parameters
    ----------
    model : nn.Module
        A trained model with weights loaded.
    input_shape : tuple
        Shape of the input images.
    threshold_cam : float
        Threshold value for the normalized CLM. Any pixels
        above this threshold will be kept, those below will 
        be set to zero.
    threshold_pred : float
        Threshold for the model predictions. Any classes above
        this theshold will be kept for the pseudo-label, while
        classes predited with probability lower will not.
    use_enhanced : bool
        Whether to use an enhanced version of CAMs. For example, 
        using eSEM or PCM.
    manual_sem : int
        Number of K seed points to use for SEM enhancement of CAM.
        Thus, if we set it to 0 we will not use SEM, whereas if we
        set it to anything above zero we will apply SEM.
        
    Note
    ----
    'use_enhanced' and 'manual_sem' cannot both be true. This is 
    because if we set 'use_enhanced' to be true then we already have 
    an enhanced version of the CAM and applying SEM to this wouldn't
    make sense.
    """

    def __init__(self, 
                 model, 
                 input_shape = (4, 304, 304),
                 threshold_cam = 0.9,
                 threshold_pred = 0.9,
                 use_enhanced = True,
                 manual_sem = 0
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
       
    def upsample_cam(self, clm):
        # if cam already the size desired no upsample
        if clm.shape[-1] == self.width:
            return clm
        # if not then upsample
        else:
            out_shape = (self.height, self.width)
            
            return np.array(upsample(clm,
                                     shape = out_shape, 
                                     interpolation = cv.INTER_CUBIC, 
                                     dtype = np.uint8))
        
    def process_batch(self, img_batch):
        """Runs inference on a batch of images.
        """
        if self.model.train():
            self.model.eval()
            
        out = self.model(img_batch.float())
        
        logits = out[1]
        features = out[-1]
        
        if self.use_enhanced:
            clms = out[2]
        else: clms = out[0]

        return logits, clms, features
        
    def get_confident_predictions(self, current_logits):
        """Finds the class indexes for classes predicted above 
        given threshold.
        """
        idx = torch.where(self.sig(current_logits).flatten() > self.pred_thres)
        idx = idx[0].cpu().numpy().tolist()
        
        return idx
    
    def prepare_cam(self, 
                    clm_, 
                    confident_idx,
                    current_features, 
                    remove_shadow, 
                    current_file,
                    current_idx
                   ):
        """Gets a single CLM ready for being used as mask. Upsamples,
        thresholds, and removes shadows.
        
        Parameters
        ----------
        clm_ : torch.tensor(shape = (width, height))
            A class localization map. Width and height will depend on the 
            amount of downsampling which was applied in the model.
        confident_idx : list
            List of the class indices which he model has predicted with a 
            probability above the threshold set when creating the CamCreator
            class object.
        current_features : torch.tensor(shape = (N, width, height))
            The final feature maps from the model used to generate the CLM.
            Has the same width and height as the 'cam_', and N depends on the
            number of filters in the final layer of the model.
        remove_shadow : bool
            Whether to remove shadow pixels from the CLM.
        current_file : str
            Full path and file name of the original image which the CLM corresponds
            to.
        current_idx : int
            The class index corresponding to the current CLM.
        """
        
        if self.manual_sem:
            clm_ = sem_seeds(clm_, current_features, self.manual_sem)
        
        clm_ = self.upsample_cam(clm_.detach().cpu().numpy())
        
        if current_idx not in confident_idx:
            return np.zeros(clm_.shape)
        
        # threshold pixels with high confidence
        if self.thresh is not None:
            clm_ = np.where(clm_ >= self.thresh,
                            clm_,
                            0).astype(np.float32)

        # threshold shadows
        if remove_shadow:
            clm_, _ = threshold_shadows(clm_, current_file)
            
        return clm_
    
    def empty_clm(self, clm):
        """Creates an array full of 255 (no data value).
        """
        clm_ = np.full(clm.shape, 255, dtype = np.float32)
        clm_ = self.upsample_cam(clm_)
        return clm_
    
    def get_pseudolabel(self, 
                        current_logits,
                        current_cams,
                        current_img,
                        current_file,
                        current_features,
                        remove_shadow = True,
                        aggregate = True):
        """Runs all of the steps required to create a psuedolabel from a raw CLM
        """
        try:
            current_cams = np.squeeze(current_cams)
        except:
            current_cams = current_cams.squeeze()
            
        # get indexes where we have prediction above threshold
        idx = self.get_confident_predictions(current_logits)

        if len(idx) == 0:
            return self.empty_clm(current_cams[0])
        
        # if p == cleared then leave shadows
        class_cams = [self.prepare_cam(current_cams[p],
                                       idx,
                                       current_features, 
                                       remove_shadow if p != 4 else False,
                                       current_file,
                                       p) 
                      for p in range(current_cams.shape[0])]
        
        class_cams = np.array(class_cams)
        
        # combine all class masks into one singular output mask
        if aggregate:
            output_mask = np.where(np.max(class_cams, axis = 0) == 0, 
                                   255,
                                   np.argmax(class_cams, axis = 0))

        else:
            # dictionary where the key gives the class index
            # and the value is the class mask
            output_mask = {cl : m for cl, m in zip([i for i in range(len(class_cams))], 
                                                   class_cams)
                          }
        
        return output_mask
    
    def __call__(self, 
                 img_batch, 
                 filename = None, 
                 remove_shadow = False,
                 aggregate = True):
        """
        Runs the main processing steps to generate an SEM CAM (SCAM).
        
        Parameters
        ----------
        img_batch : torch.tensor(shape = (b, c, w, h))
            Batch of Tensors which have been pre-processed for prediction by 
            the model. Parameter b = batch size, c = number image channels,
            w = width, h = height.
        filename : optional, array-like
            Only needed if the remove_shadow parameter is not None so we can
            check if Cleared is in the filename (since it is better not to 
            take shadow pixels from these types of images).
        remove_shadow : bool
            Whether to remove shadow areas from the CLM areas.
        aggregate : bool
            Whether to combine all CLMs into a single mask.
            
        """ 
        # process the img_batch through the model
        logits, clms, features = self.process_batch(img_batch)
        
        # check if we have a batch of size of 1
        # if so we need to unsqueeze, otherwise we loop on
        # the wrong dimension
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        
        batch_size = logits.shape[0]
        clm_batch = [self.get_pseudolabel(logits[i],
                                          clms[i],
                                          img_batch[i],
                                          filename[i],
                                          features[i],
                                          remove_shadow,
                                          aggregate) 
                     for i in range(batch_size)]
        
        return clm_batch
            
    def save_cams(self,
                  generator,
                  out_dir,
                  device,
                  remove_shadow = True,
                  verbose = True):
        """Takes a generator and extracts CLMs from each image loaded.
        """
        counter = 0
        
        for loaded in generator:
            
            img_batch = loaded[0]
            lbl_batch = loaded[1]
            area_btch = loaded[2]
            name_ = loaded[3]
            xform = loaded[4]
            img_batch = img_batch.to(device)

            counter += img_batch.shape[0]
            print('Images processed:', counter)
            
            # get CLMs
            mask_batch = self.__call__(img_batch, 
                                       name_, 
                                       remove_shadow = remove_shadow)

            save(mask_batch, name_, xform, out_dir, crs, 
                 dtype = np.uint8, shape = (self.height, self.width))
                    