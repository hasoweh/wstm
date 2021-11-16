from .inputAssertions import *
from rasterio import Affine
import numpy as np
import cv2 as cv
import rasterio

def load_image(fname):
    """Loads an image from file
    """
    assert_str(fname, "fname")
    with rasterio.open(fname) as src:
        img = src.read()
    return img

def get_img_meta(crs, shape, dtype, xform_matrix):
    """Creates image metadata for rasterio format.
    """
    x = xform_matrix
    if crs is not None:
        xform = Affine(x[0], x[1], x[2], x[3], x[4], x[5])
    else:
        xform = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    return {'driver': 'GTiff', 
            'dtype': dtype, 
            'nodata': 255, 
            'width': shape[1], 
            'height': shape[0],
            'count': 1, 
            'transform': xform,
            'crs' : crs
            }   

def upsample_multiband(imgarr,
                       shape, 
                       interpolation, 
                       dtype):
    """Upsamples multiple bands of an image at once.
    """
    
    bands = shape[0]
    if bands <= 1:
        m = """First element of 'shape' should be more than 1 for multi-band,
        found %d"""
        raise AssertionError(m % bands)
        
    img_out = [cv.resize(imgarr[i,:,:], 
                         (shape[1], shape[2]), 
                         interpolation=interpolation)
               for i in range(bands)]
    
    return np.array(img_out)

def upsample(imgarr,
             shape = (4, 304, 304), 
             interpolation = cv.INTER_CUBIC, 
             dtype = np.uint8):
    """Performs upsampling of any 2- or 3-D image array
    
    Parameters
    ----------
    imgarr : np.ndarray()
        An array to be upsampled.
    shape : tuple
        Desired dimensions of the upsampled image.
    interpolation : cv2.function
        A method determining the type of interpolation to use.
    dtype : np.dtype
        The desired data type of the output array.
    """
    
    assert_array(imgarr, "imgarr")
    assert_tuple(shape, "shape")
        
    # if we have multiple bands
    if len(shape) == 3:
        img_out = upsample_multiband(imgarr,
                                     shape, 
                                     interpolation, 
                                     dtype)
    elif len(shape) == 2:
        img_out = cv.resize(imgarr, 
                            shape, 
                            interpolation=interpolation)
    else:
        raise AssertionError("Length of given 'shape' is too long, can only support 3 axes.")
    
    return img_out

def threshold_shadows(mask_raster, fname, shadow_class=None):
    
    assert_str(fname, "fname")
    assert_array(mask_raster, "mask_raster")
    if shadow_class is not None:
        assert_int(shadow_class)
    
    # load image from file
    img = load_image(fname)
        
    # take the pixel values where the mask overlaps the original image
    overlapped = img[0,:,:][mask_raster>0]
    
    # find the otsu threshold between shadow and light pixels
    ret2, th2 = cv.threshold(overlapped, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    # multiply the original image by the mask
    masked = img[0,:,:] * mask_raster

    # where ever the mask is above the threshold set the mask 
    # to the idx for that class, everywhere else we set 0
    thresh_mask = np.where(masked > ret2, mask_raster, 0)
    
    if shadow_class is not None:
        # calc the quantile for the shadows
        quant = np.quantile(img[0,:,:], 0.02)
        shadow_mask = np.where(img[0,:,:] < quant, shadow_class, 0)
    
        return thresh_mask, shadow_mask
    
    else:
        return thresh_mask, shadow_class # return None as shadow mask