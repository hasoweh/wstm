"""Helper and main functions for the DSRG layer"""
import multiprocessing as mp
from . import crf, CC
import torch.nn as nn
import numpy as np
import torch

def refine_softmax(softmax, img, iterations=5, scale = 1):
    """Apply CRF to softmax layer"""
    min_prob = 0.00001

    n_class = softmax.shape[0]
    softmax[softmax < min_prob] = min_prob
    
    crf_ = crf.crf_inference(softmax,
                             img.permute(1,2,0),
                             n_iters=iterations,
                             sxy_gaussian=(3, 3), compat_gaussian=3,
                             sxy_bilateral=(49, 49), compat_bilateral=4,
                             srgb_bilateral=(5, 5, 5), 
                             n_classes = n_class
                             )
    
    crf_[crf_ < min_prob] = min_prob
    softmax_refined = crf_ / np.sum(crf_, axis=2, keepdims=True)
    
    return softmax_refined # returns with c, h, w

def get_max_prob_and_corresponding_class(img_lvl_labels, softmax_refined):
    """Finds the maximum probability value at each pixel location, and also
    gets the class which is associated with the maximum probability
    """
    cls_index = np.where(img_lvl_labels == 1)[0]
    # takes only the softmax layers of classes present
    probs_selected = softmax_refined[cls_index]
    # get class with the highest probability at each pixel location
    #probs_c = torch.argmax(probs_selected, dim=0)
    # get maximum probability at each pixel location
    probs_p = np.max(probs_selected, axis = 0)
    probs_c = np.argmax(probs_selected, axis = 0)

    return probs_c, probs_p, cls_index

def find_confident_seeds(seeds_cam, prob_thresh = 0.5):
    
    # threshold seed probabilities below some threshold
    seeds_cam = seeds_cam 
    seed_confident = np.where(seeds_cam > prob_thresh,
                              seeds_cam,
                              0)
    
    # find the max value for each dim
    max_value_along_dim = np.max(seed_confident, axis=0, keepdims=True)
    
    # set values that are max to 1, rest to 0
    seed_confident = np.where(np.logical_and(seed_confident == max_value_along_dim, 
                                             seed_confident != 0),
                              1,
                              0)
    return seed_confident
    

def get_initial_seeds(cam, img_lvl_labels):
    cam = cam # n_class,h,w
    img_lvl_labels = img_lvl_labels.reshape(img_lvl_labels.shape[0],
                                            1, 
                                            1 
                                            )# n_class,1,1

    seeds_for_present_classes = cam * img_lvl_labels
    seed_confident = find_confident_seeds(seeds_for_present_classes, 0.5)
    
    return seed_confident # array with values of either 1 or 0

def get_label_map(probs_c, probs_p, cls_index, thr, seed_confident, device):
    """Label map tries to add more labels to the intial seed labels based on softmax 
    probabilities that are higher than the threshold. 
    """
    h, w = seed_confident.shape[1:]

    # get index locations where the seed is a 1
    index = np.where(seed_confident == 1)
    label_map = np.full((h, w), 255, dtype = np.uint8) # full 255 since this is ignore label

    label_map[index[1], index[2]] = index[0] # index[0] gives us the class index value

    for (x,y), value in np.ndenumerate(probs_p): # get the max prob value at each pixel
        c = cls_index[probs_c[x,y]] # get class corresponding to the probability at the pixel
        if value > thr: # if prob > threshold then we assign the class label. Original paper had 2 thresholds (background and foreground), but we don't have BG class so we only use 1
            label_map[x,y] = c
            
    return label_map

def grow_seeds(label_map, cls, seed_confident):
#def grow_seeds(inputs):
    """We have the seed layer which came from the CAMs, and then the layer map
    which expanded the seed layer based on softmax probabilities. Now, we look
    for areas in the label map which were added. If the added area is also connected
    to an original seed area for the class, then we keep the added area, otherwise
    the added area is removed and we only use the seed area for that class.
    
    label_map: array, shape = (h,w)
    cls: int, class index value
    seed_confident: array, shape = (n_class, h, w)
    """
    #label_map, cls, seed_confident = inputs
    bool_mask = (label_map == cls)
    
    bool_mask = np.squeeze(bool_mask).astype(int)

    cclab = CC.CC_lab(bool_mask)
    cclab.connectedComponentLabel()
    
    high_confidence_set_label = set()
    for (x,y), value in np.ndenumerate(bool_mask):
        # so value comes from label_map, and label map comes from seed_confident
        # but since label_map can be expanded by the softmax probs, it can
        # differ from seed_confident at this point
        if value == 1 and seed_confident[cls, x, y] == 1:
            high_confidence_set_label.add(cclab.labels[x][y])
        # check if the sum of seed_confident across the class dimension is 1
        # essentially it checks if there is another class confidently predicted
        # to be at this x,y location. 
        elif value == 1 and np.sum(seed_confident[:, x, y]) == 1:
            cclab.labels[x][y] = -1

    for (x,y), value in np.ndenumerate(np.array(cclab.labels)):
        if value in high_confidence_set_label:
            seed_confident[cls, x, y] = 1
    
    return seed_confident

#def generate_single_batch_seed(cam, img_lvl_labels, softmax_refined, img, device, thr = 0.85):
def generate_single_batch_seed(inputs):
    """Generates the seeds for a single image in the batch. 
    """
    cam, img_lvl_labels, softmax_refined, img, device, thr = inputs
    
    seed_confident = get_initial_seeds(cam, img_lvl_labels)

    # get max class, max prob, and an array of which classes are present in img_lvl_lbl
    probs_c, probs_p, cls_index = get_max_prob_and_corresponding_class(img_lvl_labels, 
                                                                       softmax_refined)

    label_map = get_label_map(probs_c, probs_p, cls_index, thr, seed_confident, device)

    # think I need multiprocess here
    #seed_confident = multi_process_seeds(cls_index, label_map, seed_confident)
    for cls in cls_index:
        seed_confident = grow_seeds(label_map, cls, seed_confident)
    
    pseudolabel = np.where(np.max(seed_confident, axis = 0) == 0,
                           255,
                           np.argmax(seed_confident, axis = 0))

    return torch.from_numpy(pseudolabel)

def multi_process_seeds(cls_index, label_map, seed_confident):
    """We use multi-processing here for the grow_seeds function since it
    is quite slow and we need to do it for each class in the image.
    """
    set_start_method('forkserver', force=True)
    
    maps = [label_map.clone().detach().cpu() for _ in range(len(cls_index))] # create copies of the label map
    seeds = [seed_confident.clone().detach().cpu() for _ in range(len(cls_index))] # create copies of the seeds map
    inputs = zip(maps, cls_index.tolist(), seeds)
    
    pool = Pool(processes=len(cls_index))
    results = pool.map(grow_seeds, inputs) # returns the updated seed layer and the cls index of the updated layer
    
    for res in results:
        cls = res[1]
        seed_confident[cls] = res[0]
    
    return seed_confident

class DSRGLayer(nn.Module):
    
    def __init__(self, threshold = 0.85, device = "cpu"):
        super().__init__()
        self.thr = threshold
        self.device = device
        
    def prepare_inputs(self, cams, labels, probs, imgs):
        b = cams.shape[0]
        cams = [cam.cpu().numpy() for cam in cams]
        labels = [label.cpu().numpy() for label in labels]
        probs = [prob for prob in probs]
        imgs = [img.cpu().numpy() for img in imgs]
        devices  = [self.device for _ in range(b)]
        thres = [self.thr for _ in range(b)]
        
        return zip(cams, labels, probs, imgs, devices, thres)
        
    def generate_seed(self, labels, probs, cams, imgs):
        """
        im : tensor, should be unnormalized image in uint8.
        """
        batch, c, h, w = probs.shape
        softmax = probs.clone().contiguous().detach().cpu().numpy()
        
        softmax_crf = [refine_softmax(softmax[b], imgs[b], iterations=5, scale = 1) 
                       for b in range(batch)]
        
        # use multi-processing to speed things up
        pool = mp.Pool(batch)
        inputs = self.prepare_inputs(cams, labels, softmax, imgs)
        seed_c = pool.map(generate_single_batch_seed, inputs)
        pool.close()
        
        seed_c = torch.cat([t.unsqueeze(0) for t in seed_c], dim = 0)
        softmax_crf = torch.cat([torch.from_numpy(t).unsqueeze(0) for t in softmax_crf], dim = 0)
        
        return seed_c, softmax_crf

    def forward(self, x):
        im, img_labels, cues, softmax = x[0], x[1], x[2], x[3]

        # Selects NIR + RG for 3 bands in CRF
        im = im[:,:3,:,:]

        seed_c, softmax_crf = self.generate_seed(img_labels, softmax, cues, im)
        
        return seed_c.to(self.device), softmax_crf.to(self.device)