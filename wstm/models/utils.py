from torch.nn.functional import cosine_similarity
import torch.nn as nn
import torch

def norm_batch(batch, debug = False):
    """Normalize all feature maps within a batch.
    """
    
    batch_shape = batch.size()

    # get the min and maxes for each feature map in the batch
    batch_mins, _ = torch.min(batch.view(batch_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
    batch_maxs, _ = torch.max(batch.view(batch_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
    
    # normalize each feature map within the batch
    batch_normed = torch.div(batch.view(batch_shape[0:-2] + (-1,))-batch_mins,
                             batch_maxs - batch_mins)
    batch_normed = batch_normed.view(batch_shape)
    
    if debug:
        print("batch_shape", batch_shape)
        print("batch_mins", batch_mins)
        print("batch_maxs", batch_maxs)
        #print("batch_shape", batch_shape)
        #print("batch_shape", batch_shape)
        
    
    return batch_normed

def esem(cam, F, n_class):
    """eSEM method.
    """
    # get dimensions of the input feature maps
    n,c,h,w = F.size()
    
    # get similarity matrix
    simi = cosine_similarity(F.view(n, c, -1, 1), 
                             F.view(n, c, 1, -1), 
                             dim=1
        )
    
    # apply to CAM
    cam = torch.matmul(cam.view(n, n_class, h*w), 
                         simi)
    
    # reshape back to matrix
    cam = cam.view(n, n_class, h, w)
    
    # normalize the CAMs
    cam = norm_batch(cam)
    
    return cam

def sem_seeds(cam, F, seed):
    """
    Parameters:
    -----------
    cam : torch.tensor
        Class Activation Map.
    F : torch.tensor
        Multidimensional tensor containing all feature maps at ResNet layer4.
    seed : int
        Number of seeds to use from the original CAM.

    References
    ----------
    Zhang et al., (2020).
    Rethinking localization map: Towards accurate object perception 
    with self-enhancement map.

    """
    #F is high-level feature maps of size (C,W,H)
    #K is the seed number
    c, w, h = F.shape
    _, topk_indices = torch.topk(
                                cam.view(-1),
                                seed,
                                largest=True)

    F_seeds = F.view(c, -1)[:, topk_indices]
    simi = cosine_similarity(F.view(c, -1, 1),
                             F_seeds.view(c, 1, -1), 
                             dim=0
    )

    sem, _ = simi.max(dim=1)
    sem = sem.view(w, h)
    sem = (sem - sem.min())/(sem.max() - sem.min())

    return sem
