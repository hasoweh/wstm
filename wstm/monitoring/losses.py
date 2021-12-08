import torch
import torch.nn as nn
from torch.nn import functional as F
from ..utils.inputAssertions import assert_tensor
    
class AreaAwareLoss():
    """Calculates binary cross entropy loss using the area covered by a class
    in each image as a weighting factor.
    """
        
    def __call__(
        self,
        preds,
        targets,
        area_weights,
        multi_class_weights = None,
        reduction = 'mean'
        ):
        
        assert_tensor(preds, 'preds')
        assert_tensor(targets, 'targets')
        assert_tensor(area_weights, 'area_weights')
        if multi_class_weights is not None:
            assert_tensor(multi_class_weights, 'multi_class_weights')
        
        # get BCE loss
        criterion = nn.BCEWithLogitsLoss(reduction = 'none')
        loss_val = criterion(preds, targets.type(torch.float))#.to(self.device)
        
        # apply area weights to the loss values
        loss_val = loss_val * area_weights
        
        # Apply class imbalance weights
        if multi_class_weights is not None:
            loss_val = loss_val * multi_class_weights
        
        if reduction == 'mean':
            return loss_val.mean()
        elif reduction == 'none':
            return loss_val
        
class AreaAwareMulti():
    """Calculates AreaAware loss for multiple inputs.
    Used to train the PCM module and the classification
    head.
    """
    
    def __init__(self, loss_term_weights = [.9, .1]):
        self.loss_term_weights = loss_term_weights
    
    def __call__(
        self,
        preds,
        targets,
        area_weights,
        multi_class_weights = None,
        reduction = 'mean'
        ):
        
        # get the total number of losses
        n_losses = len(preds)
        # calc each loss
        loss = 0
        for w, pred in zip(loss_term_weights, preds):
            loss += w * AreaAwareLoss()(pred, 
                                        targets,
                                        area_weights,
                                        multi_class_weights = None,
                                        reduction = 'mean')

        # get mean of the losses
        L = loss / n_losses
        
        return L
