import numpy as np
import torch

def get_class_weights(config):
    class_weights = config['class_imbal_weights']
    total = np.sum(class_weights)
    weights = torch.tensor(np.array(class_weights) / total)
    weights = 1.0 / weights
    class_weights = weights / weights.sum()
    class_weights = class_weights.float()
    
    return class_weights