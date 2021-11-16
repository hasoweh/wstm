import torch
import numpy as np

def assert_tensor(obj, name):
    if not isinstance(obj, torch.Tensor):
        m = "'%s' must be a tensor, got %s."
        raise TypeError(m % (name, type(obj).__name__))

def assert_array(obj, name):
    if not isinstance(obj, np.ndarray):
        m = "'%s' must be a numpy array, got %s."
        raise TypeError(m % (name, type(obj).__name__))
        
def assert_list(obj, name):
    if not isinstance(obj, list):
        m = "'%s' must be a list, got %s."
        raise TypeError(m % (name, type(obj).__name__))
        
def assert_dict(obj, name):
    if not isinstance(obj, dict):
        m = "'%s' must be a dictionary, got %s."
        raise TypeError(m % (name, type(obj).__name__))
        
def assert_str(obj, name):
    if not isinstance(obj, str):
        m = "'%s' must be a string, got %s."
        raise TypeError(m % (name, type(obj).__name__))
    elif len(obj) == 0:
        m = "Input string for '%s' has length of zero."
        raise TypeError(m % (name, type(obj).__name__))
        
def assert_tuple(obj, name):
    if not isinstance(obj, tuple):
        m = "'%s' must be a tuple, got %s."
        raise TypeError(m % (name, type(obj).__name__))
        
def assert_float(obj, name):
    if not isinstance(obj, float):
        m = "'%s' must be a float, got %s."
        raise TypeError(m % (name, type(obj).__name__))
        
def assert_int(obj, name):
    if not isinstance(obj, int):
        m = "'%s' must be an integer, got %s."
        raise TypeError(m % (name, type(obj).__name__))