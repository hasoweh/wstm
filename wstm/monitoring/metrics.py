import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import (precision_score, 
                             recall_score, 
                             f1_score,
                             hamming_loss,
                             coverage_error,
                             label_ranking_loss,
                             label_ranking_average_precision_score,
                             zero_one_loss,
                             f1_score,
                             average_precision_score,
                             mean_squared_error
                             )
# used to avoid dividing by zero
eps = 0.000000001

def overall_acc(preds, labels):
    """Gets the percent of correctly predicted pixels.
    """
    def calc_total_pixels(shape):
        total = 0
        for num in shape:
            if total == 0:
                total += num
            else:
                total *= num
        return total
    
    tp = len(np.where(preds == labels)[0])
    total = calc_total_pixels(labels.shape)
    
    return tp / total

def precision_recall_metrics(preds, labels):
    # transpose so each class column becomes a row
    preds = preds.transpose(1,0)
    labels = labels.transpose(1,0)

    # Calculate the true positive, negative, ect.
    eps = 0.0000000008
    tp = np.sum(preds * labels, axis = 1)
    fn = abs(np.sum((preds - 1) * labels, axis = 1))
    fp = abs(np.sum((labels - 1) * preds, axis = 1))

    # precision, recall, f1, and averaged f1 scores
    # classwise precision
    precision = tp/(tp + fp+eps)
    # classwise recall
    recall = tp/(tp+fn+eps)
    # classwise f1
    f1 = 2 * ((precision * recall) / \
              (precision + recall + eps)
              )
    #classwise jaccard
    jaccard = tp / (tp + fp + fn)

    # get the average (across classes) F1 and jaccard
    macro_f1 = np.mean(f1)
    macro_jaccard = np.mean(jaccard)
    macro_r = np.mean(recall)
    macro_p = np.mean(precision)

    return precision, recall, f1, jaccard, macro_f1, macro_jaccard, macro_r, macro_p

class EpochMetrics():
    """
    Calculates the metrics for all predictions/labels in a single epoch.
    
    Parameters
    ----------
    labels : array
        Ground truth labels.
    preds : array
        Array with binary entries determining which classes are predicted.
    probs : array
        The probability scores calculated by either Softmax or Sigmoid.
    classes : array
        Names of the classes being predicted.
    print_classwise : bool
        Whether to print out classwise metrics or not.
    """
    def __init__(self, labels, preds, probs, classes, print_classwise = True):
        
        self.inputs = [labels, preds, probs]
        self.classes = classes
        
        self.prepare_inputs()
        
        self.labels = self.inputs[0]
        self.preds = self.inputs[1]
        self.probs = self.inputs[2]
        
        self.print_classwise = print_classwise

    def prepare_inputs(self):
        '''Checks input and puts them into proper data structures.
        '''
        for i, inp in enumerate(self.inputs):
            # convert inputs to np.array
            if not isinstance(inp, np.ndarray):
                inp = np.array(inp)
            # concatenate the arrays if they are 3D 
            # (we check if the first element is 2D, if so then there is a 3rd D)
            if len(inp[0].shape) == 2:
                self.inputs[i] = np.concatenate(inp)
            # make sure the number of columns match to the number of classes
            m = 'Sub-list at index %i should have the same number of columns as the number of classes (%i)'
            assert len(self.classes) == len(self.inputs[i][1]), m  % (i, len(self.classes))
    
    def sklearn_metrics(self):
        # get the errors from sklearn
        self.hamming = hamming_loss(self.labels, self.preds)
        self.coverage = coverage_error(self.labels, self.probs)
        self.lrap = label_ranking_average_precision_score(self.labels, self.probs)
        self.ranking_loss = label_ranking_loss(self.labels, self.probs)
        self.zero_one = zero_one_loss(self.labels, self.preds)
        self.sample_f1 = f1_score(self.labels, self.preds, average='samples')
        self.macro_map = average_precision_score(self.labels, self.probs, average= 'macro')
        self.micro_map = average_precision_score(self.labels, self.probs, average= 'micro')
    
    def verbose(self):
        # option to print the values on screen
        if self.print_classwise:
            for i in range(len(self.classes)):
                print(self.classes[i])
                print('Recall: ', self.recall[i])
                print('Precision: ', self.precision[i])
                print('F1: ', self.f1[i])
                print('Jaccard', self.jaccard[i])
                print()
            
        print('Macro F1:', self.macro_f1)
    
    def __call__(self):
        self.precision, self.recall, self.f1, self.jaccard, \
        self.macro_f1, self.macro_jaccard, self.macro_r, \
        self.macro_p = precision_recall_metrics(self.preds, self.labels)
        
        self.sklearn_metrics()
        self.verbose()
        
        output = [self.macro_f1, 
                  self.f1, 
                  self.recall, 
                  self.precision, 
                  self.jaccard, 
                  self.macro_jaccard, 
                  self.hamming, 
                  self.coverage, 
                  self.lrap, 
                  self.ranking_loss,
                  self.zero_one,
                  self.sample_f1, # micro f1
                  self.macro_r,
                  self.macro_p,
                  self.macro_map,
                  self.micro_map
                 ]
    
        return output