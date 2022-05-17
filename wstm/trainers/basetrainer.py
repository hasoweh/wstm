from ..monitoring.metrics import EpochMetrics
from ..monitoring.losses import AreaAwareLoss
import torch.nn as nn
import numpy as np
import torch
import json
import os

class ModelTrainer():
    
    def __init__(self, 
                 num_epochs, 
                 classes, 
                 model, 
                 device, 
                 loaders, 
                 criterion, 
                 optimizer, 
                 f_name, 
                 scheduler = None, 
                 class_imbal_weights = None, 
                 weights_path = './data/weights', 
                 save_results = None, 
                 print_cl_met = True, 
                 minimum_lr = 0.0): 
        """ 
        Parameters
        ----------
        num_epochs : int
            Number of epochs to train for.
        classes : int
            Names of the classes.
        model : nn.Module
            Model to train.
        device : str
            Device used to train the model. Either 'cpu' or 'cuda'.
        loaders : dict
            Dictionary containing generator functions used to load data for each batch.
        criterion : function
            A callable loss function.
        optimizer : torch.optim
            Optimizer function.
        f_name : str
            Name the model weights should be saved as.
        scheduler : optional, torch.optim.lr_scheduler
            Scheduler function used to modify the learning rate during training.
        class_imbal_weights : optional, np.ndarray(shape = (n))
            Array containing the class weights.
        weights_path : str
            Path where to save the model weights to.
        print_cl_met : bool
            Whether to print the validation scores of each class
            after each epoch.
        minimum_lr : float
            Minimum allowed learning rate value in case that we use
            a scheduler for decreasing.
        """
        self.print_class_metrics = print_cl_met
        self.w_path = weights_path
        self.epochs = num_epochs
        self.model = model
        self.device = device
        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.file_name = f_name
        self.classes = classes
        self.weights = class_imbal_weights
        self.sig = nn.Sigmoid()
        
        # get name of scheduler function
        self.sched_name = type(self.scheduler).__name__
        # check if it is currently implemented in this class
        allowed_sched = ['StepLR', 'ExponentialLR', 'CyclicLR', 
                         'ReduceLROnPlateau', 'NoneType']
        if self.sched_name not in allowed_sched:
            m = "Scheduler %s is not currently supported. See 'trainModel' for how the others are implemented so you can add %s appropriately."
            raise TypeError(m % (self.sched_name, self.sched_name))
        
        self.current_epoch = 0
        self.minimum_lr = minimum_lr
        # loss values after each batch
        self.train_loss_per_batch = []
        self.val_loss_per_batch = []
        # loss averaged across the epoch
        self.training_losses = []
        self.val_losses = []
        
        # set tracking lists for metrics
        self.logits_tracker = []
        self.f1_tracker = []
        self.classwise_f1 = []
        self.classwise_r = []
        self.classwise_p = []
        self.classwise_j = []
        self.jaccard_tracker = []
        self.hamming_tracker = []
        self.coverage_tracker = []
        self.lrap_tracker = []
        self.ranking_loss_tracker = []
        self.zero_one_tracker = []
        self.sample_f1_tracker = []
        self.macro_recall_tracker = []
        self.macro_prec_tracker = []
        self.macro_map_tracker = []
        self.micro_map_tracker = []
        self.loss_tracker = []
        
    @property
    def current_lr(self):
        return self.optimizer.param_groups[0]['lr']
            
    def save_weights(self):  
        if self.f1_tracker[-1] == np.max(np.array(self.f1_tracker)):
            torch.save(self.model.state_dict(), 
                       self.w_path + '/%s.pt' % self.file_name)
            print('Saving weights')
            print()

    def init_epoch(self, i):
        print('*' * 50)
        print('*' * 50)
        print('EPOCH: %d' % int(i+1))
        print('*' * 50)
        print('*' * 50)

        self.epoch_preds_val = []
        self.epoch_labels_val = []
        self.epoch_probabilites = []
    
    def get_loaded(self, loaded):
        img_batch = loaded[0]
        label_batch = loaded[1]
        area_batch = loaded[2]
        
        img_batch = img_batch.to(self.device)
        lbl_batch = label_batch.to(self.device)
        area_batch = area_batch.to(self.device)
        
        return img_batch, lbl_batch, area_batch
    
    def getLoss(self):
        # if we have multiple logits send them to device
        if isinstance(self.out, list) and len(self.out) > 1:
            self.out = [o.to(self.device) for o in self.out]
        else:
            self.out = self.out.to(self.device)
        
        return self.criterion(self.out, 
                              self.lbl_batch,
                              self.area_batch,
                              self.weights
                                 )
    
    def model_mode(self):
        if self.phase == 'training':
            print('*********TRAINING PHASE*********')
            self.trainModel()
        else:
            print('*********VALIDATION PHASE*********')
            self.validateModel()

    def init_phase(self):
        # Turn on training mode
        self.model.train(True if self.phase == 'training' else False)

        # empty list to track losses in the training epoch
        return []
    
    def trainModel(self):
        """Runs a training loop for a single epoch.
        """

        epoch_loss = self.init_phase()

        if self.scheduler is not None:
            print('Current LR: %.4f' % self.current_lr)

        # loop through all batches to perform an epoch
        for i, loaded in enumerate(self.loaders[self.phase]):
            epoch_loss = self.train_step(loaded, epoch_loss)
        
        mean_loss = np.mean(np.array(epoch_loss))
        print("*" * 20)
        print('Training loss for epoch %d : ' % int(self.current_epoch + 1), 
              mean_loss)
        print()
        self.loss_tracker.append(mean_loss)

        # track the final logits in the epoch
        self.track_outputs()

        # update scheduler
        if self.sched_name in ['StepLR', 'ExponentialLR']:
            if self.scheduler.get_last_lr()[0] > self.minimum_lr:
                self.scheduler.step()

        self.training_losses.append(mean_loss)

    def train_step(self, loaded, epoch_loss):
        '''Performs a single batch step in a training epoch'''
        
        self.img_batch, self.lbl_batch, self.area_batch = self.get_loaded(loaded)
        
        self.optimizer.zero_grad()

        # process batch through network
        self.out = self.model(self.img_batch.float())
        loss_val = self.getLoss()

        # track loss for the given epoch
        epoch_loss.append(loss_val.item())
        self.train_loss_per_batch.append(loss_val.item())

        loss_val.backward()
        self.optimizer.step()

        # cyclic updates across batchs
        if self.sched_name in ['CyclicLR']:
            self.scheduler.step()
    
        return epoch_loss
    
    def track_outputs(self):
        '''Stores output predictions in a list. Can be overwritten
        to allow for tracking of additional outputs.
        E.g. Multi-task-learning.
        '''
        pass
        #self.logits_tracker.append(self.out.detach().cpu().numpy())
        
    def validateModel(self):
        """Runs a validation loop for a single epoch.

        """
        epoch_loss = self.init_phase()

        # loop through all batches
        with torch.no_grad():

            for loaded in self.loaders[self.phase]:
                epoch_loss = self.valStep(loaded, epoch_loss)
                
            print("*" * 20)
            print('Validation loss for epoch %d : ' % int(self.current_epoch + 1), 
                  np.mean(np.array(epoch_loss)))
            print()

            self.val_losses.append(np.mean(np.array(epoch_loss)))
            
    def valStep(self, loaded, epoch_loss):
        self.img_batch, self.lbl_batch, self.area_batch = self.get_loaded(loaded)

        # process batch through network
        self.out = self.model(self.img_batch.float())
        
        if len(self.out) > 1 and isinstance(self.out, list):
            if type(self.model).__name__ == 'MSGSRNet':
                # use the logits from the last layer for MSGSR
                self.pred_out = np.array(self.sig(self.out[-1]).cpu() > 0.5, dtype=float)
            else:
                self.pred_out = np.array(self.sig(self.out[0]).cpu() > 0.5, dtype=float)
        else:
            self.pred_out = np.array(self.sig(self.out).cpu() > 0.5, dtype=float)

        # store in array 
        self.store_val_output()

        # calculate loss
        loss_val = self.getLoss()

        # track loss
        epoch_loss.append(loss_val.item())
        self.val_loss_per_batch.append(loss_val.item())
        
        return epoch_loss
        
    def store_val_output(self):
        # if we have multiple logit outputs (e.g. PCM)
        if len(self.out) > 1 and isinstance(self.out, list):
            self.epoch_probabilites.append(self.out[0].cpu().numpy())
            
        # if only 1 logit output
        else:
            self.epoch_probabilites.append(self.out.cpu().numpy())
            
        self.epoch_labels_val.append(self.lbl_batch.cpu().numpy())
        self.epoch_preds_val.append(self.pred_out)
      
    def run(self):
        for i in range(self.epochs):
            self.init_epoch(i)
        
            # switch between training and validation 
            for phase in ['training', 'validation']:
                self.phase = phase
                self.model_mode()

                if self.phase == 'validation':
                    self.store_metrics()
                    if self.sched_name == 'ReduceLROnPlateau':
                        self.scheduler.step(self.f1_tracker[-1])
            # if highest F1 score so far then we save the weights
            self.save_weights()
            
            self.current_epoch = int(i+1)
        
        return self.model

    def store_metrics(self):
        metrics_ = EpochMetrics(self.epoch_labels_val, 
                                self.epoch_preds_val, 
                                self.epoch_probabilites,
                                self.classes,
                                self.print_class_metrics)()

        self.f1_tracker.append(metrics_[0])
        self.classwise_f1.append(metrics_[1])
        self.classwise_r.append(metrics_[2])
        self.classwise_p.append(metrics_[3])
        self.classwise_j.append(metrics_[4])
        self.jaccard_tracker.append(metrics_[5])
        self.hamming_tracker.append(metrics_[6])
        self.coverage_tracker.append(metrics_[7])
        self.lrap_tracker.append(metrics_[8])
        self.ranking_loss_tracker.append(metrics_[9])
        self.zero_one_tracker.append(metrics_[10])
        self.sample_f1_tracker.append(metrics_[11])
        self.macro_recall_tracker.append(metrics_[12])
        self.macro_prec_tracker.append(metrics_[13])
        self.macro_map_tracker.append(metrics_[14])
        self.micro_map_tracker.append(metrics_[15])