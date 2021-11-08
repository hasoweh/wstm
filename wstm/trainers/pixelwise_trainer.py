from sklearn.metrics import jaccard_score
from .basetrainer import ModelTrainer
from ..utils.inputAssertions import *
from rasterio.crs import CRS
from rasterio import Affine
import torch.nn as nn
import numpy as np
import rasterio
import torch
import os

default = CRS.from_wkt('PROJCS["unknown",GEOGCS["unknown",DATUM["Unknown_based_on_GRS80_ellipsoid",SPHEROID["GRS 1980",6378137,298.257222101004,AUTHORITY["EPSG","7019"]]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]')

class PixelwiseTrainer(ModelTrainer):
    
    def __init__(self, 
                 num_epochs,
                 loaders, 
                 model, 
                 device, 
                 criterion, 
                 optimizer, 
                 weights_f_name, 
                 scheduler = None,
                 weights_path = './weaksuper/weights',
                 minimum_lr = 0,
                 crs = default
                ):
        super().__init__(num_epochs, 
                         ['placeholder'], 
                         model, 
                         device, 
                         loaders, 
                         criterion, 
                         optimizer, 
                         weights_f_name, 
                         scheduler, 
                         None, 
                         weights_path, 
                         None, 
                         True, 
                         0.0)
        
        assert_str(weights_path, 'weights_path')
        assert_str(weights_f_name, 'weights_f_name')
        
        self.sm = nn.Softmax(dim=1)
        self.crs = crs
        
        # set tracking lists for metrics
        self.logits_tracker = []
        self.pixacc_tracker = []
        self.miou_tracker = []
        self.classwise_iou_tracker = []
        self.classwise_dice_tracker = []
        self.mean_dice_tracker = []
       
    def save_weights(self):
        if self.val_losses[-1] == np.min(np.array(self.val_losses)):
            print('Saving weights')
            torch.save(self.model.state_dict(), 
                       self.w_path + '/%s.pt' % self.file_name)
            
    def getLoss(self):
        return self.criterion(self.out, 
                              self.mask_batch.long()
                              )

    def init_epoch(self, i):
        print('*' * 50)
        print('*' * 50)
        print('EPOCH: %d' % int(i+1))
        print('*' * 50)
        print('*' * 50)

        self.epoch_masks = []
        self.epoch_preds = []
        self.epoch_probabilites = []
        
    def get_loaded(self, loaded):
        img_batch = loaded[0]
        mask_batch = loaded[1]
        img_batch, mask_batch = img_batch.to(self.device),mask_batch.to(self.device)
        
        self.batch_files = loaded[2]
        self.batch_xform = loaded[3]
        
        return img_batch, torch.squeeze(mask_batch)
    
    def store_metrics(self):
        
        truth = self.mask_batch.detach().cpu().numpy().flatten()
        pred = np.argmax(self.sm(self.out).detach().cpu().numpy(), axis = 1)
        pred = pred.flatten().astype(np.uint8)
        
        # get the labels included in both truth & pred
        lbls = []
        lbls.append(list(np.unique(truth)))
        lbls.append(list(np.unique(pred)))
        # flatten and remove 255 lbl
        lbls = np.unique([i for subl in lbls for i in subl if i != 255])
        
        miou = jaccard_score(truth, 
                             pred, 
                             labels = lbls, 
                             average='macro')
        print('mIOU across %d classes:' % len(lbls), miou)
        self.miou_tracker.append(miou
                                )

    def train_step(self, loaded, epoch_loss):
        '''Performs a single batch step in a training epoch'''
        
        # get training data
        self.img_batch, self.mask_batch = self.get_loaded(loaded)

        # reset gradients
        self.optimizer.zero_grad()

        # process batch through network
        self.out = self.model(self.img_batch.float())

        # get loss value
        loss_val = self.getLoss()

        # track loss for the given epoch
        epoch_loss.append(loss_val.item())
        self.train_loss_per_batch.append(loss_val.item())

        # backpropagation
        loss_val.backward()

        # update the parameters
        self.optimizer.step()

        # cyclic updates across batchs
        if self.sched_name in ['CyclicLR']:
            self.scheduler.step()
    
        return epoch_loss
    
    def valStep(self, loaded, epoch_loss):
        self.img_batch, self.mask_batch = self.get_loaded(loaded)

        # process batch through network
        self.out = self.model(self.img_batch.float())
        preds = self.sm(self.out).cpu().numpy()
        self.pred_out = np.argmax(preds, axis = 1)

        # calculate loss
        loss_val = self.getLoss()

        # track loss
        epoch_loss.append(loss_val.item())
        self.val_loss_per_batch.append(loss_val.item())
        
        return epoch_loss
      
    def run(self):
        for i in range(self.epochs):
            self.init_epoch(i)
        
            # switch between training and testing 
            for phase in ['training', 'testing']:
                self.phase = phase
                self.model_mode()

                if self.phase == 'testing':
                    self.store_metrics()
                    if self.sched_name == 'ReduceLROnPlateau':
                        self.scheduler.step(self.val_losses[-1])
            # if highest F1 score so far then we save the weights
            self.save_weights()
            
            self.current_epoch = int(i+1)
        
        return self.model
    