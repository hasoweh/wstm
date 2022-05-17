from .dsrg_layer import DSRGLayer
import torch.nn as nn
import torch

class DSRG_Loss(nn.Module):
    
    def __init__(self, ignore_label = 255, threshold = 0.85, device = "cpu"):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index = ignore_label, 
                                                   size_average=True
                                                  ).to(device)
        self.DSRGLayer = DSRGLayer(threshold, device)
        self.device = device
        

    def get_constrain_loss(self, softmax, crf):
        """Constrain loss function
        
        Parameters
        ----------
        softmax : Tensor
            Final feature map prediction
        crf : Tensor
            Output of dense CRF
        
        Returns
        -------
        loss : Tensor
            Output of constrain loss function
        """
        crf_smooth = torch.exp(crf)
        #loss = torch.mean(torch.sum(crf_smooth * torch.log(crf_smooth/(softmax+1e-8)+1e-8),
        #                            axis=3))
        loss = crf_smooth * torch.log(crf_smooth/softmax+1e-8)
        loss = torch.mean(torch.sum(loss, axis=1))
        return loss
    
    def forward(self, softmax, logits, cams, img, img_labels):
        pseudolabels, crf = self.DSRGLayer([img, img_labels, cams, softmax])
        
        seed_loss = self.criterion(logits, pseudolabels.long().to(self.device))
        constrain_loss = self.get_constrain_loss(softmax, crf)
        return seed_loss + constrain_loss