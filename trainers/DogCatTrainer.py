import os

from base.trainer import BaseTrainer


class DogCatTrainer(BaseTrainer):

    def __init__(self, model, optimizer, criterion,  metric, device, log = None):
        super(DogCatTrainer,self).__init__(model,optimizer,criterion,metric,device,log)

    def iter( self, batch):
        batch = [t.to(self.device) for t in batch]
        image, label = batch
        logits = self.model(image)
        loss = self.criterion(logits,label)
        return loss, label, logits
