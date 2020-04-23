import os
import torch
from base.trainer import BaseTrainer
from torch.utils.data import DataLoader
from utils.utils import Loss, EarlyStopping
from tqdm import tqdm


class Yolov3Trainer(BaseTrainer):

    def __init__(self, model, optimizer, criterion, metric, device, anchors, log = None):
        super(Yolov3Trainer,self).__init__(model,optimizer,criterion,metric,device,log)
        self.anchors = anchors

    def iter( self, batch):
        batch = [t.type(torch.FloatTensor) for t in batch]
        batch = [t.to(self.device) for t in batch]
        image, y_true13, y_true26, y_true52 = batch
        image = image.permute(0,3,1,2).type(torch.cuda.FloatTensor)
        output = self.model(image)
        total_loss, xy_loss, wh_loss, conf_loss, prob_loss = self.criterion(output, [y_true13,y_true26, y_true52],self.anchors)
        return [total_loss, xy_loss, wh_loss, conf_loss, prob_loss], [y_true13, y_true26, y_true52], output

    def train(self,
              train_dataset,
              epochs,
              gradient_accumalation_step,
              dev_dataset = None,
              train_batch_size = 16,
              dev_batch_size = 32,
              num_workers = 2,
              not_imporve_step = 3,
              gradient_clipping = 5):

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size= train_batch_size,
                                  shuffle= True,
                                  num_workers= num_workers)

        dev_loader = None

        if dev_dataset is not None:
            dev_loader = DataLoader(dataset=dev_dataset,batch_size = dev_batch_size, shuffle= False, num_workers = num_workers)
            early_stopping = EarlyStopping(not_improve_step=not_imporve_step, verbose= True)

        self.model.to(self.device)

        global_step = 0
        val_global_step = 0
        best_loss = 1000

        for i in range(epochs):

            self.model.train()

            train_epoch_total_loss = Loss()
            train_epoch_xy_loss = Loss()
            train_epoch_wh_loss = Loss()
            train_epoch_conf_loss = Loss()
            train_epoch_class_loss = Loss()

            self.metric.clear_memory()

            for batch in tqdm(train_loader):

                global_step +=1
                step_loss, y_true, y_pred = self.iter(batch)

                # self.metric.write(y_true,y_pred)
                total_loss, xy_loss, wh_loss, conf_loss, class_loss = step_loss
                total_loss = total_loss / gradient_accumalation_step
                total_loss.backward()

                train_epoch_total_loss.write(total_loss.item())
                train_epoch_xy_loss.write(xy_loss.item())
                train_epoch_wh_loss.write(wh_loss.item())
                train_epoch_conf_loss.write(conf_loss.item())
                train_epoch_class_loss.write(class_loss.item())

                self.log.write('total_loss/training_total_loss',total_loss.item(),global_step)
                self.log.write('xy_loss/training_xy_loss',xy_loss.item(),global_step)
                self.log.write('wh_loss/training_wh_loss',wh_loss.item(),global_step)
                self.log.write('conf_loss/training_conf_loss',conf_loss.item(),global_step)
                self.log.write('class_loss/training_class_loss',class_loss.item(),global_step)

                if global_step % gradient_accumalation_step == 0:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), gradient_clipping)
                    self.optimizer.step()
                    self.model.zero_grad()

            train_total_loss_average = train_epoch_total_loss.average()
            train_total_xy_average = train_epoch_xy_loss.average()
            train_total_wh_average = train_epoch_wh_loss.average()
            train_total_conf_average = train_epoch_conf_loss.average()
            train_total_class_average = train_epoch_class_loss.average()

            if dev_loader is not None:

                self.model.eval()

                dev_epoch_total_loss = Loss()
                dev_epoch_xy_loss = Loss()
                dev_epoch_wh_loss = Loss()
                dev_epoch_conf_loss = Loss()
                dev_epoch_class_loss = Loss()

                self.metric.clear_memory()

                with torch.no_grad():

                    for batch in tqdm(dev_loader):
                        step_loss, y_true, y_pred = self.iter(batch)
                        total_loss, xy_loss, wh_loss, conf_loss, class_loss = step_loss

                        # self.metric.write(y_true,y_pred)

                        dev_epoch_total_loss.write(total_loss.item())
                        dev_epoch_xy_loss.write(xy_loss.item())
                        dev_epoch_wh_loss.write(wh_loss.item())
                        dev_epoch_conf_loss.write(conf_loss.item())
                        dev_epoch_class_loss.write(class_loss.item())

                        self.log.write('total_loss/dev_total_loss',total_loss.item(),val_global_step)
                        self.log.write('xy_loss/dev_xy_loss',xy_loss.item(),val_global_step)
                        self.log.write('wh_loss/dev_wh_loss',wh_loss.item(),val_global_step)
                        self.log.write('conf_loss/dev_conf_loss',conf_loss.item(),val_global_step)
                        self.log.write('class_loss/dev_class_loss',class_loss.item(),val_global_step)

                dev_total_loss_average = dev_epoch_total_loss.average()
                dev_xy_loss_average = dev_epoch_xy_loss.average()
                dev_wh_loss_average = dev_epoch_wh_loss.average()
                dev_conf_loss_average = dev_epoch_conf_loss.average()
                dev_class_loss_average = dev_epoch_class_loss.average()

                stop = early_stopping.step(val= dev_total_loss_average)

                if stop:
                    break

                if dev_total_loss_average <= best_loss:
                    best_loss = dev_total_loss_average
                    model_path = 'model_epoch_{0}_best_loss{1:.2f}.pth'.format(i+1,best_loss)
                    self.save_model(model_path)

                # dev_result = self.metric.average()
                # for tag, item in dev_result.items():
                #     self.log.write(tag,item,i+1)

                print('epoch - {0}, global_step:{1}, train_total_loss:{2:.2f}, train_xy_loss:{3:.2f}, train_wh_loss:{4:.2f}, train_conf_loss:{5:.2f}, train_class_loss: {6:.2f}, dev_total_loss:{7:.2f}, dev_xy_loss:{8:.2f}, dev_wh_loss:{9:.2f}, dev_conf_loss:{10:.2f},dev_class_loss:{11:.2f} '.format(i+1, global_step, train_total_loss_average, train_total_xy_average, train_total_wh_average, train_total_conf_average, train_total_class_average, dev_total_loss_average, dev_xy_loss_average, dev_wh_loss_average, dev_conf_loss_average, dev_class_loss_average))

                # print(train_result, dev_result)

            else:

                if train_total_loss_average <= best_loss:
                    best_loss = train_total_loss_average
                    model_path = 'model_epoch_{0}_best_loss{1:.2f}.pth'.format(i+1,best_loss)
                    self.save_model(model_path)

                print('epoch - {0}, global_step:{1}, train_total_loss:{2:.2f}, train_xy_loss:{3:.2f}, train_wh_loss:{4:.2f}, train_conf_loss:{5:.2f}, train_class_loss:'.format(i+1, global_step, train_total_loss_average, train_total_xy_average, train_total_wh_average, train_total_conf_average, train_total_class_average))
