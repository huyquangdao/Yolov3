from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from utils.loss_meter import Loss

class BaseTrainer:

    def __init__(self, model, optimizer, criterion,  metric, device, log = None):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
        self.log = log
        self.device = device

    def iter(self, batch):
        raise NotImplementedError('You must implement this method')

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        print('perfomance grain, save mode:')


    def train(self,
              train_dataset,
              epochs,
              gradient_accumalation_step,
              dev_dataset = None,
              train_batch_size = 16,
              dev_batch_size = 32,
              num_workers = 2,
              gradient_clipping = 5):

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size= train_batch_size,
                                  shuffle= True,
                                  num_workers= num_workers)

        dev_loader = None

        if dev_dataset is not None:
            dev_loader = DataLoader(dataset=dev_dataset,batch_size = dev_batch_size, shuffle= False, num_workers = num_workers)

        self.model.to(self.device)

        global_step = 0
        best_loss = 1000

        for i in range(epochs):

            self.model.train()
            train_epoch_loss = Loss()

            self.metric.clear_memory()

            for batch in tqdm(train_loader):

                global_step +=1
                step_loss, y_true, y_pred = self.iter(batch)
                self.metric.write(y_true,y_pred)
                step_loss = step_loss / gradient_accumalation_step
                step_loss.backward()

                train_epoch_loss.write(step_loss.item())
                self.log.write('training_loss',step_loss.item(),global_step)

                if global_step % gradient_accumalation_step == 0:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), gradient_clipping)
                    self.optimizer.step()
                    self.model.zero_grad()

            train_loss = train_epoch_loss.average()

            train_result = self.metric.average()

            for tag, item in train_result.items():
                self.log.write(tag,item,i+1)

            if dev_loader is not None:

                self.model.eval()
                dev_epoch_loss = Loss()

                self.metric.clear_memory()

                with torch.no_grad():

                    for batch in tqdm(dev_loader):
                        step_loss, y_true, y_pred = self.iter(batch)
                        self.metric.write(y_true,y_pred)
                        dev_epoch_loss.write(step_loss.item())
                        self.log.write('validation_loss',step_loss.item(),global_step)

                dev_loss = dev_epoch_loss.average()

                if dev_loss <= best_loss:
                    best_loss = dev_loss
                    model_path = 'model_epoch_{0}_best_loss{1:.2f}.pth'.format(i+1,best_loss)
                    self.save_model(model_path)

                dev_result = self.metric.average()
                for tag, item in dev_result.items():
                    self.log.write(tag,item,i+1)

                print('epoch - {0}, global_step:{1}, train_loss:{2:.2f}, dev_loss:{3:.2f}'.format(i+1, global_step, train_loss, dev_loss))

                print(train_result, dev_result)

            else:

                if train_loss <= best_loss:
                    best_loss = train_loss
                    model_path = 'model_epoch_{0}_best_loss{1:.2f}.pth'.format(i+1,best_loss)
                    self.save_model(model_path)

                print('epoch - {0},global_step:{1}, train_loss:{2:.2f}'.format(i+1, global_step, train_loss))

                print(train_result)
