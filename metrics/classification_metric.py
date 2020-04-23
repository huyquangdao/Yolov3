from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from base.metric import BaseMetric
import torch


class ClassificationMetric(BaseMetric):

    def __init__(self, n_classes):
        super(ClassificationMetric, self).__init__()
        self.n_classes = n_classes
        self.y_true = []
        self.y_pred = []

    def write(self, y_true_batch, y_pred_batch):

        self.y_true.append(y_true_batch)
        self.y_pred.append(y_pred_batch)

    def clear_memory(self):

        self.y_true = []
        self.y_pred = []

    def average(self):

        self.y_true = torch.cat(self.y_true, dim=0)
        self.y_pred = torch.cat(self.y_pred, dim=0)

        y_true_classes = self.y_true.detach().cpu().data.numpy().tolist()
        y_pred_classes = torch.argmax(
            self.y_pred, dim=-1).detach().cpu().data.numpy().tolist()

        if self.n_classes > 2:
            result = {
                'accuracy': accuracy_score(y_true_classes, y_pred_classes),
                'f1_micro': f1_score(y_true=y_true_classes, y_pred=y_pred_classes, average='micro'),
                'f1_macro': f1_score(y_true=y_true_classes, y_pred=y_pred_classes, average='macro')
            }
        else:
            result = {
                'accuracy': accuracy_score(y_true_classes, y_pred_classes),
                'f1': f1_score(y_true=y_true_classes, y_pred=y_pred_classes, average='binary'),
            }
        return result
