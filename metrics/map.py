from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from base.metric import BaseMetric
import torch

class MeanAveragePrecisionMetric(BaseMetric):

    def __init__(self, n_classes):
        super(MeanAveragePrecisionMetric,self).__init__()
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
        pass
