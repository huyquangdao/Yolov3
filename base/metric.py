
class BaseMetric:

    def __init__(self):
        pass

    def write_batch(self, y_true_batch, y_pred_batch):
        pass

    def clear_memory(self):
        pass

    def average(self):
        pass
