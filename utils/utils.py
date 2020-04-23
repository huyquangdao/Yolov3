from base.meters import BaseMeters
import time
import random
import torch
import numpy as np


class Loss(BaseMeters):

    def __init__(self):
        super(Loss, self).__init__()


class Timer:

    def __init__(self):
        pass

    def __call__(self, function):

        def wrapper(*args, **kwargs):
            start = time.time()
            result = function(*args, **kwargs)
            end = time.time()
            print('function:%r took: %2.2f sec' %
                  (function.__name__,  end - start))
            return result

        return wrapper


class EarlyStopping:

    def __init__(self, not_improve_step,  verbose=True):

        self.not_improve_step = not_improve_step
        self.verbose = verbose
        self.best_val = 10000
        self.count = 0

    def step(self, val):
        if val <= self.best_val:
            self.best_val = val
            self.count = 0
            return False
        else:
            self.count += 1
            if self.count > self.not_improve_step:
                if self.verbose:
                    print('Performance not Improve after {0}, Early Stopping Execute .......'.format(
                        self.count))
                return True
            return False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
