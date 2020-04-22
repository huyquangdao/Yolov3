
from torch.utils.tensorboard import SummaryWriter

class Writer:

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(self.log_dir)

    def write(self,name,number,global_step):
        self.writer.add_scalar(tag=name,scalar_value=number,global_step=global_step)

