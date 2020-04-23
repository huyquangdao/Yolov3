import glob
import os

class BaseParser:

    def __init__(self, type_name, file_dir, name_dir):
        self.type_name = type_name
        self.file_dir = file_dir
        self.name_dir = name_dir

    def process_one(self, file):
        raise NotImplementedError('You must implement this function')

    def parse_dataset(self):

        list_file_path = glob.glob(os.path.join(self.file_dir,'*'))
        dataset = []
        for file_path in list_file_path:
            file_name, labels,  list_all_boxes = self.process_one(file_path)
            assert len(labels) == len(list_all_boxes)
            data = {'file_name':file_name, 'labels':labels  ,'list_all_boxes':list_all_boxes}
            dataset.append(data)
        return dataset

from torch.utils.data import Dataset, DataLoader
