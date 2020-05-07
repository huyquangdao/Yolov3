from base.parser import BaseParser
import numpy as np
from utils.data_utils import get_voc_names


class FaceParser(BaseParser):

    def __init__(self, type_name, file_dir, name_dir):
        super().__init__(type_name, file_dir, name_dir)
        self.face_names = get_voc_names(self.name_dir)

    def process_one(self, file):

        with open(file, 'r') as f:
            lines = f.readlines()

            list_labels = []
            list_boxes = []
            image_name = lines[0]

            for line in lines[1:]:
                label, x_min, x_max, y_min, y_max = line.strip().split(' ')
                list_labels.append(self.face_names[label])
                list_boxes.append([float(x_min), float(
                    y_min), float(x_max), float(y_max)])

            return image_name, np.array(list_labels), np.array(list_boxes)
