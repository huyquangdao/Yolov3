import matplotlib.pyplot as plt
from utils.data_utils import resize_with_bbox, build_ground_truth, read_anchors
from parser_data.face_parser import FaceParser
from base.dataset import BaseDataset
import os
import cv2


class FaceDataset(BaseDataset):

    def __init__(self,
                 type_name,
                 name_dir,
                 annotation_dir,
                 anchor_dir,
                 image_dir,
                 image_size,
                 letterbox=True,
                 is_train=True):

        self.parser = FaceParser(type_name, annotation_dir, name_dir)
        self.is_train = is_train
        self.image_dir = image_dir
        self.image_size = image_size
        self.is_train = is_train
        self.letterbox = letterbox

        self.anchors = read_anchors(anchor_dir)
        self.dataset = self.parser.parse_dataset()

        print(len(self.dataset))

        self.n_classes = len(self.parser.face_names)

    def __len__(self):
        return len(self.dataset)

    def __read_image(self, image_name):
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        assert len(image.shape) == 3
        image = image[..., ::-1]
        return image

    def __getitem__(self, idx):

        file_name, labels, list_boxes = self.dataset[idx]['file_name'], self.dataset[
            idx]['labels'], self.dataset[idx]['list_all_boxes']

        print(labels)

        image = self.__read_image(file_name)

        image, boxes = resize_with_bbox(img=image,
                                        bbox=list_boxes,
                                        new_width=self.image_size,
                                        new_height=self.image_size,
                                        letterbox=self.letterbox)

        # print(file_name, boxes)

        # cv2.imwrite('image1.jpg',image)

        assert image.shape == (self.image_size, self.image_size, 3)

        # print(boxes)

        assert (boxes < self.image_size + 50).all()

        # print(boxes)

        y_true13, y_true26, y_true52 = build_ground_truth(n_classes=self.n_classes,
                                                          labels=labels,
                                                          boxes=boxes,
                                                          anchors=self.anchors,
                                                          image_size=self.image_size)

        return image, y_true13, y_true26, y_true52
