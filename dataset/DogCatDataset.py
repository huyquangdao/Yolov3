import os
import glob
from torchvision.transforms import Compose, ToPILImage, ToTensor, RandomCrop, Normalize, Resize, RandomHorizontalFlip
import cv2

from base.dataset import BaseDataset

label2class = {
    'dog':0,
    'cat':1
}

class2label = {
    0:'dog',
    1:'cat'
}

class CatDogDataset(BaseDataset):

    def __init__(self, file_path, transform = None, is_train = True):

        super(CatDogDataset,self).__init__()

        self.is_train = is_train

        if self.is_train:
            self.file_path = file_path
            self.transform = transform

            self.images_path, self.labels = read_raw_dataset(self.file_path)
        else:
            self.file_path = file_path
            self.transform = transform
            self.data = read_test_dataset(self.file_path)
            self.data.sort(key = lambda x: x['id'])

    def __len__(self):
        if self.is_train:
            return len(self.labels)
        else:
            return len(self.data)

    def __getitem__(self, idx):

        if self.is_train:
            image_path, label = self.images_path[idx], self.labels[idx]
            image = read_image(image_path)
            if self.transform:
                image = self.transform(image)
            return image, label
        else:
            d = self.data[idx]
            id = d['id']
            path = d['image']
            image = read_image(path)
            if self.transform :
                image = self.transform(image)
            return id, image

def read_raw_dataset(file_dir):

    image_names = os.listdir(file_dir)
    image_paths = []
    labels = []
    for name in image_names:
        label = name.split('.')[0]
        label_idx = label2class[label]
        labels.append(label_idx)
        path = os.path.join(file_dir, name)
        image_paths.append(path)

    assert len(labels) == len(image_paths)

    return image_paths, labels

def read_test_dataset(file_dir):

    image_names = os.listdir(file_dir)
    data = []
    for name in image_names:
        idx = int(name.split('.')[0])
        path = os.path.join(file_dir,name)
        data.append({'id':idx,'image':path})

    return data

def read_image(image_path):

    image = cv2.imread(image_path)
    image = image[...,::-1]
    return image


