from inference.yolo_inference import YoloInference
from models.yolov3 import Yolov3
import argparse
import torch
import cv2
import numpy as np
from utils.data_utils import read_anchors


def parse_arg():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='image_path', type=str)
    parser.add_argument('--anchors_dir', help='image_path',
                        type=str, default='data/voc_anchors.txt')
    parser.add_argument('--image_size', help='image size',
                        type=int, default=416)
    parser.add_argument(
        '--n_classes', help='number of classes', type=int, default=20)
    parser.add_argument('--model', help='model_dir', type=str)
    parser.add_argument('--gpu', type=bool, default=1)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_arg()
    image = cv2.imread(args.image)

    image = image / 255.

    image = image.astype(np.float32)

    model = Yolov3(args.n_classes)
    model.load_state_dict(torch.load(args.model))

    if args.gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    anchors = torch.from_numpy(read_anchors(args.anchors_dir))

    inference = YoloInference(
        model=model, device=device, n_classes=args.n_classses, anchors=anchors)

    boxes_, scores_, labels_ = inference.inference(image)

    print(boxes_)

    print(scores_)

    print(labels_)
