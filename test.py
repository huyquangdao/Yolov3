from utils.data_utils import read_anchors, letterbox_resize
import numpy as np
import cv2
import torch
import argparse
from models.yolov3 import Yolov3
from inference.yolo_inference import YoloInference
from torch.autograd import Variable
import matplotlib.pyplot as plt

from utils.utils import Timer

def parse_arg():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='image_path', type=str,
                        default='data/dog-cycle-car.png')
    parser.add_argument('--anchors_dir', help='image_path',
                        type=str, default='data/face_anchors.txt')
    parser.add_argument('--image_size', help='image size',
                        type=int, default=416)
    parser.add_argument(
        '--n_classes', help='number of classes', type=int, default=1)
    parser.add_argument('--model', help='model_dir', type=str,
                        default='weights/yolov3.weights')
    parser.add_argument('--gpu', type=bool, default=0)
    parser.add_argument('--letterbox', type=bool, default=1)

    parser.add_argument('--cfg', help='yolo config file',
                        type=str, default='cfg/yolov3-voc.cfg')

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_arg()
    image_ori = cv2.imread(args.image)

    if args.letterbox:
        image, resize_ratio, dw, dh = letterbox_resize(
            image_ori, args.image_size, args.image_size)
    else:
        height_ori, width_ori = image_ori.shape[:2]
        image = cv2.resize(image_ori, (args.image_size, args.image_size))

    image = image[..., ::-1]

    image = image / 255.

    image = image.astype(np.float32)

    model = Yolov3(n_classes=args.n_classes)


    timer = Timer()

    # model.eval()

    if args.gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model.load_state_dict(torch.load(args.model, map_location= device))

    anchors = torch.from_numpy(read_anchors(args.anchors_dir))

    image = Variable(torch.from_numpy(image))

    inference = YoloInference(
        model=model, device=device, n_classes=args.n_classes, anchors=anchors)

    boxes_, scores_, labels_ = timer(inference.inference)(
        image, iou_threshold=0.45, score_threshold=0.4, top_k=50)

    if args.letterbox:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori/float(args.image_size))
        boxes_[:, [1, 3]] *= (height_ori/float(args.image_size))

    image = cv2.imread(args.image)

    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)

    for box in boxes_[:3]:
        box = [int(t) for t in box]
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # break

    plt.imshow(image)
    plt.show()

    cv2.imwrite('result.jpg',image)