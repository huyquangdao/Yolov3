import torch.nn as nn
import torch.optim as optim
import argparse
import torch
import os
from trainers.yolov3_trainer import Yolov3Trainer
# from models.yolov3 import Yolov3, YoloLossLayer

from models.yolov3_2 import Yolov3, YoloLossLayer

from metrics.map import MeanAveragePrecisionMetric
from utils.utils import set_seed, Summary

from utils.data_utils import read_anchors

from dataset.voc_dataset import VocDataset
from utils.log import Writer


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir', help='Your training directory', default='data/train')
    parser.add_argument(
        '--test_dir', help='Your testing directory', default='data/test')
    parser.add_argument(
        '--anchors_dir', help='Your anchors directory', default='data/voc_anchors.txt')
    parser.add_argument(
        '--name_dir', help='Your name directory', default='data/voc_name.txt')
    parser.add_argument(
        '--image_size', help='Your training image size', default=416, type=int)
    parser.add_argument(
        '--batch_size', help='Your training batch size', default=8, type=int)
    parser.add_argument(
        '--num_workers', help='number of process', default=2, type=int)
    parser.add_argument('--seed', help='random seed', default=1234, type=int)
    parser.add_argument('--epoch', help='training epochs',
                        default=20, type=int)
    parser.add_argument('--learning_rate',
                        help='learning rate', default=5e-4)
    parser.add_argument('--val_batch_size',
                        help='Your validation batch size', default=8)
    parser.add_argument(
        '--grad_clip', help='gradient clipping theshold', default=5, type=int)
    parser.add_argument('--grad_accum_step',
                        help='gradient accumalation step', default=1)
    parser.add_argument('--n_classes', help='Number of classes', default=20)

    parser.add_argument('--gpu', help='use gpu', default=1, type=bool)

    parser.add_argument(
        '--log_dir', help='Log directory path', default='logs', type=str)

    parser.add_argument(
        '--focal_loss', help='use focal loss', default=0, type=bool)
    parser.add_argument(
        '--label_smooth', help='use label smooth', default=0, type=bool)

    parser.add_argument(
        '--letterbox', help='use letterbox resize', default=1, type=bool)

    parser.add_argument(
        '--weight_decay', help='l2 regularization term', default=5e-4, type=float)

    parser.add_argument('--cfg', help='yolo config file',
                        required=True, type=str)

    parser.add_argument(
        '--pretrained', help='yolo pretrained weights', default='', type=str)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    set_seed(args.seed)

    model = Yolov3(cfgfile=args.cfg, n_classes=args.n_classes,
                   image_size=args.image_size)

    if args.pretrained != '':
        model.load_weights(args.pretrained)

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    metric = MeanAveragePrecisionMetric(n_classes=args.n_classes)

    train_dataset = VocDataset(type_name='detection',
                               name_dir=args.name_dir,
                               annotation_dir=os.path.join(
                                   args.train_dir, 'Annotations'),
                               anchor_dir=args.anchors_dir,
                               image_dir=os.path.join(
                                   args.train_dir, 'JPEGImages'),
                               image_size=args.image_size,
                               letterbox=args.letterbox,
                               is_train=True)

    test_dataset = VocDataset(type_name='detection',
                              name_dir=args.name_dir,
                              annotation_dir=os.path.join(
                                  args.test_dir, 'Annotations'),
                              anchor_dir=args.anchors_dir,
                              image_dir=os.path.join(
                                  args.test_dir, 'JPEGImages'),
                              image_size=args.image_size,
                              letterbox=args.letterbox,
                              is_train=False)

    writer = Writer(log_dir=args.log_dir)

    if args.gpu:
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')

    criterion = YoloLossLayer(n_classes=args.n_classes, image_size=args.image_size,
                              device=DEVICE, use_focal_loss=args.focal_loss, use_label_smooth=args.label_smooth)

    anchors = torch.from_numpy(read_anchors(args.anchors_dir)).to(DEVICE)

    summary = Summary(model=model, train_dataset=train_dataset,
                      args=args, dev_dataset=test_dataset, device=DEVICE)
    summary()

    trainer = Yolov3Trainer(model=model,
                            optimizer=optimizer,
                            criterion=criterion,
                            metric=metric,
                            log=writer,
                            device=DEVICE,
                            anchors=anchors
                            )

    trainer.train(train_dataset=train_dataset,
                  dev_dataset=test_dataset,
                  epochs=args.epoch,
                  gradient_accumalation_step=args.grad_accum_step,
                  train_batch_size=args.batch_size,
                  dev_batch_size=args.val_batch_size,
                  num_workers=args.num_workers,
                  gradient_clipping=args.grad_clip)
