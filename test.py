from inference.DogCatInference import DogCatInference
from models.DogCatModel import Resnet34
import argparse
import torch
import cv2
from torchvision.transforms import Compose, ToPILImage, ToTensor, RandomCrop, Normalize, Resize, RandomHorizontalFlip


def parse_arg():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image',help='image_path',type=str)
    parser.add_argument('--image_size',help='image size',type=int, default=224)
    parser.add_argument('--n_classes',help='number of classes',type=int, default=2)
    parser.add_argument('--model',help='model_dir',type=str)
    parser.add_argument('--gpu',type=bool,default=1)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_arg()
    image = cv2.imread(args.image)
    model = Resnet34(args.n_classes,pretrained=False)
    model.load_state_dict(torch.load(args.model))

    test_transform = Compose([
        ToPILImage(),
        Resize(size=[args.image_size, args.image_size]),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    if args.gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    inference = DogCatInference(model=model,device=device,transform=test_transform)
    result = inference.inference(image)

    print(result)




