import argparse
import cv2
import numpy as np
import torch
from glob import glob

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from collections import OrderedDict
from ECmodels import *


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset / Model parameters
    parser.add_argument('--output_dir', metavar='DIR', default='./cam/V3',
                        help='path to output')
    ''' efficientnet_b4 resnet swin_vit_t  pit_xs vit_t '''
    parser.add_argument('--model', default='efficientnet_b4', type=str, metavar='MODEL',
                        help='Name of model to train (default: "resnet18"')
    parser.add_argument('--pth_dir', metavar='DIR', default='./output/V3_ec',
                        help='path to pth file')

    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./input_images/V3',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


# for ViT-t
def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


# for PiT
def pit_reshape_transform(tensor, height=7, width=7):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


# for Swin-T
def swin_reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def image2cam(image_path, cam, outdir):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (image_size, image_size))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    img_name = image_path.split('/')[-1].split('.')[0]
    cv2.imwrite(os.path.join(out_dir, f'{img_name}_{args.method}.jpg'), cam_image)

if __name__ == '__main__':
    """ python vit_gradcam.py -image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    # model = torch.hub.load('facebookresearch/deit:main',
    #                        'deit_tiny_patch16_224', pretrained=True)

    model, image_size = initialize_model(args.model)
    pth_file = os.path.join(args.pth_dir, args.model, 'Best_checkpoint.pth')
    state_dict = torch.load(pth_file)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    # load params
    model.load_state_dict(new_state_dict)  # 重新加载这个模型。
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    # print(model)


    # regist target_layers
    if args.model in ['resnet50', 'resnet152']:
        target_layers = [model.layer4[-1]]
    elif args.model in ['efficientnet_b3', 'efficientnet_b4']:
        target_layers = [model.blocks[-2][0]]
    elif args.model in ['vit_t']:
        target_layers = [model.blocks[-1].norm1]
    elif args.model in ['swin_vit_t']:
        target_layers = [model.layers[-1].blocks[-1].norm1]
    elif args.model in ['pit_xs']:
        target_layers = [model.transformers[-1].blocks[-1].norm1]
    elif args.model in ['ecpnet']:
        target_layers = [model.conv_trans_12.trans_block.norm2]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.model in ['vit_t', 'ecpnet']:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=vit_reshape_transform)
    elif args.model in ['pit_xs']:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=pit_reshape_transform)
    elif args.model in ['swin_vit_t']:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=swin_reshape_transform)
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda)

    out_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)

    image_names = glob(os.path.join(args.image_path, '*.png'))
    for img in image_names:
        image2cam(img, cam, out_dir)