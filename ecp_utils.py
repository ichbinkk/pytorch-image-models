"""
EC Models
=============================

**Author:** `ichbinkk`

"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from timm.models import create_model
import my_models.mpvit

from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image
import timm.models as tm
import pandas as pd





# set train and val data prefixs
prefixs = None
# prefixs = ['A1','A2','B1','B2','C1','C2','D1','D2','E1','E2']
# prefixs = ['A1','A2','B1','B2','C1','C2', 'D2']
# prefixs = ['A1','B1','C2', 'D2', 'E1']
# prefixs = ['']


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes=1, feature_extract=False, use_pretrained=False, drop=0, drop_path=0.1, drop_block=None):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet34":
        """ Resnet
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        """ Resnet
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        """ Resnet
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    # if model_name == "resnet":
    #     """ Resnet34
    #     """
    #     model_ft = tm.resnet34(pretrained=use_pretrained)
    #     set_parameter_requires_grad(model_ft, feature_extract)
    #     num_ftrs = model_ft.fc.in_features
    #     model_ft.fc = nn.Linear(num_ftrs, num_classes)
    #     input_size = 224

    elif model_name == "regnety_040":
        """ regnet
            regnety_040， regnety_080， regnety_160
        """
        model_ft = tm.regnety_040(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.get_classifier().in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "regnety_080":
        """ regnet
            regnety_040， regnety_080， regnety_160
        """
        model_ft = tm.regnety_040(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.get_classifier().in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "regnety_160":
        """ regnet
            regnety_040， regnety_080， regnety_160
        """
        model_ft = tm.regnety_040(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.get_classifier().in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "efficientnet_b2":
        """ 
            efficientnet_b2 256, efficientnet_b3 288, efficientnet_b4 320
        """
        model_ft = tm.efficientnet_b2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.get_classifier().in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 256

    elif model_name == "efficientnet_b3":
        """ 
            efficientnet_b2 256, efficientnet_b3 288, efficientnet_b4 320
        """
        model_ft = tm.efficientnet_b3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.get_classifier().in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 288

    elif model_name == "efficientnet_b4":
        """ 
            efficientnet_b2 256, efficientnet_b3 288, efficientnet_b4 320
        """
        model_ft = tm.efficientnet_b4(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.get_classifier().in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 320

    elif model_name == "vit_t":
        model_ft = tm.vit_tiny_patch16_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vit_s":
        model_ft = tm.vit_small_patch32_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "pit_xs":
        """ pit
        pit_xs_224
        """
        model_ft = tm.pit_xs_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.get_classifier().in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "pit_s":
        """ pit
        pit_s_224
        """
        model_ft = tm.pit_s_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.get_classifier().in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "deit_s":
        """ deit
            deit_small_patch16_224
        """
        model_ft = tm.deit_small_patch16_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "deit_b":
        """ deit
            deit_base_patch16_224
        """
        model_ft = tm.deit_small_patch16_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "mixer":
        """ mixer
        """
        model_ft = tm.mixer_b16_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "swin_vit_t":
        """ swin-vit
        tm.swin_tiny_patch4_window7_224
        """
        model_ft = tm.swin_tiny_patch4_window7_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "swin_vit_s":
        """ swin-vit
        tm.swin_small_patch4_window7_224
        """
        model_ft = tm.swin_small_patch4_window7_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "mpvit_tiny":
        """
        mpvit
        """
        model_ft = create_model(
            "mpvit_tiny",
            pretrained=use_pretrained,
            num_classes=num_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
            drop_block_rate=drop_block,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        # for param in model_ft.trans_cls_head.parameters():
        #     param.requires_grad = True  # it was require_grad
        input_size = 224

    elif model_name == "mpvit_xsmall":
        """
        mpvit
        """
        model_ft = create_model(
            "mpvit_xsmall",
            pretrained=use_pretrained,
            num_classes=num_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
            drop_block_rate=drop_block,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        # for param in model_ft.trans_cls_head.parameters():
        #     param.requires_grad = True  # it was require_grad
        input_size = 224

    elif model_name == "mpvit_small":
        """
        mpvit
        """
        model_ft = create_model(
            "mpvit_small",
            pretrained=use_pretrained,
            num_classes=num_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
            drop_block_rate=drop_block,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        # for param in model_ft.trans_cls_head.parameters():
        #     param.requires_grad = True  # it was require_grad
        input_size = 224

    elif model_name == "mpvit_base":
        """
        mpvit
        """
        model_ft = create_model(
            "mpvit_base",
            pretrained=use_pretrained,
            num_classes=num_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
            drop_block_rate=drop_block,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        # for param in model_ft.trans_cls_head.parameters():
        #     param.requires_grad = True  # it was require_grad
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg11":
        """ VGG11
        """
        model_ft = models.vgg11(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg19":
        """ VGG19
        """
        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet121":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# use PIL Image to read image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


# define your Dataset. Assume each line in your .txt file is [name '\t' label], for example:0001.jpg 1
class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = []
            self.img_label = []
            # self.img_name = [os.path.join(img_path, line.strip().split('\t')[0]) for line in lines]
            # self.img_label = [float(line.strip().split('\t')[1]) for line in lines]
            for line in lines:
                img_name = line.strip().split('\t')[0]
                img_prefix = img_name.split('-')[0]
                if prefixs is not None:
                    if img_prefix in prefixs:   # 指定训练数据位置
                        self.img_name.append(os.path.join(img_path,img_name))
                        self.img_label.append(float(line.strip().split('\t')[1]))
                else:
                    self.img_name.append(os.path.join(img_path, img_name))
                    self.img_label.append(float(line.strip().split('\t')[1]))
        # 对y做归一化
        ln = len(self.img_label)
        y = self.img_label
        y = torch.tensor(y)
        y = y.numpy()
        print(np.shape(y))
        meanVal = np.mean(y)
        stdVal = np.std(y)
        y = (y - meanVal) / stdVal
        self.img_label = y
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label


### 读取txt文本的第k列数值data ###
def loadCol(infile, k):
    f = open(infile, 'r')
    sourceInLine = f.readlines()
    dataset = []
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split('\t')
        dataset.append(temp2[k])
    # for i in range(0, len(dataset)):
    #     for j in range(k):
    #         #dataset[i].append(float(dataset[i][j]))
    #         dataset[i][j] = float(dataset[i][j])
    if dataset[0].split('.')[-1] != 'png':
        dataset = [float(s) for s in dataset]
    return dataset


'''读取文本的第k列字符串data'''
def loadColStr(infile, k):
    f = open(infile, 'r')
    sourceInLine = f.readlines()
    dataset = []
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split('\t')
        if prefixs is not None:
            if temp2[0].split('-')[0] in prefixs:
                dataset.append(float(temp2[k]))
        else:
            dataset.append(float(temp2[k]))
    # for i in range(0, len(dataset)):
    #     for j in range(k):
    #         #dataset[i].append(float(dataset[i][j]))
    #         dataset[i][j] = float(dataset[i][j])
    return dataset


def Normalize(data):
    # res = []
    data = np.array(data)
    meanVal = np.mean(data)
    stdVal = np.std(data)
    res = (data - meanVal) / stdVal
    return res, meanVal, stdVal


def InvNormalize(data, meanVal, stdVal):
    # data为tensor时
    # data = data.cpu().numpy()

    # data为list时
    data = np.array(data)
    return (data*stdVal)+meanVal


def seg_index(infile):
    f = open(infile, 'r')
    sourceInLine = f.readlines()
    index = []
    i = 0
    for line in sourceInLine:
        temp1 = line.strip('\n')
        filepath = temp1.split('\t')[0]
        filename = filepath.split('.')[0]
        layer_id = filename.split('-')[-1]
        if layer_id == '0':
            index.append(i)
        i += 1
    index.append(i)
    return index


'''write to excel'''
def save_excel(data1, data2, data3, data4, file):
    writer = pd.ExcelWriter(file)  # 写入Excel文件
    data1 = pd.DataFrame(data1)
    data2 = pd.DataFrame(data2)
    data3 = pd.DataFrame(data3)
    data4 = pd.DataFrame(data4)
    # data.to_excel(writer, sheet_name='Sheet1', float_format='%.2f', header=False, index=False)

    data1.to_excel(writer, sheet_name='res', header=False, index=False)
    data2.to_excel(writer, sheet_name='metrics_hist', header=False, index=False)
    data3.to_excel(writer, sheet_name='predicted_ec', header=False, index=False)
    data4.to_excel(writer, sheet_name='loss_hist', header=False, index=False)

    # writer.save()
    writer.close()


'''draw txt data'''
def draw_txt(file,k):
    data = np.loadtxt(file, usecols=(k))
    plt.plot(data)
    plt.show()


if __name__ == "__main__":
    draw_txt('F1.txt', 1)