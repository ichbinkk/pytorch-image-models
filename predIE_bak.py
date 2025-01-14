"""
Finetuning Torchvision Models
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

from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image
import timm.models as tm
import argparse
# 衡量误差
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square




# Number of classes in the dataset
num_classes = 1

# Batch size for training (change depending on how much memory you have)
# batch_size = 64

# Number of epochs to train for
# num_epochs = 100

lr = 0.001

parm = {}  # 初始化保存模块参数的parm字典

# Set argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('--data_dir', metavar='DIR', default='../dataset/lattice_ecc',
                    help='path to dataset')
'''
Setting model and training params, some can use parser to get value.
Models to choose from [resnet, regnet, efficientnet, vit, pit, mixer, deit, swin-vit
alexnet, vgg, squeezenet, densenet, inception]
'''
parser.add_argument('--model', default='resnet', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet"')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-ep', '--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: )')
parser.add_argument('-ft', '--use-pretrained', type=bool, default=False, metavar='N',
                    help='Flag to use fine tuneing(default: False)')
parser.add_argument('-fe', '--feature-extract', type=bool, default=False, metavar='N',
                    help='False to finetune the whole model. True to update the reshaped layer params(default: False)')

# set train and val data prefixs
# prefixs = ['A1','A2','B1','B2','C1','C2','D1','D2','E1','E2']
prefixs = ['A1','A2','B1','B2','C1','C2', 'D2']
# prefixs = ['A1','B1','C2', 'D2', 'E1']

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    train_acc_history = []
    val_acc_history = []
    result = []

    kernal = np.empty((0, 3))

    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = 999999

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                # labels = labels.to(device)
                # labels = torch.tensor(labels, dtype=torch.float) #复制tensor
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        outputs = outputs.view(-1)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        outputs = outputs.view(-1)
                        loss = criterion(outputs, labels)
                        if phase == 'val' and epoch == num_epochs-1:
                            # result.append(outputs)
                            temp = outputs.detach().cpu().numpy()
                            for i in range(outputs.size()[0]):
                                result.append(temp[i])

                    # _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'train':
                train_acc_history.append(epoch_loss)
            if phase == 'val':
                val_acc_history.append(epoch_loss)
                if epoch_loss < min_loss:
                    min_loss = epoch_loss  # update min_loss
                    print('Best val min_loss: {:4f} in Epoch {}/{}'.format(min_loss, epoch, num_epochs - 1))
                    best_model_wts = copy.deepcopy(model.state_dict())

            # 删掉acc 因为回归问题不存在acc
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            #
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            #
            # # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
            #     val_acc_history.append(epoch_acc)

        print()
        # 制定训练序号的同时 保存模块参数（卷积核权重）
    #     if epoch % 19 == 0:
    #         for name, parameters in model.named_parameters():
    #             # print(name, ':', parameters.size())
    #             parm[name] = parameters.detach().cpu().numpy()
    #         print(parm['layer1.0.conv1.weight'][0, 0, :, :])
    #         # kernal = (kernal, parm['layer1.0.conv1.weight'][0, 0, :, :])
    #         kernal = np.append(kernal, parm['layer1.0.conv1.weight'][0, 0, :, :], axis=0)
    #
    # np.savetxt("Parm.txt", kernal)

    time_elapsed = time.time() - since
    print('Training complete in time of {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # save best model weights
    save_path = './output/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.pth'
    torch.save(best_model_wts, save_path)
    return model, train_acc_history, val_acc_history, result


######################################################################
# Set Model Parameters’ .requires_grad attribute
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This helper function sets the ``.requires_grad`` attribute of the
# parameters in the model to False when we are feature extracting. By
# default, when we load a pretrained model all of the parameters have
# ``.requires_grad=True``, which is fine if we are training from scratch
# or finetuning. However, if we are feature extracting and only want to
# compute gradients for the newly initialized layer then we want all of
# the other parameters to not require gradients. This will make more sense
# later.
#

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
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

    elif model_name == "regnet":
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

    elif model_name == "vit":
        """ vit
        """
        model_ft = tm.vit_tiny_patch16_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "pit":
        """ pit
        pit_xs_224, pit_s_224
        """
        model_ft = tm.pit_xs_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.get_classifier().in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "deit":
        """ deit
            deit_small_patch16_224, deit_base_patch16_224
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

    elif model_name == "swin-vit":
        """ swin-vit
        tm.swin_tiny_patch4_window7_224, tm.swin_small_patch4_window7_224
        """
        model_ft = tm.swin_small_patch4_window7_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
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

    elif model_name == "densenet":
        """ Densenet
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
                # if img_prefix is not None:
                if img_prefix in prefixs:   # 指定训练数据位置
                    self.img_name.append(os.path.join(img_path,img_name))
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


### 读取文本的第k列字符串data ###
def loadColStr(infile, k):
    f = open(infile, 'r')
    sourceInLine = f.readlines()
    dataset = []
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split('\t')
        if temp2[0].split('-')[0] in prefixs:
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

if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    # get all args params
    args = parser.parse_args()
    infile = args.data_dir
    model_name = args.model
    batch_size = args.batch_size
    num_epochs = args.epochs
    use_pretrained = args.use_pretrained
    feature_extract = args.feature_extract

    # get data Dir name
    fn = infile.split('/')[-1]

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained)

    # output path
    out_path = './output'
    if not os.path.exists(out_path):
        # 如果不存在则创建目录
        os.makedirs(out_path)
    # Print the model we just instantiated
    # print(model_ft)

    ### Print computational params ###
    # from torchstat import stat
    # stat(model_ft, (3, 224, 224))
    #
    # from thop import profile
    # input = torch.randn(1, 3, 224, 224)
    # macs, params = profile(model_ft, inputs=(input,))
    # print('The model Params: {:.2f}M, MACs: {:.2f}M'.format(params/10e6, macs/10e6))

    ######################################################################
    # Load Data
    # ---------
    #
    # Now that we know what the input size must be, we can initialize the data
    # transforms, image datasets, and the dataloaders. Notice, the models were
    # pretrained with the hard-coded normalization values, as described
    # `here <https://pytorch.org/docs/master/torchvision/models.html>`__.
    #

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize(input_size),
            # transforms.CenterCrop(input_size),
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")
    # load dataset

    image_datasets = {x: customData(img_path=infile,
                                    txt_path=os.path.join(infile, (x + '.txt')),
                                    data_transforms=data_transforms,
                                    dataset=x) for x in ['train', 'val']}

    # wrap your data and label into Tensor
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=batch_size,
                                                 shuffle = True if x == 'train' else False,
                                                 # num_workers=1
                                                       ) for x in ['train', 'val']}  # 这里的shuffle可以打乱数据顺序！！！

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    ######################################################################
    # Create the Optimizer
    # --------------------
    #
    # Now that the model structure is correct, the final step for finetuning
    # and feature extracting is to create an optimizer that only updates the
    # desired parameters. Recall that after loading the pretrained model, but
    # before reshaping, if ``feature_extract=True`` we manually set all of the
    # parameter’s ``.requires_grad`` attributes to False. Then the
    # reinitialized layer’s parameters have ``.requires_grad=True`` by
    # default. So now we know that *all parameters that have
    # .requires_grad=True should be optimized.* Next, we make a list of such
    # parameters and input this list to the SGD algorithm constructor.
    #
    # To verify this, check out the printed parameters to learn. When
    # finetuning, this list should be long and include all of the model
    # parameters. However, when feature extracting this list should be short
    # and only include the weights and biases of the reshaped layers.
    #

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized

    # optimizer_ft = optim.SGD(params_to_update, lr, momentum=0.9)
    optimizer_ft = optim.Adam(params_to_update, lr, weight_decay=0.05)

    ######################################################################
    # Run Training and Validation Step
    # --------------------------------
    #
    # Finally, the last step is to setup the loss for the model, then run the
    # training and validation function for the set number of epochs. Notice,
    # depending on the number of epochs this step may take a while on a CPU.
    # Also, the default learning rate is not optimal for all of the models, so
    # to achieve maximum accuracy it would be necessary to tune for each model
    # separately.
    #

    # Setup the loss fxn
    criterion = nn.MSELoss()

    ### 使用迁移学习 ###
    # Train and evaluate
    model_ft, train_hist, val_hist, result = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


    ######################################################################
    ### 比较 迁移学习和普通学习 的训练结果（可选） ###
    # Comparison with Model Trained from Scratch
    # ------------------------------------------
    #
    # Just for fun, lets see how the model learns if we do not use transfer
    # learning. The performance of finetuning vs. feature extracting depends
    # largely on the dataset but in general both transfer learning methods
    # produce favorable results in terms of training time and overall accuracy
    # versus a model trained from scratch.
    #

    ### 不使用迁移学习 ###
    # Initialize the non-pretrained version of the model used for this run
    # scratch_model, _ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
    # scratch_model = scratch_model.to(device)
    # scratch_optimizer = optim.SGD(scratch_model.parameters(), lr, momentum=0.9)
    # # scratch_optimizer = optim.Adam(scratch_model.parameters(), lr=0.01, weight_decay=0.01)
    # scratch_criterion = nn.MSELoss()
    # model, train_hist, val_hist, result = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name == "inception"))

    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the transfer learning method and
    #  the model trained from scratch
    # ohist = []
    # shist = []

    # ohist = [h.cpu().numpy() for h in hist]
    # shist = [h.cpu().numpy() for h in scratch_hist]

    # ohist = hist
    # shist = scratch_hist

    # 绘制训练和验证的损失曲线
    plt.figure()
    plt.title("Train and val Loss history vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), train_hist, label="train_hist")
    plt.plot(range(1,num_epochs+1), val_hist, label="val_hist")
    # plt.ylim((0, 2.))
    # plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.savefig(os.path.join('./output/', ('Hist of ' + fn + "_" + str(model_name) + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size) + '.png')))
    plt.show()
    hist = np.vstack((train_hist, val_hist))
    np.savetxt("./output/Hist of " + fn + "_" + str(model_name) + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size), hist.T)
    # np.savetxt("./output/train_hist of " + fn + "_" + str(model_name) + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size), train_hist)
    # np.savetxt("./output/val_hist of " + fn + "_" + str(model_name) + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size), val_hist)

    #######################################################################
    # -----------------------Evaluation--------------------------------
    ### 逆归一化，输出 val 的‘预测的结果’ ###
    test_lab = loadColStr(os.path.join(infile, 'val.txt'), 1)
    _, meanVal, stdVal = Normalize(test_lab)
    result = InvNormalize(result, meanVal, stdVal)

    ### 逆归一化，输出 test 的‘预测的结果’ ###
    # test_img_path = loadCol(os.path.join(infile, 'test.txt'), 0)
    # test_lab = loadCol(os.path.join(infile, 'test.txt'), 1)
    # _, meanVal, stdVal = Normalize(test_lab)
    #
    # model_ft.eval()
    # torch.no_grad()
    # result = []
    # transform = transforms.Compose([
    #     transforms.Resize(input_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    # for i in range(len(test_img_path)):
    #     test_img = Image.open(os.path.join(infile, test_img_path[i])).convert('RGB')
    #     test_img = transform(test_img).unsqueeze(0)
    #     test_img = test_img.to(device)
    #     out = model_ft(test_img)
    #     out = out.detach().cpu().numpy()
    #     out_v = out[0][0] * stdVal + meanVal
    #     # print(out_v)
    #     result.append(out_v)

    # 绘制期望和预测的结果
    plt.figure()
    plt.title(model_name + "_" +str(num_epochs) + "_" + str(lr) + "_" + str(batch_size) + "Validation Result")
    ts = range(len(test_lab))
    plt.plot(ts, test_lab, label="test_lab")
    plt.plot(ts, result, label="pred_lab")
    # plt.xlim((0, 200))
    # plt.ylim((0, 450000))
    plt.legend()
    plt.show()
    # 输出预测的结果到txt文件
    res = np.vstack((test_lab, result))
    np.savetxt('./output/' + 'Results of ' + fn +"_" + model_name +"_"+ str(num_epochs) + "_" + str(lr) + "_" + str(batch_size), res.T, fmt='%s')

    ### 分析'每层误差信息' ###
    result = np.array(result)
    test_lab = np.array(test_lab)
    print('[每层误差信息]')
    error = (result - test_lab)/test_lab
    print('Mean(error): {:.2%}.'.format(np.mean(error)))
    print('Max(error): {:.2%}.'.format(np.max(error)))
    print('Min(error): {:.2%}.'.format(np.min(error)))
    print('Std(error): {:.2f}.'.format(np.std(error)))
    # 调用误差 RMSE, Mae, R^2
    Rs = mean_squared_error(test_lab, result) ** 0.5
    Mae = mean_absolute_error(test_lab, result)
    R2_s = r2_score(test_lab, result)
    print('Root mean_squared_error: {:.2f}J, Mean_absolute_error: {:.2f}, R2_score: {:.2f}.'.format(Rs, Mae, R2_s))

    ### 分析'总误差信息' ###
    print('[总误差信息]')
    E1 = np.sum(test_lab)
    E2 = np.sum(result)
    Er = (E1-E2)/E2
    print('Actual total EC: {:.2f}J, Predicted total EC: {:.2f}J, Er: {:.2%}'.format(E1,E2,Er))

    res_error = [np.mean(error), np.max(error), np.min(error), np.std(error), E1, E2, Er, Rs, Mae, R2_s]
    np.savetxt('./output/' + 'Error of ' + fn + "_" + model_name + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size),
               np.array(res_error), fmt='%s')


