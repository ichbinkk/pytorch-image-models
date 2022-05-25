"""
Finetuning Torchvision Models For 3D-ECP
=============================

**Author:** `ichbinkk`
*date: 10/11/2021
"""
from __future__ import print_function
from __future__ import division
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device_id = [0, 1, 2, 3]

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time

import copy

from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image
import timm.models as tm
import argparse

# 衡量误差
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square

from ecp_utils import *
from torch.utils.tensorboard import SummaryWriter

# Number of classes in the dataset
num_classes = 1

lr = 0.001

parm = {}  # 初始化保存模块参数的parm字典

# Set argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('--data_dir', metavar='DIR', default='../dataset/V4_ec',
                    help='path to dataset')
'''
    Setting model and training params, some can use parser to get value.
    Models to choose from [resnet, regnet, efficientnet, vit, pit, mixer, deit, swin-vit
    alexnet, vgg, squeezenet, densenet, inception]
'''
parser.add_argument('--model', default='swin_vit_s', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet18"')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-ep', '--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: )')
parser.add_argument('-ft', '--use-pretrained', type=bool, default=False, metavar='N',
                    help='Flag to use fine tuneing(default: False)')
parser.add_argument('-fe', '--feature-extract', type=bool, default=False, metavar='N',
                    help='False to finetune the whole model. True to update the reshaped layer params(default: False)')


def train_model(model, dataloaders, criterion, optimizer, GT, aVal, bVal, num_epochs=25, is_inception=False):
    since = time.time()

    # log loss history
    train_loss, val_loss = 0, 0
    train_acc_history = []
    val_acc_history = []

    # log result
    last_result = []
    best_result = []
    layer_error = []

    # log metrics
    metrics_history = np.zeros([num_epochs, 3])

    best_model_wts = copy.deepcopy(model.state_dict())

    # for record min/max value
    best_epoch = 0
    min_loss = 999999
    min_metric = 999999

    # seg_index
    index = seg_index(os.path.join(infile, 'val.txt'))

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        result = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # inputs = inputs.to(device)
                # labels = labels.to(device)

                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                '''
                    Forward
                    track history if only in train
                '''
                with torch.set_grad_enabled(phase == 'train'):
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

                        '''save val result'''
                        if phase == 'val':
                            # result.append(outputs)
                            temp = outputs.detach().cpu().numpy()
                            for i in range(outputs.size()[0]):
                                result.append(temp[i])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)

            '''
                After training or val one epoch completed
            '''
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'train':
                train_loss = epoch_loss
                train_acc_history.append(epoch_loss)
            if phase == 'val':
                if epoch == num_epochs - 1:
                    last_result = result
                val_acc_history.append(epoch_loss)
                val_loss = epoch_loss

                # get every epoch error
                result = InvNormalize(result, aVal, bVal)

                '''layer-wise metric'''
                Rs = mean_squared_error(GT, result) ** 0.5
                Mae = mean_absolute_error(GT, result)
                R2_s = r2_score(GT, result)
                metrics_history[epoch][0] = Rs

                error = np.abs((result - GT) / GT)
                LME = np.mean(error)
                metrics_history[epoch][1] = LME

                '''model-wise metric'''
                Er = np.zeros([len(index)-1, 1])
                for i in range(len(index)-1):
                    t1 = GT[index[i]:index[i+1]]
                    t2 = result[index[i]:index[i+1]]
                    E1 = np.sum(GT[index[i]:index[i+1]])
                    E2 = np.sum(result[index[i]:index[i+1]])
                    Er[i] = 1 - np.abs((E1 - E2) / E1)
                Er = np.mean(Er)
                metrics_history[epoch][2] = Er
                'Epoch {}/{}'.format(epoch, num_epochs - 1)

                '''print metrics for each epoch'''
                print('[Epoch {}/{}] Train_loss: {:.4f} | Val_loss: {:.4f} | LME: {:.2%} | RMSE: {:.2f}J | MAE: {:.2f}J | R2_s: {:.2f} | Er: {:.2%}'.format(epoch+1,
                      num_epochs, train_loss, val_loss, LME, Rs, Mae, R2_s, Er))
                # print('GT: {:.2f}J | ECP: {:.2f}J | Er: {:.2%}'.format(E1, E2, Er))


                """ 
                    Choose the best model using various metrics
                    [Rs, LME, Er]
                """
                epoch_metric = LME
                if epoch_metric < min_metric:
                    min_metric = epoch_metric  # update min_loss
                    # print('[Best model updated] Min_metric: {:.2f} in Epoch {}/{}'.format(min_metric, epoch + 1,
                    #                                                                       num_epochs))
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    best_result = result
                    layer_error = error
        # print()

    time_elapsed = time.time() - since
    print('Training complete in time of {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best model with Min_metric: {:.2f} is in Epoch {}/{}'.format(min_metric, best_epoch+1, num_epochs))

    '''load best model weights'''
    model.load_state_dict(best_model_wts)

    # Save best model weights
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    save_path = os.path.join(out_path, 'Best_checkpoint.pth')
    torch.save(best_model_wts, save_path)

    return model, train_acc_history, val_acc_history, best_epoch, best_result, layer_error, metrics_history


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

    '''check output path for different data and models'''
    out_path = os.path.join('./output', fn, model_name)
    if not os.path.exists(out_path):
        # 如果不存在则创建目录
        os.makedirs(out_path)
    # writer = SummaryWriter(out_path)

    # Print the model we just instantiated
    # print(model_ft)

    '''
            Analyze flops and params
        '''
    if args.model in ['ecpnet', 'ecpnetno']:
        input = (torch.rand(1, 3, 224, 224), torch.rand(1, 1, 4))
    else:
        input = (torch.rand(1, 3, 224, 224),)
    '''[1] using fvcore'''
    # from fvcore.nn import FlopCountAnalysis, parameter_count_table, parameter_count
    # print(parameter_count_table(model_ft))
    #
    # flops = FlopCountAnalysis(model_ft, input)
    # print("FLOPs: ", flops.total())

    '''[2] using thop'''
    # from thop import profile
    # input = torch.randn(1, 3, 224, 224)
    # macs, params = profile(model_ft, inputs=(input,))
    # print('The model Params: {:.2f}M, MACs: {:.2f}M'.format(params/10e6, macs/10e6))
    '''[3] using torchstat'''
    # from torchstat import stat
    # stat(model_ft, (3, 224, 224))

    ######################################################################
    # Load Data
    # ---------
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

    # Send the model to GPU
    # model = torch.nn.DataParallel(model_ft, device_ids=device_id)
    model_ft = torch.nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)

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

    criterion = nn.MSELoss()

    '''
        load Val Ground Truth dataset to compare in every Val epoch
    '''
    val_lab = loadColStr(os.path.join(infile, 'val.txt'), 1)
    _, aVal, bVal = Normalize(val_lab)
    # train_lab = loadColStr(os.path.join(infile, 'train.txt'), 1)
    # _, aVal, bVal = Normalize(train_lab)

    '''Train and evaluate'''
    model_ft, train_hist, val_hist, best_epoch, result, layer_error, metrics_history = train_model(model_ft, dataloaders_dict,
        criterion, optimizer_ft, val_lab, aVal, bVal, num_epochs=num_epochs, is_inception=(model_name=="inception"))

    #######################################################################
    '''
        Plot training loss
    '''
    plt.figure()
    plt.title("Train and val Loss history vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), train_hist, label="train_hist")
    plt.plot(range(1,num_epochs+1), val_hist, label="val_hist")
    # plt.ylim((0, 2.))
    # plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    # plt.savefig(os.path.join(out_path, 'Hist_' + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size) + '.png'))
    plt.show()
    hist = np.vstack((train_hist, val_hist))
    hist_path = os.path.join(out_path, 'Hist_' + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size))
    # np.savetxt(hist_path, hist.T)

    #######################################################################
    ''' 
        Val phase
    '''

    ''' layered EC result '''
    plt.figure()
    plt.title(model_name + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size) + "Validation Result")
    ts = range(len(val_lab))
    plt.plot(ts, val_lab, label="val_lab")
    plt.plot(ts, result, label="pred_lab")
    plt.legend()
    plt.show()

    res = np.vstack((val_lab, result))
    res_path = os.path.join(out_path, 'Results_' + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size))
    # np.savetxt(res_path, res.T, fmt='%s')


    ''' layered error '''
    # writer.add_scalars('Validation/result', {'val_lab': val_lab, 'pred_lab': result}, ts)

    # print()
    # print('[Layered error]')
    # print('Mean(error): {:.2%} | Max(error): {:.2%} | Min(error): {:.2%} | Std(error): {:.2f}'.format(
    #     np.mean(error), np.max(error), np.min(error), np.std(error)))

    Rs = mean_squared_error(val_lab, result) ** 0.5
    Mae = mean_absolute_error(val_lab, result)
    R2_s = r2_score(val_lab, result)
    # print()
    # print('[Statistic error]')
    # print('RMSE: {:.2f}J | MAE: {:.2f} | R2_s: {:.2f}.'.format(Rs, Mae, R2_s))

    '''total error'''
    Er = metrics_history[best_epoch, 2]

    '''Print Best metrics'''
    print('RMSE: {:.2f}J | LME: {:.2%} | Er: {:.2%} '.format(Rs, np.mean(layer_error), Er))

    RE_history = metrics_history[:, 2]
    plt.figure()
    plt.plot(range(args.epochs), RE_history, label="Model-wise accuracy history vs. Epoch")
    plt.legend()
    plt.show()

    metrics_path = os.path.join(out_path, 'Metrics_history.xlsx')

    '''best performance'''
    res_error = [[model_name, np.max(layer_error), np.min(layer_error), np.std(layer_error), Mae, R2_s, Rs, np.mean(layer_error), Er]]
    error_path = os.path.join(out_path, 'Error_' + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size))
    # np.savetxt(error_path, np.array(res_error), fmt='%s')

    ''' 
       Save excel
    '''
    excel_path = res_path + '.xlsx'
    save_excel(res_error, metrics_history, res.T, hist.T, excel_path)


    ##########################################################################
    '''Test phase'''
    # print("------Test using best trained model------")
    # eval_EC(model_name, model_ft, test_path, infile, args.eval_phase)