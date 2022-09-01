import openpyxl
from glob import glob
import argparse
import os

# Set argparse
parser = argparse.ArgumentParser(description='Sum_excel')

parser.add_argument('--file', metavar='DIR', default='./output/EC-syy-V2/res.xlsx',
                    help='path to excels')

model_names = ['EfficientNet-B3 \cite{tan2019efficientnet}', 'EfficientNet-B4 \cite{tan2019efficientnet}',
              'ResNet50 \cite{he2016deep}', 'ResNet152 \cite{he2016deep}',
              'DeiT-S \cite{touvron2021training}', 'DeiT-B \cite{touvron2021training}',
              'Swin-T \cite{liu2021swin}', 'Swin-S \cite{liu2021swin}',
              'DSN (ours)']


def print_exl(data, sheet_index):
    sheet = data.worksheets[sheet_index]
    table = list(sheet.values)
    res = []
    for i, line in enumerate(table):
        if i%2 == 0:
            j = int(i/2)
            a1 = list(table[i])
            a2 = list(table[i+1])
            print('{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} \\\\'.format(model_names[j],
                                                                            a1[0], a2[0],a1[1], a2[1],a1[2], a2[2]))
    print("------------------End------------------")

if __name__ == "__main__":
    # get all args params
    args = parser.parse_args()

    data = openpyxl.load_workbook(args.file)

    print_exl(data, 0)
    print_exl(data, 1)
    print_exl(data, 2)

