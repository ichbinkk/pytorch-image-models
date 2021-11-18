import os
import numpy as np
import argparse
import pandas as pd
from openpyxl import load_workbook


def init_args():
    # Set argparse
    parser = argparse.ArgumentParser(description='Process data')

    parser.add_argument('--file_name', metavar='file',
                        default='./output/lattice_ec/deit_base_patch16_224/Results_300_0.001_128',
                        help='path to data')
    parser.add_argument('--excel_name', metavar='file',
                        default='./A.xlsx',
                        help='path to data')

    args = parser.parse_args()

    return args


def txt2excel():
    '''write to excel'''
    writer = pd.ExcelWriter(args.excel_name)  # 写入Excel文件
    data = np.loadtxt(args.file_name, usecols=(1))
    data = pd.DataFrame(data)

    data.to_excel(writer, sheet_name='Sheet1', float_format='%.2f', header=False, index=False)
    writer.save()
    writer.close()

if __name__ == '__main__':
    args = init_args()
    txt2excel()

