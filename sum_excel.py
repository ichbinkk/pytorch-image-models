import openpyxl
from glob import glob
import argparse
import os

# Set argparse
parser = argparse.ArgumentParser(description='Sum_excel')

parser.add_argument('--data_dir', metavar='DIR', default='./output/V4_ec',
                    help='path to excels')

if __name__ == "__main__":
    # get all args params
    args = parser.parse_args()
    out_file = os.path.join(args.data_dir, 'result.xlsx')

    matrix = []
    for root, dirs_name, files_name in os.walk(args.data_dir):
        for i in files_name:
            if i.split('.')[-1] in ['xlsx']:
                file_name = os.path.join(root, i)
                data = openpyxl.load_workbook(file_name)
                sheet = data.worksheets[0]
                table = list(sheet.values)
                res = []
                for v in table[0]:
                    res.append(v)
                matrix.append([res[0], res[6], res[7]*100, res[8]*100])
                # print(file_name)

    # matrix_t = list(map(list,zip(*matrix)))

    wb = openpyxl.Workbook()
    s = wb.get_sheet_by_name('Sheet')
    for line in matrix:
        s.append(line)
    wb.save(out_file)