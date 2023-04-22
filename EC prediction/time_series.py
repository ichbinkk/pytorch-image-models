# Import libraries

from random import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error  # 均方误差


def eval(val_lab, result):
    ''' layered error '''
    Rs = mean_squared_error(val_lab, result) ** 0.5
    # Mae = mean_absolute_error(val_lab, result)
    # R2_s = r2_score(val_lab, result)

    error = np.abs((result - val_lab) / val_lab)
    LME = np.mean(error)

    '''total error'''
    E1 = np.sum(val_lab)
    E2 = np.sum(result)
    Er = 1 - np.abs((E1 - E2) / E1)

    '''Print Best metrics'''
    print('RMSE: {:.2f}J | LME: {:.2%} | Er: {:.2%} '.format(Rs, LME, Er))


if __name__ == "__main__":

    # Generate a sample dataset
    # data = [x + random() for x in range(1, 100)]

    df = pd.read_excel('train.xlsx', index_col=None, header=None)
    data = df.values

    df = pd.read_excel('ec_time.xlsx', index_col=None, header=None)
    gt = df.values
    plt.plot(range(0, len(gt)), gt)

    '''
    AR — AutoRegression 自回归
    '''
    # 拟合模型
    from statsmodels.tsa.ar_model import AutoReg #直接导入自回归模型
    model = AutoReg(data, lags=1) #数据喂入模型进行拟合即可
    model_fit = model.fit()
    # plt.plot(range(0, len(data)), data)
    plt.show()

    # 预测
    yhat = model_fit.predict(len(data)+1, len(data)+len(gt))
    print(len(yhat))
    plt.plot(range(0, len(yhat)), yhat)
    plt.show()
    eval(gt, yhat)


    '''
    ARIMA AutoRegression Integrated Moving Average 自回归综合移动平均线
    '''
    # Import libraries
    from statsmodels.tsa.arima.model import ARIMA

    # fit model
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()

    # make prediction
    yhat = model_fit.predict(len(data) + 1, len(data) + len(gt))
    print(len(yhat))
    plt.plot(range(0, len(yhat)), yhat)
    plt.show()
    eval(gt, yhat)
