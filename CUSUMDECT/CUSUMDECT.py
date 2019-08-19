# encoding: UTF-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib



def detect_via_cusum_lg(ts, istart=30, threshold_times=5):
    """
    detect a time series using  cusum algorithm
    :param ts: the time series to be detected
    :param istart: the data from index 0 to index istart will be used as cold startup data to train
    :param threshold_times: the times for setting threshold
    :return:
    """

    S_h = 0
    S_l = 0
    S_list = np.zeros(istart)

    meanArray = talib.SMA(ts,timeperiod = istart)
    stdArray = talib.STDDEV(np.log(ts/meanArray),timeperiod = istart)
    for i in range(istart+1, len(ts)-1):
        tslog = np.log(ts[i] / meanArray[i - 1])

        S_h_ = max(0, S_h + tslog - stdArray[i-1])
        S_l_ = min(0, S_l + tslog + stdArray[i-1])

        if S_h_> threshold_times * stdArray[i-1]:
            S_list = np.append(S_list,1)
            S_h_ = 0
        elif abs(S_l_)> threshold_times *  stdArray[i-1]:
            S_list = np.append(S_list, -1)
            S_l_ = 0
        else:
            S_list = np.append(S_list, 0)
        S_h = S_h_
        S_l = S_l_
    return S_list


#数据导入
df5min =  pd.read_csv("bar5rb8888.csv")
dt0 = np.array(df5min["close"])

listup,listdown = [],[]
s_list = detect_via_cusum_lg(dt0,istart=30, threshold_times=5)
for i in range(0,len(s_list)):
    if s_list[i] == 1:
        listup.append(i)
    elif s_list[i] == -1 :
        listdown.append(i)


plt.subplot(2,1,1)
plt.plot(dt0, color='y', lw=2.)
plt.plot(dt0, '^', markersize=5, color='r', label='UP signal', markevery=listup)
plt.plot(dt0, 'v', markersize=5, color='g', label='DOWN signal', markevery=listdown)
plt.legend()
plt.subplot(2,1,2)
plt.title('s_list')
plt.plot(s_list,'r-')

plt.show()
