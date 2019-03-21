# encoding: UTF-8
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("C:\\Project\\rb1905.csv")
df["datetime"] = pd.to_datetime(df['datetime'])
df = df.reset_index(drop=True)
# print(df)

# plt.figure(figsize = (18,9))
# plt.plot(range(df.shape[0]),(df['low']+df['high'])/2.0)
# plt.xticks(range(0,df.shape[0],500),df['datetime'].loc[::500],rotation=45)
# plt.xlabel('datetime',fontsize=18)
# plt.ylabel('Mid Price',fontsize=18)
# plt.show()

high_prices = df.loc[:,'high'].as_matrix()
low_prices = df.loc[:,'low'].as_matrix()
mid_prices = (high_prices+low_prices)/2.

train_data = mid_prices[:55000]
test_data = mid_prices[55000:]

scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
print(train_data)

smoothing_window_size = 2500
for di in range(0,50000,smoothing_window_size):
    print(di)
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

# You normalize the last bit of remaining data 
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

train_data = train_data.reshape(-1)
test_data = scaler.transform(test_data).reshape(-1)

EMA = 0.0
gamma = 0.1
for ti in range(52500):
    EMA = gamma*train_data[ti] + (1-gamma)*EMA
    train_data[ti] = EMA

all_mid_data = np.concatenate([train_data,test_data],axis=0)

# plt.figure(figsize = (18,9))
# plt.plot(range(df.shape[0]),(df['low']+df['high'])/2.0)
# plt.xticks(range(0,df.shape[0],500),df['datetime'].loc[::500],rotation=45)
# plt.xlabel('datetime',fontsize=18)
# plt.ylabel('Mid Price',fontsize=18)
# plt.show()