# encoding: UTF-8
from pymongo import MongoClient, ASCENDING
import pandas as pd
import numpy as np
from datetime import datetime
import talib
import matplotlib.pyplot as plt
import scipy.stats as scs

class DataAnalyzer(object):
    def __init__(self, exportpath="C:\Project\\", datformat=['datetime', 'high', 'low', 'open', 'close','volume']):
        self.mongohost = None
        self.mongoport = None
        self.db = None
        self.collection = None
        self.df = pd.DataFrame()
        self.exportpath = exportpath
        self.datformat = datformat

    def db2df(self, db, collection, start, end, mongohost="localhost", mongoport=27017, export2csv=False):
        """读取MongoDB数据库行情记录，输出到Dataframe中"""
        self.mongohost = mongohost
        self.mongoport = mongoport
        self.db = db
        self.collection = collection
        dbClient = MongoClient(self.mongohost, self.mongoport, connectTimeoutMS=500)
        db = dbClient[self.db]
        cursor = db[self.collection].find({'datetime':{'$gte':start, '$lt':end}}).sort("datetime",ASCENDING)
        self.df = pd.DataFrame(list(cursor))
        self.df = self.df[self.datformat]
        self.df = self.df.reset_index(drop=True)
        path = self.exportpath + self.collection + ".csv"
        if export2csv == True:
            self.df.to_csv(path, index=True, header=True)
        return self.df

    def csv2df(self, csvpath, dataname="csv_data", export2csv=False):
        """读取csv行情数据，输入到Dataframe中"""
        csv_df = pd.read_csv(csvpath)
        self.df = csv_df[self.datformat]
        self.df["datetime"] = pd.to_datetime(self.df['datetime'])
        self.df = self.df.reset_index(drop=True)
        path = self.exportpath + dataname + ".csv"
        if export2csv == True:
            self.df.to_csv(path, index=True, header=True)
        return self.df

    def df2Barmin(self, inputdf, barmins, crossmin=1, export2csv=False):
        """输入分钟k线dataframe数据，合并多多种数据，例如三分钟/5分钟等，如果开始时间是9点1分，crossmin = 0；如果是9点0分，crossmin为1"""
        dfbarmin = pd.DataFrame()
        highBarMin = 0
        lowBarMin = 0
        openBarMin = 0
        volumeBarmin = 0
        datetime = 0
        for i in range(0, len(inputdf) - 1):
            bar = inputdf.iloc[i, :].to_dict()
            if openBarMin == 0:
                openBarmin = bar["open"]
            if highBarMin == 0:
                highBarMin = bar["high"]
            else:
                highBarMin = max(bar["high"], highBarMin)

            if lowBarMin == 0:
                lowBarMin = bar["low"]
            else:
                lowBarMin = min(bar["low"], lowBarMin)
            closeBarMin = bar["close"]
            datetime = bar["datetime"]
            volumeBarmin += int(bar["volume"])
            # X分钟已经走完
            if not (bar["datetime"].minute + crossmin) % barmins:  # 可以用X整除
                # 生成上一X分钟K线的时间戳
                barMin = {'datetime': datetime, 'high': highBarMin, 'low': lowBarMin, 'open': openBarmin,
                          'close': closeBarMin, 'volume' : volumeBarmin}
                dfbarmin = dfbarmin.append(barMin, ignore_index=True)
                highBarMin = 0
                lowBarMin = 0
                openBarMin = 0
                volumeBarmin = 0
        if export2csv == True:
            dfbarmin.to_csv(self.exportpath + "bar" + str(barmins)+ str(self.collection) + ".csv", index=True, header=True)
        return dfbarmin



    #--------------------------------------------------------------
    def Percentage(self, inputdf, export2csv=True):
        """ 计算 Percentage """
        dfPercentage = inputdf
        for i in range(1, len(inputdf)):

            if dfPercentage.loc[ i - 1, "close"] == 0.0:
                percentage = 0
            else:
                percentage = ((dfPercentage.loc[i, "close"] - dfPercentage.loc[i - 1, "close"]) / dfPercentage.loc[ i - 1, "close"]) * 100.0
            dfPercentage.loc[i, "Perentage"] = percentage

        dfPercentage = dfPercentage.fillna(0)
        dfPercentage = dfPercentage.replace(np.inf, 0)
        if export2csv == True:
            dfPercentage.to_csv(self.exportpath + "Percentage_" + str(self.collection) + ".csv", index=True, header=True)
        return dfPercentage

    def resultValuate(self,inputdf, nextBar, export2csv=True):
        summayKey = ["Percentage","TestValues"]
        dft = pd.DataFrame(columns=summayKey)


    def addResultBar(self, inputdf, export2csv = False):
        dfaddResultBar = inputdf
        ######cci在（100 - 200),(200 -300）后的第2根，第4根，第6根的价格走势######################
        dfaddResultBar["next2BarClose"] = None
        dfaddResultBar["next4BarClose"] = None
        dfaddResultBar["next6BarClose"] = None
        dfaddResultBar["next5BarCloseMakrup"] = None
        for i in range(1, len(dfaddResultBar) - 6):
            dfaddResultBar.loc[i, "next2BarPercentage"] = dfaddResultBar.loc[i + 2, "close"] - dfaddResultBar.loc[i, "close"]
            dfaddResultBar.loc[i, "next4BarPercentage"] = dfaddResultBar.loc[i + 4, "close"] - dfaddResultBar.loc[i, "close"]
            dfaddResultBar.loc[i, "next6BarPercentage"] = dfaddResultBar.loc[i + 6, "close"] - dfaddResultBar.loc[i, "close"]
            if dfaddResultBar.loc[i, "close"] > dfaddResultBar.loc[i + 2, "close"]:
                dfaddResultBar.loc[i, "next2BarClose"] = -1
            elif dfaddResultBar.loc[i, "close"] < dfaddResultBar.loc[i + 2, "close"]:
                dfaddResultBar.loc[i, "next2BarClose"] = 1

            if dfaddResultBar.loc[i, "close"] > dfaddResultBar.loc[i + 4, "close"]:
                dfaddResultBar.loc[i, "next4BarClose"] = -1
            elif dfaddResultBar.loc[i, "close"] < dfaddResultBar.loc[i + 4, "close"]:
                dfaddResultBar.loc[i, "next4BarClose"] = 1

            if dfaddResultBar.loc[i, "close"] > dfaddResultBar.loc[i + 6, "close"]:
                dfaddResultBar.loc[i, "next6BarClose"] = -1
            elif dfaddResultBar.loc[i, "close"] < dfaddResultBar.loc[i + 6, "close"]:
                dfaddResultBar.loc[i, "next6BarClose"] = 1

        dfaddResultBar = dfaddResultBar.fillna(0)
        if export2csv == True:
            dfaddResultBar.to_csv(self.exportpath + "addResultBar" + str(self.collection) + ".csv", index=True, header=True)
        return dfaddResultBar


def PrecentAnalysis(inputdf):
    dfPercentage = inputdf
    #######################################分析分布########################################
    plt.figure(figsize=(10,3))
    plt.hist(dfPercentage['Perentage'],bins=300,histtype='bar',align='mid',orientation='vertical',color='r')
    plt.show()



    for Perentagekey in range(1,5):
        lpHigh = np.percentile(dfPercentage['Perentage'], 100-Perentagekey)
        lpLow = np.percentile(dfPercentage['Perentage'], Perentagekey)

        de_anaylsisH = dfPercentage.loc[(dfPercentage["Perentage"]>= lpHigh)]
        HCount = de_anaylsisH['Perentage'].count()
        de_anaylsisL = dfPercentage.loc[(dfPercentage["Perentage"] <= lpLow)]
        LCount = de_anaylsisL['Perentage'].count()


        percebtage = de_anaylsisH[de_anaylsisH["next2BarClose"]>0]["next2BarClose"].count()*100.000/HCount
        de_anaylsisHsum = de_anaylsisH["next2BarPercentage"].sum()
        de_anaylsisLsum = de_anaylsisL["next2BarPercentage"].sum()
        print('Precent 大于 %s, %s时候，k线数量为 %s，第二根K线结束价格上涨概率为 %s%%;' %(lpHigh,100-Perentagekey,HCount , percebtage))
        print('和值 %s' %(de_anaylsisHsum))

        de_anaylsisL = dfPercentage.loc[(dfPercentage["Perentage"]<= lpLow)]
        percebtage = de_anaylsisL[de_anaylsisL["next2BarClose"]<0]["next2BarClose"].count()*100.000/LCount
        print('Precent 小于于 %s, %s时候，k线数量为 %s, 第二根K线结束价格下跌概率为 %s%%' %(lpLow,Perentagekey,LCount, percebtage))
        print('和值 %s' %(de_anaylsisLsum))

        de_anaylsisHsum = de_anaylsisH["next4BarPercentage"].sum()
        de_anaylsisLsum = de_anaylsisL["next4BarPercentage"].sum()
        percebtage = de_anaylsisH[de_anaylsisH["next4BarClose"] > 0]["next2BarClose"].count() * 100.000 / HCount
        print('Precent 大于 %s, %s时候，第四根K线结束价格上涨概率为 %s%%' % (lpHigh, 100 - Perentagekey, percebtage))
        # print('和值 %s' % (de_anaylsisHsum))
        percebtage = de_anaylsisL[de_anaylsisL["next4BarClose"] < 0]["next2BarClose"].count() * 100.000 / LCount
        print('Precent 小于于 %s, %s时候，第四根K线结束价格下跌概率为 %s%%' % (lpLow, Perentagekey, percebtage))
        print('和值 %s' % (de_anaylsisLsum))

        de_anaylsisHsum = de_anaylsisH["next6BarPercentage"].sum()
        de_anaylsisLsum = de_anaylsisL["next6BarPercentage"].sum()
        percebtage = de_anaylsisH[de_anaylsisH["next6BarClose"] > 0]["next2BarClose"].count() * 100.000 / HCount
        print('Precent 大于 %s, %s时候，第六根K线结束价格上涨概率为 %s%%' % (lpHigh, 100 - Perentagekey, percebtage))
        print('和值 %s' % (de_anaylsisHsum))
        percebtage = de_anaylsisL[de_anaylsisL["next6BarClose"] < 0]["next2BarClose"].count() * 100.000 /LCount
        print('Precent 小于于 %s, %s时候，第六根K线结束价格下跌概率为 %s%%' % (lpLow, Perentagekey, percebtage))
        print('和值 %s' % (de_anaylsisLsum))



if __name__ == '__main__':
    DA = DataAnalyzer()
    #数据库导入
    start = datetime.strptime("20180901", '%Y%m%d')
    end = datetime.today()
    df = DA.db2df(db="VnTrader_1Min_Db", collection="m1905", start = start, end = end)
    #csv导入
    # df = DA.csv2df("rb1905.csv")
    df10min = DA.df2Barmin(df,10)
    dfPercentage = DA.Percentage(df10min)
    dfPercentage = DA.addResultBar(dfPercentage)
    PrecentAnalysis(dfPercentage)

