# encoding: UTF-8
from pymongo import MongoClient, ASCENDING
import pandas as pd
import numpy as np
from datetime import datetime
import talib
import matplotlib.pyplot as plt
import scipy.stats as scs


class DataAnalyzer(object):
    def __init__(self, exportpath="C:\Project\\", datformat=['datetime', 'high', 'low', 'open', 'close', 'volume']):
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
        cursor = db[self.collection].find({'datetime': {'$gte': start, '$lt': end}}).sort("datetime", ASCENDING)
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
        # self.df["high"] = self.df['high'].astype(float)
        # self.df["low"] = self.df['low'].astype(float)
        # self.df["open"] = self.df['open'].astype(float)
        # self.df["close"] = self.df['close'].astype(float)
        # self.df["volume"] = self.df['volume'].astype(int)
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
                          'close': closeBarMin, 'volume': volumeBarmin}
                dfbarmin = dfbarmin.append(barMin, ignore_index=True)
                highBarMin = 0
                lowBarMin = 0
                openBarMin = 0
                volumeBarmin = 0
        if export2csv == True:
            dfbarmin.to_csv(self.exportpath + "bar" + str(barmins) + str(self.collection) + ".csv", index=True,
                            header=True)
        return dfbarmin

    def dfcci(self, inputdf, n, export2csv=True):
        """调用talib方法计算CCI指标，写入到df并输出"""
        dfcci = inputdf
        dfcci["cci"] = None
        for i in range(n, len(inputdf)):
            df_ne = inputdf.loc[i - n + 1:i, :]
            cci = talib.CCI(np.array(df_ne["high"]), np.array(df_ne["low"]), np.array(df_ne["close"]), n)
            dfcci.loc[i, "cci"] = cci[-1]

        dfcci = dfcci.fillna(0)
        dfcci = dfcci.replace(np.inf, 0)
        if export2csv == True:
            dfcci.to_csv(self.exportpath + "dfcci" + str(self.collection) + ".csv", index=True, header=True)
        return dfcci

    # --------------------------------------------------------------
    def Percentage(self, inputdf, export2csv=True):
        """调用talib方法计算CCI指标，写入到df并输出"""
        dfPercentage = inputdf
        # dfPercentage["Percentage"] = None
        for i in range(1, len(inputdf)):
            # if dfPercentage.loc[i,"close"]>dfPercentage.loc[i,"open"]:
            #     percentage = ((dfPercentage.loc[i,"high"] - dfPercentage.loc[i-1,"close"])/ dfPercentage.loc[i-1,"close"])*100
            # else:
            #     percentage = (( dfPercentage.loc[i,"low"] - dfPercentage.loc[i-1,"close"] )/ dfPercentage.loc[i-1,"close"])*100
            if dfPercentage.loc[i - 1, "close"] == 0.0:
                percentage = 0
            else:
                percentage = ((dfPercentage.loc[i, "close"] - dfPercentage.loc[i - 1, "close"]) / dfPercentage.loc[
                    i - 1, "close"]) * 100.0
            dfPercentage.loc[i, "Perentage"] = percentage

        dfPercentage = dfPercentage.fillna(0)
        dfPercentage = dfPercentage.replace(np.inf, 0)
        if export2csv == True:
            dfPercentage.to_csv(self.exportpath + "Percentage_" + str(self.collection) + ".csv", index=True,
                                header=True)
        return dfPercentage

    def resultValuate(self, inputdf, nextBar, export2csv=True):
        summayKey = ["Percentage", "TestValues"]
        dft = pd.DataFrame(columns=summayKey)

    def dfMACD(self, inputdf, n, export2csv=False):
        """调用talib方法计算MACD指标，写入到df并输出"""
        dfMACD = inputdf
        for i in range(n, len(inputdf)):
            df_ne = inputdf.loc[i - n + 1:i, :]
            macd, signal, hist = talib.MACD(np.array(df_ne["close"]), 12, 26, 9)

            dfMACD.loc[i, "macd"] = macd[-1]
            dfMACD.loc[i, "signal"] = signal[-1]
            dfMACD.loc[i, "hist"] = hist[-1]
            if dfMACD.loc[i, "hist"] > 0 and dfMACD.loc[i - 1, "hist"] <= 0:
                dfMACD.loc[i, "histIndictor"] = 1
            elif dfMACD.loc[i, "hist"] < 0 and dfMACD.loc[i - 1, "hist"] >= 0:
                dfMACD.loc[i, "histIndictor"] = -1

        dfMACD = dfMACD.fillna(0)
        dfMACD = dfMACD.replace(np.inf, 0)
        if export2csv == True:
            dfMACD.to_csv(self.exportpath + "macd" + str(self.collection) + ".csv", index=True, header=True)
        return dfMACD


    def dfBOLL(self, inputdf, n, dev, export2csv=False):
        """调用talib方法计算MACD指标，写入到df并输出"""
        # mid = self.sma(n, array)
        # std = self.std(n, array)
        #
        # up = mid + std * dev
        # down = mid - std * dev
        dfBil = inputdf
        for i in range(100, len(inputdf)):
            df_ne = inputdf.loc[i - 100 + 1:i, :]
            mid = talib.SMA(np.array(df_ne["close"]), n)
            std = talib.STDDEV(np.array(df_ne["close"]), n)
            up = mid[-1] + std[-1] * dev
            down = mid[-1] - std[-1] * dev
            dfBil.loc[i, "mid"] = mid[-1]
            dfBil.loc[i, "up"] = up
            dfBil.loc[i, "down"] = down
            if dfBil.loc[i, "up"] != np.inf and dfBil.loc[i, "high"] > dfBil.loc[i, "up"]:
                dfBil.loc[i, "BuyPoint"] = dfBil.loc[i, "high"] - dfBil.loc[i, "up"]
            elif dfBil.loc[i, "down"] != np.inf and dfBil.loc[i, "low"] < dfBil.loc[i, "down"]:
                dfBil.loc[i, "ShortPoint"] = dfBil.loc[i, "low"] - dfBil.loc[i, "down"]

        dfBil = dfBil.fillna(0)
        dfBil = dfBil.replace(np.inf, 0)
        if export2csv == True:
            dfBil.to_csv(self.exportpath + "BILLBOLL" + str(self.collection) + ".csv", index=True, header=True)
        return dfBil

    def addResultBar(self, inputdf, startBar=2, endBar=12, step=2, export2csv=False):
        dfaddResultBar = inputdf
        ######cci在（100 - 200),(200 -300）后的第2根，第4根，第6根的价格走势######################

        for i in range(1, len(dfaddResultBar) - endBar - step):
            for nextbar in range(startBar, endBar, step):
                dfaddResultBar.loc[i, "next" + str(nextbar) + "BarDiffer"] = dfaddResultBar.loc[i + nextbar, "close"] - \
                                                                             dfaddResultBar.loc[i, "close"]
                if dfaddResultBar.loc[i, "close"] > dfaddResultBar.loc[i + nextbar, "close"]:
                    dfaddResultBar.loc[i, "next" + str(nextbar) + "BarClose"] = -1
                elif dfaddResultBar.loc[i, "close"] < dfaddResultBar.loc[i + nextbar, "close"]:
                    dfaddResultBar.loc[i, "next" + str(nextbar) + "BarClose"] = 1

        #     #######计算######################
        #         dfaddResultBar.loc[i,"next5BarCloseMakrup"] = dfaddResultBar.loc[i+5,"close"] - dfaddResultBar.loc[i,"close"]
        dfaddResultBar = dfaddResultBar.fillna(0)
        dfaddResultBar = dfaddResultBar.replace(np.inf, 0)
        if export2csv == True:
            dfaddResultBar.to_csv(self.exportpath + "addResultBar" + str(self.collection) + ".csv", index=True,
                                  header=True)
        return dfaddResultBar

    def resultOutput(self, de_anaylsisH, startBar=2, endBar=12, step=2, export2csv=False):
        HCount = len(de_anaylsisH)
        # LCount = de_anaylsisL['ShortPoint'].count()
        print ("CheckPoint : %s" % (HCount))
        dfResult = pd.DataFrame()

        for bar in range(startBar, endBar, step):
            Upcount = len(de_anaylsisH[de_anaylsisH["next" + str(bar) + "BarClose"] > 0])
            Upprecent = Upcount * 100.000 / HCount
            Downcount = len(de_anaylsisH[de_anaylsisH["next" + str(bar) + "BarClose"] < 0])
            Downprecent = Downcount * 100.000 / HCount
            closemean = np.mean(de_anaylsisH["next" + str(bar) + "BarDiffer"])
            closesum = np.sum(de_anaylsisH["next" + str(bar) + "BarDiffer"])
            closestd = np.std(de_anaylsisH["next" + str(bar) + "BarDiffer"])
            closemax = np.max(de_anaylsisH["next" + str(bar) + "BarDiffer"])
            closemin = np.min(de_anaylsisH["next" + str(bar) + "BarDiffer"])
            print("k线数量为 %s， ，第%s根K线结束, 上涨k线为%s 价格上涨概率为 %s%%;" % (HCount, bar, Upcount, Upprecent))
            print("k线数量为 %s， ，第%s根K线结束, 下跌k线为%s 价格下跌概率为 %s%%;" % (HCount, bar, Downcount, Downprecent))
            print('和值 %s, 均值 %s, std %s, max: %s, min: %s' % (closesum, closemean, closestd, closemax, closemin))
            dfResult = dfResult.append(
                [{"Bar Count": bar, "TotalCount": HCount, "Upcount": Upcount, "Upprecent": Upprecent,
                  "Downcount": Downcount, "Downprecent": Downprecent, "closesum": closesum,
                  "closemean": closemean, "closestd": closestd, "closemax": closemax,
                  "closemin": closemin
                  }])

        dfResult = dfResult.fillna(0)
        dfResult = dfResult.replace(np.inf, 0)
        if export2csv == True:
            dfResult.to_csv(self.exportpath + "addResultBar" + str(self.collection) + ".csv", index=True, header=True)
        return dfResult

    def macdAnalysis(self, inputdf, export2csv=True):
        dfMACD = inputdf
        dfAnalysis = pd.DataFrame()
        #######################################分析cci分布########################################

        for hist in range(10, 25, 5):
            lpHigh = np.percentile(dfMACD['macd'], 100 - hist)
            lpLow = np.percentile(dfMACD['macd'], hist)

            df = pd.DataFrame()
            de_anaylsisH = dfMACD.loc[(dfMACD["macd"] >= lpHigh)]
            de_anaylsisH = de_anaylsisH.loc[(de_anaylsisH["histIndictor"] == 1)]
            df = self.resultOutput(de_anaylsisH, 2, 12, 2)
            df["hist"] = lpHigh
            df["histIndictor"] = 1
            dfAnalysis = dfAnalysis.append(df)

            df = pd.DataFrame()
            de_anaylsisL = dfMACD.loc[(dfMACD["macd"] <= lpLow)]
            de_anaylsisL = de_anaylsisL.loc[(de_anaylsisL["histIndictor"] == -1)]
            df = self.resultOutput(de_anaylsisL, 2, 12, 2)
            df["hist"] = lpHigh
            df["histIndictor"] = -1
            dfAnalysis = dfAnalysis.append(df)

        dfAnalysis = dfAnalysis.fillna(0)
        dfAnalysis = dfAnalysis.replace(np.inf, 0)
        if export2csv == True:
            dfAnalysis.to_csv(self.exportpath + "_Anaylsis" + str(self.collection) + ".csv", index=False, header=True)
        return dfAnalysis



if __name__ == '__main__':
    DA = DataAnalyzer()
    # 数据库导入
    start = datetime.strptime("20180801", '%Y%m%d')
    end = datetime.strptime("20190501", '%Y%m%d')
    df = DA.db2df(db="VnTrader_1Min_Db", collection="CF905", start=start, end=end)
    # csv导入
    # df = DA.csv2df("rb1905.csv")
    df5min = DA.df2Barmin(df, 10)


    # print ("Dev is %s-------------------" %dev)
    df5minAdd = DA.addResultBar(df5min, export2csv=True)
    dfMACD = DA.dfMACD(df5minAdd, 100, export2csv=True)
    DA.macdAnalysis(dfMACD, export2csv=True)


