# encoding: UTF-8
from pymongo import MongoClient, ASCENDING
import pandas as pd
import numpy as np
from datetime import datetime
import talib


class DataAnalyzer(object):
    def __init__(self, exportpath="C:\Project\\", datformat=['datetime', 'high', 'low', 'open', 'close']):
        self.mongohost = None
        self.mongoport = None
        self.db = None
        self.collection = None
        self.df = pd.DataFrame()
        self.exportpath = exportpath
        self.datformat = ['datetime', 'high', 'low', 'open', 'close']

    def db2df(self, db, collection, start, end, mongohost="localhost", mongoport=27017, export2xls=True):
        """读取MongoDB数据库行情记录，输出到Dataframe中"""
        self.mongohost = mongohost
        self.mongoport = mongoport
        self.db = db
        self.collection = collection
        dbClient = MongoClient(self.mongohost, self.mongoport, connectTimeoutMS=500)
        db = dbClient[self.db]
        cursor = db[self.collection].find({'datetime': {'$gte': start}, 'datetime': {'$lt': end}}).sort("datetime",
                                                                                                        ASCENDING)
        self.df = pd.DataFrame(list(cursor))
        self.df = self.df[self.datformat]
        self.df = self.df.reset_index(drop=True)
        path = self.exportpath + self.collection + ".csv"
        if export2xls == True:
            self.df.to_csv(path, index=True, header=True)
        return self.df

    def csv2df(self, csvpath, dataname="csv_data", export2xls=True):
        """读取csv行情数据，输入到Dataframe中"""
        csv_df = pd.read_csv(csvpath)
        self.df = csv_df[self.datformat]
        self.df = self.df.reset_index(drop=True)
        path = self.exportpath + dataname + ".csv"
        if export2xls == True:
            self.df.to_csv(path, index=True, header=True)
        return self.df

    def df2Barmin(self, inputdf, barmins, crossmin=1, export2xls=True):
        """输入分钟k线dataframe数据，合并多多种数据，例如三分钟/5分钟等，如果开始时间是9点1分，crossmin = 0；如果是9点1分，crossmin为1"""
        dfbarmin = pd.DataFrame()
        highBarMin = 0
        lowBarMin = 0
        openBarMin = 0
        closeBarMin = 0
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
            # X分钟已经走完
            if not (bar["datetime"].minute + crossmin) % barmins:  # 可以用X整除
                # 生成上一X分钟K线的时间戳
                barMin = {'datetime': datetime, 'high': highBarMin, 'low': lowBarMin, 'open': openBarmin,
                          'close': closeBarMin}
                dfbarmin = dfbarmin.append(barMin, ignore_index=True)
                highBarMin = 0
                lowBarMin = 0
                openBarMin = 0
        if export2xls == True:
            dfbarmin.to_csv(self.exportpath + self.collection + str(barmins) + ".csv", index=True, header=True)
        return dfbarmin

    def dfcci(self, inputdf, n, export2xls=True):
        """调用talib方法计算CCI指标，写入到df并输出"""
        dfcci = inputdf
        dfcci["cci"] = None
        for i in range(n, len(inputdf) - n):
            df_ne = inputdf.loc[i - n + 1:i, :]
            cci = talib.CCI(np.array(df_ne["high"]), np.array(df_ne["low"]), np.array(df_ne["close"]), n)
            dfcci.loc[i, "cci"] = cci[-1]
            if export2xls == True:
                dfcci.to_csv(self.exportpath + "dfcci" + ".csv", index=True, header=True)
        return dfcci

if __name__ == '__main__':
	DA = DataAnalyzer()
	start = datetime(year=2019, month=3, day=1, hour=0, minute=0, second=0, microsecond=0)
	end = datetime.today()
	df = DA.db2df(db="VnTrader_1Min_Db", collection="rb1905", start = start, end = end)
	DA.dfcci(df, 15)

