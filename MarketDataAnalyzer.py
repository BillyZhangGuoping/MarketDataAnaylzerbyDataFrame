# encoding: UTF-8
from pymongo import MongoClient,ASCENDING
import pandas as pd
from datetime import datetime
class DataAnalyzer(object):
    def __init__(self,exportpath = "C:\Project\\", datformat = ['datetime', 'high', 'low', 'open', 'close']):
        self.mongohost = 0
        self.mongoport = 0
        self.db = 0
        self.collection = 0
        self.df = pd.DataFrame()
        self.exportpath = exportpath
        self.datformat = ['datetime', 'high', 'low', 'open', 'close']

    def db2df(self,db,collection,start,end,mongohost = "localhost",mongoport = 27017,export2xls = True):
        self.mongohost = mongohost
        self.mongoport = mongoport
        self.db = db
        self.collection = collection
        dbClient = MongoClient(self.mongohost, self.mongoport, connectTimeoutMS=500)
        db = dbClient[self.db]
        cursor = db[self.collection].find({'datetime': {'$gte': start},'datetime':{'$lt':end}}).sort("datetime", ASCENDING)
        self.df = pd.DataFrame(list(cursor))
        self.df = self.df[self.datformat]
        self.df = self.df.reset_index(drop=True)
        path = self.exportpath + self.collection + ".csv"
        if export2xls == True:
            self.df.to_csv(path , index=True, header=True)
        return self.df

    def df2Barmin(self, inputdf, barmins,crossmin = 1,export2xls = True):
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
            dfbarmin.to_excel(self.exportpath + self.collection + str(barmins) + ".xlsx" , index=True, header=True)
        return dfbarmin
DA = DataAnalyzer()
start = datetime(year=2017,month=7,day =1,hour=0, minute=0, second=0, microsecond=0)
end = datetime.today()
df = DA.db2df(db = "VnTrader_1Min_Db", collection = "rb0000",start = start, end =end)
