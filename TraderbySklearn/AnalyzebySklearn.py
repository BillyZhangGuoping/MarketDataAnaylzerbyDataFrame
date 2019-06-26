# encoding: UTF-8
import warnings
warnings.filterwarnings("ignore")
from pymongo import MongoClient, ASCENDING
import pandas as pd
import numpy as np
from datetime import datetime
import talib
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.model_selection import train_test_split
# LogisticRegression 逻辑回归
from sklearn.linear_model import LogisticRegression
# DecisionTreeClassifier 决策树
from sklearn.tree import DecisionTreeClassifier
# SVC 支持向量分类
from sklearn.svm import SVC
# MLP 神经网络
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
class DataAnalyzerforSklearn(object):
    """
    这个类是为了SVM做归纳分析数据，以未来6个bar的斜率线性回归为判断分类是否正确。
    不是直接分析HLOC，而且用下列分非线性参数（就是和具体点位无关）
    1.Percentage
    2.std
    4.MACD
    5.CCI
    6.ATR
    7. 该bar之前的均线斜率
    8. RSI
    """
    def __init__(self, exportpath="C:\\Project\\", datformat=['datetime', 'high', 'low', 'open', 'close','volume']):
        self.mongohost = None
        self.mongoport = None
        self.db = None
        self.collection = None
        self.df = pd.DataFrame()
        self.exportpath = exportpath
        self.datformat = datformat
        self.startBar = 2
        self.endBar = 12
        self.step = 2
        self.pValue = 0.015
    #-----------------------------------------导入数据-------------------------------------------------
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
    #-----------------------------------------开始计算指标-------------------------------------------------
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
    def dfatr(self, inputdf, n, export2csv=True):
        """调用talib方法计算ATR指标，写入到df并输出"""
        dfatr = inputdf
        for i in range((n+1), len(inputdf)):
            df_ne = inputdf.loc[i - n :i, :]
            atr = talib.ATR(np.array(df_ne["high"]), np.array(df_ne["low"]), np.array(df_ne["close"]), n)
            dfatr.loc[i, "atr"] = atr[-1]
        dfatr = dfatr.fillna(0)
        dfatr = dfatr.replace(np.inf, 0)
        if export2csv == True:
            dfatr.to_csv(self.exportpath + "dfatr" + str(self.collection) + ".csv", index=True, header=True)
        return dfatr
    def dfrsi(self, inputdf, n, export2csv=True):
        """调用talib方法计算ATR指标，写入到df并输出"""
        dfrsi = inputdf
        dfrsi["rsi"] = None
        for i in range(n+1, len(inputdf)):
            df_ne = inputdf.loc[i - n :i, :]
            rsi = talib.RSI(np.array(df_ne["close"]), n)
            dfrsi.loc[i, "rsi"] = rsi[-1]
        dfrsi = dfrsi.fillna(0)
        dfrsi = dfrsi.replace(np.inf, 0)
        if export2csv == True:
            dfrsi.to_csv(self.exportpath + "dfrsi" + str(self.collection) + ".csv", index=True, header=True)
        return dfrsi
    def Percentage(self, inputdf, export2csv=True):
        """调用talib方法计算CCI指标，写入到df并输出"""
        dfPercentage = inputdf
        # dfPercentage["Percentage"] = None
        for i in range(1, len(inputdf)):
            # if dfPercentage.loc[i,"close"]>dfPercentage.loc[i,"open"]:
            #     percentage = ((dfPercentage.loc[i,"high"] - dfPercentage.loc[i-1,"close"])/ dfPercentage.loc[i-1,"close"])*100
            # else:
            #     percentage = (( dfPercentage.loc[i,"low"] - dfPercentage.loc[i-1,"close"] )/ dfPercentage.loc[i-1,"close"])*100
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
    def dfMACD(self, inputdf, n, export2csv=False):
        """调用talib方法计算MACD指标，写入到df并输出"""
        dfMACD = inputdf
        for i in range(n, len(inputdf)):
            df_ne = inputdf.loc[i - n + 1:i, :]
            macd,signal,hist = talib.MACD(np.array(df_ne["close"]),12,26,9)
            dfMACD.loc[i, "macd"] = macd[-1]
            dfMACD.loc[i, "signal"] = signal[-1]
            dfMACD.loc[i, "hist"] = hist[-1]
        dfMACD = dfMACD.fillna(0)
        dfMACD = dfMACD.replace(np.inf, 0)
        if export2csv == True:
            dfMACD.to_csv(self.exportpath + "macd" + str(self.collection) + ".csv", index=True, header=True)
        return dfMACD
    def dfSTD(self, inputdf, n, export2csv=False):
        """调用talib方法计算MACD指标，写入到df并输出"""
        dfSTD = inputdf
        for i in range(n, len(inputdf)):
            df_ne = inputdf.loc[i - n + 1:i, :]
            std = talib.STDDEV(np.array(df_ne["close"]),n)
            dfSTD.loc[i, "std"] = std[-1]
        dfSTD = dfSTD.fillna(0)
        dfSTD = dfSTD.replace(np.inf, 0)
        if export2csv == True:
            dfSTD.to_csv(self.exportpath + "dfSTD" + str(self.collection) + ".csv", index=True, header=True)
        return dfSTD
    #-----------------------------------------加入趋势分类-------------------------------------------------
    def addTrend(self, inputdf,  trendsetp=6, export2csv=False):
        """以未来6个bar的斜率线性回归为判断分类是否正确"""
        dfTrend = inputdf
        for i in range(1, len(dfTrend) - trendsetp-1):
            histRe = np.array(dfTrend["close"])[i:i+trendsetp]
            xAixs = np.arange(trendsetp) + 1
            res = st.linregress(y=histRe, x=xAixs)
            if res.pvalue < self.pValue+0.01:
                if res.slope > 0.5:
                    dfTrend.loc[i,"tradeindictor"] = 1
                elif res.slope < -0.5:
                    dfTrend.loc[i, "tradeindictor"] = -1
        dfTrend = dfTrend.fillna(0)
        dfTrend = dfTrend.replace(np.inf, 0)
        if export2csv == True:
            dfTrend.to_csv(self.exportpath + "addTrend" + str(self.collection) + ".csv", index=True, header=True)
        return dfTrend
def GirdValuate(X_train, y_train):
    """1）LogisticRegression
    逻辑回归
    2）DecisionTreeClassifier
    决策树
    3）SVC
    支持向量分类
    4）MLP
    神经网络"""
    clf_DT=DecisionTreeClassifier()
    param_grid_DT= {'max_depth': [1,2,3,4,5,6]}
    clf_Logit=LogisticRegression()
    param_grid_logit= {'solver': ['liblinear','lbfgs','newton-cg','sag']}
    clf_svc=SVC()
    param_grid_svc={'kernel':('linear', 'poly', 'rbf', 'sigmoid'),
                    'C':[1, 2, 4],
                    'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]}
    clf_mlp = MLPClassifier()
    param_grid_mlp= {"hidden_layer_sizes": [(100,), (100, 30)],
                                 "solver": ['adam', 'sgd', 'lbfgs'],
                                 "max_iter": [20],
                                 "verbose": [False]
                                 }
    #打包参数集合
    clf=[clf_DT,clf_Logit,clf_mlp,clf_svc]
    param_grid=[param_grid_DT,param_grid_logit,param_grid_mlp,param_grid_svc]
    from sklearn.model_selection import StratifiedKFold  # 交叉验证
    kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)  # 将训练/测试数据集划分10个互斥子集，这样方便多进程测试
    #网格测试
    for i in range(0,4):
        grid=GridSearchCV(clf[i], param_grid[i], scoring='accuracy',n_jobs = -1,cv = kflod)
        grid.fit(X_train, y_train)
        print (grid.best_params_,': ',grid.best_score_)
if __name__ == '__main__':
    # 读取数据
    # exportpath = "C:\\Users\shui0\OneDrive\Documents\Project\\"
    exportpath = "C:\Project\\"
    DA = DataAnalyzerforSklearn(exportpath)
    #数据库导入
    start = datetime.strptime("20180501", '%Y%m%d')
    end = datetime.strptime("20190501", '%Y%m%d')
    df = DA.db2df(db="VnTrader_1Min_Db", collection="rb8888", start = start, end = end)
    df5min = DA.df2Barmin(df, 5)
    df5minAdd = DA.addTrend(df5min, export2csv=True)
    df5minAdd = DA.dfMACD(df5minAdd, n=34, export2csv=True)
    df5minAdd = DA.dfatr(df5minAdd, n=25, export2csv=True)
    df5minAdd = DA.dfrsi(df5minAdd, n=35, export2csv=True)
    df5minAdd = DA.dfcci(df5minAdd,n = 30,export2csv=True)
    df5minAdd = DA.dfSTD(df5minAdd, n=30, export2csv=True)
    df5minAdd = DA.Percentage(df5minAdd,export2csv = True)
    #划分测试验证。
    df_test = df5minAdd.loc[60:,:]        #只从第60个开始分析，因为之前很多是空值
    y= np.array(df_test["tradeindictor"]) #只保留结果趋势结果，转化为数组
    X = df_test.drop(["close","datetime","high","low","open","volume"],axis = 1).values #不是直接分析HLOC，只保留特征值，转化为数组
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0) #三七
    print("训练集长度: %s, 测试集长度: %s" %(len(X_train),len(X_test)))
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import SelectPercentile
    from sklearn.feature_selection import mutual_info_classif
    #特征工作，可以按照百分比选出最高分特征类，取最优70%，也可以用SelectKBest，指定要几个特征类。
    print(X_train.shape)
    selectPer = SelectPercentile(mutual_info_classif, percentile=70)
    # selectPer = SelectKBest(mutual_info_classif, k=7)
    X_train = selectPer.fit_transform(X_train, y_train)
    print(X_train.shape)
    X_test = selectPer.transform(X_test)
    # 也可以用Fpr选择
    # selectFea=SelectFpr(alpha=0.01)
    # X_train_new = selectFea.fit_transform(X_train, y_train)
    # X_test_new = selectFea.transform(X_test)
    # 这里使用下面模式进行分析，然后利用网格调参
    GirdValuate(X_train,y_train)
    # 使用选取最好的模型，进行测试看看拼接
    # • 模型预测：model.predict()
    # • Accuracy：metrics.accuracy_score()
    # • Presicion：metrics.precision_score()
    # • Recall：metrics.recall_score()
    from sklearn import metrics
    clf_selected=MLPClassifier(hidden_layer_sizes=(100,30), max_iter=20, solver='adam') #此处填入网格回测最优模型和参数,
    # {'hidden_layer_sizes': (100, 30), 'max_iter': 20, 'solver': 'adam', 'verbose': False} :  0.9897016507648039
    clf_selected.fit(X_train, y_train)
    y_pred = clf_selected.predict(X_test)
    #accuracy
    accuracy=metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    print ('accuracy:',accuracy)
    #precision
    precision=metrics.precision_score(y_true=y_test, y_pred=y_pred,average="micro")
    print ('precision:',precision)
    #recall
    recall=metrics.recall_score(y_true=y_test, y_pred=y_pred,average="micro")
    print ('recall:',recall)
    #实际值和预测值
    print (y_test)
    print (y_pred)
    dfresult = pd.DataFrame({'Actual':y_test,'Predict':y_pred})
    dfresult.to_csv(exportpath + "result"  + ".csv", index=True, header=True)
    from sklearn.externals import joblib
    #模型保存到本地
    joblib.dump(clf_selected,'clf_selected.m')
    #模型的恢复
    clf_tmp=joblib.load('clf_selected.m')