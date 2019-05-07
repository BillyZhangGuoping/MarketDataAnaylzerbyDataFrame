# encoding: UTF-8

from __future__ import print_function
import sys
import json
from datetime import datetime,date
from time import time, sleep

from pymongo import MongoClient, ASCENDING
import pandas as pd

from vnpy.trader.vtObject import VtBarData, VtTickData
from vnpy.trader.app.ctaStrategy.ctaBase import (MINUTE_DB_NAME,
                                                 DAILY_DB_NAME,
                                                 TICK_DB_NAME)

import jqdatasdk as jq

# 加载配置
config = open('config.json')
setting = json.load(config)

mc = MongoClient()  # Mongo连接
dbMinute = mc[MINUTE_DB_NAME]  # 数据库
# dbDaily = mc[DAILY_DB_NAME]
# dbTick = mc[TICK_DB_NAME]

USERNAME = setting['Username']
PASSWORD = setting['Password']
jq.auth(USERNAME, PASSWORD)

FIELDS = ['open', 'high', 'low', 'close', 'volume']


# ----------------------------------------------------------------------
def generateVtBar(row, symbol):
    """生成K线"""
    bar = VtBarData()

    bar.symbol = symbol
    bar.vtSymbol = symbol
    bar.open = row['open']
    bar.high = row['high']
    bar.low = row['low']
    bar.close = row['close']
    bar.volume = row['volume']
    bardatetime = row.name
    bar.date = bardatetime.strftime("%Y%m%d")

    bar.time = bardatetime.strftime("%H%M%S")
    # 将bar的时间改成提前一分钟
    hour = bar.time[0:2]
    minute = bar.time[2:4]
    sec = bar.time[4:6]
    if minute == "00":
        minute = "59"

        h = int(hour)
        if h == 0:
            h = 24

        hour = str(h - 1).rjust(2, '0')
    else:
        minute = str(int(minute) - 1).rjust(2, '0')
    bar.time = hour + minute + sec

    bar.datetime = datetime.strptime(' '.join([bar.date, bar.time]), '%Y%m%d %H%M%S')
    return bar


# ----------------------------------------------------------------------
def jqdownloadMinuteBarBySymbol(symbol,startDate,endDate):
    """下载某一合约的分钟线数据"""
    start = time()

    cl = dbMinute[symbol]
    cl.ensure_index([('datetime', ASCENDING)], unique=True)  # 添加索引

    df = jq.get_price(setting[symbol],start_date = startDate,end_date = endDate, frequency='1m', fields=FIELDS,skip_paused = True)
    for ix, row in df.iterrows():
        bar = generateVtBar(row, symbol)
        d = bar.__dict__
        flt = {'datetime': bar.datetime}
        cl.replace_one(flt, d, True)

    end = time()
    cost = (end - start) * 1000

    print(u'合约%s的分钟K线数据下载完成%s - %s，耗时%s毫秒' % (symbol, df.index[0], df.index[-1], cost))
    print(jq.get_query_count())

def jqdownloadMappingExcel(exportpath = "C:\Project\\"):
    getfuture = jq.get_all_securities(types=['futures'], date=None)
    getfuture.to_excel(
                    exportpath + "Mapping" + str(date.today())  + ".xls",
                    index=True, header=True)
if __name__ == '__main__':
    jqdownloadMappingExcel()
    #下载主力合约
    jqdownloadMinuteBarBySymbol('rb0000', '2018-1-1', '2019-5-1')
    #下载单个品种
    jqdownloadMinuteBarBySymbol('zn1807','2018-6-2','2018-6-8')
