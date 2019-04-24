# encoding: UTF-8

"""
展示如何执行参数优化。
"""

from __future__ import division
from __future__ import print_function
from vnpy.trader.app.ctaStrategy.ctaBacktesting import BacktestingEngine, MINUTE_DB_NAME, OptimizationSetting
from vnpy.trader.app.ctaStrategy.strategy.strategyBBIBoll2V import BBIBoll2VStrategy
from CtaBacktesting.DraftStrage.strategyBBIBOLLING import BBIBollINGStrategy
from vnpy.trader.app.ctaStrategy.strategy.strategyBollChannel import BollChannelStrategy
from vnpy.trader.app.ctaStrategy.strategy.strategyHCLBOLL import HCLBOLLStrategy
import random
import numpy as np
from deap import creator, base, tools, algorithms
import multiprocessing
import time, datetime
import pandas as pd


def object_func(strategy_avgTuple):
    """
    本函数为优化目标函数，根据随机生成的策略参数，运行回测后自动返回2个结果指标：收益回撤比和夏普比率
    """

    strategy_avg = strategy_avgTuple[0]
    paraSet = strategy_avgTuple[1]
    symbol = paraSet["symbol"]
    strategy = paraSet["strategy"]

    # import time, random
    # a1 = (2018, 5, 30, 0, 0, 0, 0, 0, 0)  # 设置开始日期时间元组（1976-01-01 00：00：00）
    # a2 = (2019, 1, 15, 23, 59, 59, 0, 0, 0)  # 设置结束日期时间元组（1990-12-31 23：59：59）

    # start = time.mktime(a1)  # 生成开始时间戳
    # end = time.mktime(a2)  # 生成结束时间戳
    # random.seed(seed)
    # t1 = random.randint(start, end)  # 在开始和结束时间戳中随机取出一个
    # t2 = (t1 + 10000000)  # 将时间戳生成时间元组
    # date_touple1 = time.localtime(t1)  # 将时间戳生成时间元组
    # date_touple2 = time.localtime(t2)  # 将时间戳生成时间元组
    # date_s = time.strftime("%Y%m%d", date_touple1)  # 将时间元组转成格式化字符串（1976-05-21）
    # date_e = time.strftime("%Y%m%d", date_touple2)

    # 创建回测引擎对象
    engine = BacktestingEngine()
    # 设置回测使用的数据
    engine.setBacktestingMode(engine.BAR_MODE)  # 设置引擎的回测模式为K线
    engine.setDatabase("VnTrader_1Min_Db", symbol["vtSymbol"])  # 设置使用的历史数据库
    engine.setStartDate(symbol["StartDate"])  # 设置回测用的数据起始日期
    engine.setEndDate(symbol["EndDate"])  # 设置回测用的数据起始日期

    # 配置回测引擎参数
    engine.setSlippage(symbol["Slippage"])  # 1跳
    engine.setRate(symbol["Rate"])  # 佣金大小
    engine.setSize(symbol["Size"])  # 合约大小
    engine.setPriceTick(symbol["Slippage"])  # 最小价格变动
    engine.setCapital(symbol["Capital"])

    setting = {}
    for item in range(len(strategy_avg)):
        setting.update(strategy_avg[item])

    engine.clearBacktestingResult()
    # 加载策略

    engine.initStrategy(strategy, setting)
    # 运行回测，返回指定的结果指标
    engine.runBacktesting()  # 运行回测
    # 逐日回测
    # engine.calculateDailyResult()
    backresult = engine.calculateBacktestingResult()

    try:
        capital = round(backresult['capital'], 3)  # 收益回撤比
        profitLossRatio = round(backresult['profitLossRatio'], 3)  # 夏普比率                 #夏普比率
        sharpeRatio = round(backresult['sharpeRatio'], 3)
        totalResultCount = round(backresult['totalResult'],3)

    except Exception, e:
        print("Error: %s, %s" %(str(Exception),str(e)))
        sharpeRatio = 0
        profitLossRatio = 0  # 收益回撤比
        averageWinning = 0  # 夏普比率                 #夏普比率
        capital = 0
        totalResultCount = 0
    return sharpeRatio, profitLossRatio, totalResultCount


class GeneticOptimizeStrategy(object):
    strategy = None
    symbol = {}
    parameterlist = {}
    parameterPackage = {}

    # ------------------------------------------------------------------------
    def __init__(self, Strategy, Symbollist, Parameterlist):
        self.strategy = Strategy
        self.symbol = Symbollist
        self.parameterlist = Parameterlist
        self.parameterPackage = {
            "strategy":self.strategy,
            "symbol":self.symbol,
        }


    # ------------------------------------------------------------------------
    def parameter_generate(self):
        '''
        根据设置的起始值，终止值和步进，随机生成待优化的策略参数
        '''
        parameter_list = []
        for key, value in self.parameterlist.items():
            if isinstance(value, tuple):
                if len(value) == 3:
                    parameter_list.append({key:random.randrange(value[0], value[1], value[2])})
                elif len(value) == 2:
                    parameter_list.append({key:random.uniform(value[0], value[1])})
            elif isinstance(value, list):
                parameter_list.append({key:random.choice(value)})
            else:
                parameter_list.append({key:value})
        return parameter_list


    def mutArrayGroup(self, individual, parameterlist, indpb):
        size = len(individual)
        paralist = parameterlist()
        for i in xrange(size):
            if random.random() < indpb:
                individual[i] = paralist[i]

        return individual,

    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))  # 1.0 求最大值；-1.0 求最小值
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    def optimize(self,parameterPackage):
        # 设置优化方向：最大化收益回撤比，最大化夏普比率
        toolbox = base.Toolbox()  # Toolbox是deap库内置的工具箱，里面包含遗传算法中所用到的各种函数
        pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count()))
        toolbox.register("map", pool.map)

        # 初始化
        toolbox.register("individual", tools.initIterate, creator.Individual,
                         self.parameter_generate)  # 注册个体：随机生成的策略参数parameter_generate()
        toolbox.register("population", tools.initRepeat, list,
                         toolbox.individual)  # 注册种群：个体形成种群
        toolbox.register("mate", tools.cxTwoPoint)  # 注册交叉：两点交叉
        toolbox.register("mutate", self.mutArrayGroup, parameterlist=self.parameter_generate,
                         indpb=0.4)  # 注册变异：随机生成一定区间内的整数
        toolbox.register("evaluate", object_func)  # 注册评估：优化目标函数object_func()
        toolbox.register("select", tools.selNSGA2)  # 注册选择:NSGA-II(带精英策略的非支配排序的遗传算法)

        # 遗传算法参数设置
        MU = 1200  # 设置每一代选择的个体数
        LAMBDA =1200 # 设置每一代产生的子女数
        pop = toolbox.population(2500)  # 设置族群里面的个体数量
        CXPB, MUTPB, NGEN = 0.4, 0.3, 200 # 分别为种群内部个体的交叉概率、变异概率、产生种群代数
        hof = tools.ParetoFront()  # 解的集合：帕累托前沿(非占优最优集)

        # 解的集合的描述统计信息
        # 集合内平均值，标准差，最小值，最大值可以体现集合的收敛程度
        # 收敛程度低可以增加算法的迭代次数
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        np.set_printoptions(suppress=True)  # 对numpy默认输出的科学计数法转换
        stats.register("mean", np.mean, axis=0)  # 统计目标优化函数结果的平均值
        stats.register("std", np.std, axis=0)  # 统计目标优化函数结果的标准差
        stats.register("min", np.min, axis=0)  # 统计目标优化函数结果的最小值
        stats.register("max", np.max, axis=0)  # 统计目标优化函数结果的最大值
        # 运行算法
        algorithms.eaMuPlusLambda2(parameterPackage, pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                                  halloffame=hof, verbose=True)  # esMuPlusLambda是一种基于(μ+λ)选择策略的多目标优化分段遗传算法

        return pop

    def poptoExcel(self, pop, number = 2500, path = "C:/data/"):
        #按照输入统计数据队列和路径，输出excel，这里不提供新增模式，如果想，可以改
        #dft.to_csv(path,index=False,header=True, mode = 'a')
        path = path +  self.strategy.className + "_" + self.symbol[ "vtSymbol"] + str(datetime.date.today())+ ".xls"
        summayKey = ["StrategyParameter","TestValues"]
        best_ind = tools.selBest(pop, number)
        dft = pd.DataFrame(columns=summayKey)

        for i in range(0,len(best_ind)-1):
            if i == 0:
                # new = pd.DataFrame([{"StrategyParameter":self.complieString(best_ind[i])},{"TestValues":best_ind[i].fitness.values}], index=["0"])
                dft = dft.append([{"StrategyParameter":self.complieString(best_ind[i]),"TestValues":best_ind[i].fitness.values}], ignore_index=True)
            elif str(best_ind[i-1]) == (str(best_ind[i])):
                pass
            else:
                #new = pd.DataFrame({"StrategyParameter":self.complieString(best_ind[i]),"TestValues":best_ind[i].fitness.values}, index=["0"])
                dft = dft.append([{"StrategyParameter":self.complieString(best_ind[i]),"TestValues":best_ind[i].fitness.values}], ignore_index=True)

        dft.to_excel(path,index=False,header=True)
        print("回测统计结果输出到" + path)

    def complieString(self,individual):
        setting = {}
        for item in range(len(individual)):
            setting.update(individual[item])
        return str(setting)



if __name__ == "__main__":
    Strategy = BBIBoll2VStrategy

    SymbolList = [
        {
            "vtSymbol": 'rb1901',
            "StartDate": "20180501",
            "EndDate": "20181101",
            "Slippage": 1,
            "Size": 10,
            "Rate": 2 / 10000,
            "Capital": 10000
        },
        {
            "vtSymbol": 'rb1905',
            "StartDate": "20181101",
            "EndDate": "20190401",
            "Slippage": 1,
            "Size": 10,
            "Rate": 2 / 10000,
            "Capital": 10000
        }
        # {
        #     "vtSymbol": 'bu1812',
        #     "StartDate": "20180501",
        #     "EndDate": "20181201",
        #     "Slippage": 2,
        #     "Size": 10,
        #     "Rate": 2 / 10000,
        #     "Capital": 15000
        # },
        # {
        #     "vtSymbol": 'm1901',
        #     "StartDate": "20180501",
        #     "EndDate": "20181201",
        #     "Slippage": 1,
        #     "Size": 10,
        #     "Rate": 2 / 10000,
        #     "Capital": 10000
        # }
        ]
    Parameterlist = {
        'atrWindow': (10, 30, 1),  # ATR窗口数
        'barMins': 5,
        'slMultiplier': (2.0, 6.0),  # 计算止损距离的乘数
        'profitRate': (0.001, 0.010),
        'bbibollWindow':(30, 50, 1),
        'bbibollDev':(6.5,7.5),
        'CDate':(2,5,1),
        'endsize': (0, 5, 1),
        'endplus': (0, 5, 1),
        'barsize': (0, 5, 1),
        'barplus': (0, 5, 1)
    }
    for Symbol in SymbolList:
        parameterPackage = {
            "symbol": Symbol,
            "parameterlist": Parameterlist,
            "strategy": Strategy
        }
        GE = GeneticOptimizeStrategy(Strategy, Symbol, Parameterlist)
        pop = GE.optimize(parameterPackage)
        GE.poptoExcel(pop)
        del GE, pop
        print("-- End of (successful) %s evolution --", Symbol["vtSymbol"])
