# encoding: UTF-8

"""
展示如何执行参数优化。
"""

from __future__ import division
from __future__ import print_function
from vnpy.trader.app.ctaStrategy.ctaBacktesting import BacktestingEngine, MINUTE_DB_NAME, OptimizationSetting
from vnpy.trader.app.ctaStrategy.strategy.strategyBBIBoll2V import BBIBoll2VStrategy
from vnpy.trader.app.ctaStrategy.strategy.strategyBollChannel import BollChannelStrategy
import random
import numpy as np
from deap import creator, base, tools, algorithms
import multiprocessing
from scoop import futures
import time,datetime


def parameter_generate():
    '''
    根据设置的起始值，终止值和步进，随机生成待优化的策略参数
    '''
    parameter_list = []
<<<<<<< HEAD
    timerange = [3,5,10,15,20]

    p1 = random.randrange(10,55,1)      #入场窗口
    p2 = random.randrange(1,15,1)      #出场窗口
    p3 = random.randrange(20,55,1)      #基于ATR窗口止损窗
    p4 = random.randrange(1,12,1)      #出场窗口
    p5 = random.randrange(20,70,1)     #基于ATR的动态调仓
    p6 = random.randrange(1,30,1)
    p7 = random.choice(timerange)
=======
    timerange = [1,2,3,4,5,10,15,20]
    p1 = random.randrange(25,40,5)      #入场窗口
    p2 = random.randrange(5,8,1)      #出场窗口
    p3 = random.randrange(25,40,5)      #基于ATR窗口止损窗
    # p4 = random.randrange(25,40,5)     #基于ATR的动态调仓
    p4 = random.choice(timerange)
>>>>>>> cf6e3e9f0ec8b9cca0840f1d2a24c0ce17d24064

    parameter_list.append(p1)
    parameter_list.append(p2)
    parameter_list.append(p3)
    parameter_list.append(p4)
    parameter_list.append(p5)
    parameter_list.append(p6)
    parameter_list.append(p7)

    return parameter_list

def object_func(strategy_avg):
    """
    本函数为优化目标函数，根据随机生成的策略参数，运行回测后自动返回2个结果指标：收益回撤比和夏普比率
    """


    a1=(2018,1,8,0,0,0,0,0,0)              #设置开始日期时间元组（1976-01-01 00：00：00）
    a2=(2019,1,1,23,59,59,0,0,0)    #设置结束日期时间元组（1990-12-31 23：59：59）

    start=time.mktime(a1)    #生成开始时间戳
    end=time.mktime(a2)      #生成结束时间戳
    t1=random.randint(start,end)    #在开始和结束时间戳中随机取出一个
    t2 =(t1+10000000)         #将时间戳生成时间元组
    date_touple1=time.localtime(t1)          #将时间戳生成时间元组
    date_touple2=time.localtime(t2)          #将时间戳生成时间元组
    date_s=time.strftime("%Y%m%d",date_touple1)  #将时间元组转成格式化字符串（1976-05-21）
    date_e=time.strftime("%Y%m%d",date_touple2) 
    #随机生成10个日期字符串

    # 创建回测引擎对象
    engine = BacktestingEngine()
    # 设置回测使用的数据
    engine.setBacktestingMode(engine.BAR_MODE)      # 设置引擎的回测模式为K线
<<<<<<< HEAD
    engine.setDatabase("VnTrader_1Min_Db", 'rb.hot')  # 设置使用的历史数据库
    engine.setStartDate(date_s)                 # 设置回测用的数据起始日期
    engine.setEndDate(date_e)                   # 设置回测用的数据起始日期
=======
    engine.setDatabase("VnTrader_1Min_Db", 'rb0000')  # 设置使用的历史数据库
    engine.setStartDate('20170301')                 # 设置回测用的数据起始日期
    engine.setEndDate('20170501')                   # 设置回测用的数据起始日期
>>>>>>> cf6e3e9f0ec8b9cca0840f1d2a24c0ce17d24064

    # 配置回测引擎参数
    engine.setSlippage(1)
    engine.setRate(1/100)
    engine.setSize(10)
    engine.setPriceTick(1)
    engine.setCapital(10000)

    setting = {
                'bollWindow': strategy_avg[0],       #布林带窗口
                'bollDev': strategy_avg[1],        #布林带通道阈值
                'bbibollWindow':strategy_avg[2],
                'bbibollDev':strategy_avg[3],
                'slMultiplier':strategy_avg[4]/10.0,
                'profitRate':strategy_avg[5]/1000.0,
                'barMins':strategy_avg[6],
               }    #ATR窗口

    #加载策略
    engine.initStrategy(BBIBoll2VStrategy, setting)
    # 运行回测，返回指定的结果指标
    engine.runBacktesting()          # 运行回测
    #逐日回测
    engine.calculateDailyResult()
    backresult = engine.calculateDailyStatistics()[1]

    returnDrawdownRatio = round(backresult['annualizedReturn'],3)  #收益回撤比
    sharpeRatio= round(backresult['sharpeRatio'],3)                   #夏普比率                 #夏普比率
    return returnDrawdownRatio , sharpeRatio


# 设置优化方向：最大化收益回撤比，最大化夏普比率
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # 1.0 求最大值；-1.0 求最小值
creator.create("Individual", list, fitness=creator.FitnessMulti)


def optimize():
    """"""
    toolbox = base.Toolbox()  # Toolbox是deap库内置的工具箱，里面包含遗传算法中所用到的各种函数

    # 初始化
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     parameter_generate)  # 注册个体：随机生成的策略参数parameter_generate()
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)  # 注册种群：个体形成种群
    toolbox.register("mate", tools.cxTwoPoint)  # 注册交叉：两点交叉
    toolbox.register("mutate", tools.mutUniformInt, low=4, up=40, indpb=0.6)  # 注册变异：随机生成一定区间内的整数
    toolbox.register("evaluate", object_func)  # 注册评估：优化目标函数object_func()
    toolbox.register("select", tools.selNSGA2)  # 注册选择:NSGA-II(带精英策略的非支配排序的遗传算法)

    # 遗传算法参数设置
    MU = 40  # 设置每一代选择的个体数
    LAMBDA = 260  # 设置每一代产生的子女数
    pop = toolbox.population(400)  # 设置族群里面的个体数量
<<<<<<< HEAD
    CXPB, MUTPB, NGEN = 0.5, 0.0, 100  # 分别为种群内部个体的交叉概率、变异概率、产生种群代数
=======
    CXPB, MUTPB, NGEN = 0.5, 0.00, 40  # 分别为种群内部个体的交叉概率、变异概率、产生种群代数
>>>>>>> cf6e3e9f0ec8b9cca0840f1d2a24c0ce17d24064
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
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)  # esMuPlusLambda是一种基于(μ+λ)选择策略的多目标优化分段遗传算法

    return pop

if __name__ == "__main__":

    toolbox = base.Toolbox()  # Toolbox是deap库内置的工具箱，里面包含遗传算法中所用到的各种函数
    pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count()-2))
    toolbox.register("map", pool.map)
    # toolbox.register("map", futures.map)
    # 初始化
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     parameter_generate)  # 注册个体：随机生成的策略参数parameter_generate()
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)  # 注册种群：个体形成种群
    toolbox.register("mate", tools.cxTwoPoint)  # 注册交叉：两点交叉
    toolbox.register("mutate", tools.mutUniformInt, low=2, up=60, indpb=0.6)  # 注册变异：随机生成一定区间内的整数
    toolbox.register("evaluate", object_func)  # 注册评估：优化目标函数object_func()
    toolbox.register("select", tools.selNSGA2)  # 注册选择:NSGA-II(带精英策略的非支配排序的遗传算法)

    # 遗传算法参数设置
    MU = 1000  # 设置每一代选择的个体数
    LAMBDA = 500  # 设置每一代产生的子女数
    pop = toolbox.population(3000)  # 设置族群里面的个体数量
    CXPB, MUTPB, NGEN = 0.5, 0.00, 300  # 分别为种群内部个体的交叉概率、变异概率、产生种群代数
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
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof,verbose=True)  # esMuPlusLambda是一种基于(μ+λ)选择策略的多目标优化分段遗传算法
    
    
    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(pop, 10)
    for i in best_ind:
        print("best_ind",i)
        print("best_value",i.fitness.values)

