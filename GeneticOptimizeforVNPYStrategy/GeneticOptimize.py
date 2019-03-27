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

def parameter_generate():
    '''
    根据设置的起始值，终止值和步进，随机生成待优化的策略参数
    '''
    parameter_list = []
    timerange = [1,2,3,4,5,10,15,20]
    p1 = random.randrange(20,55,5)      #入场窗口
    p2 = random.randrange(2,12,1)      #出场窗口
    p3 = random.randrange(20,55,5)      #基于ATR窗口止损窗
    p4 = random.randrange(2,12,1)      #出场窗口
    p5 = random.randrange(25,55,5)     #基于ATR的动态调仓
    p6 = random.randrange(3,12,1)
    p7 = random.choice(timerange)

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
    # 创建回测引擎对象
    engine = BacktestingEngine()
    # 设置回测使用的数据
    engine.setBacktestingMode(engine.BAR_MODE)      # 设置引擎的回测模式为K线
    engine.setDatabase("VnTrader_1Min_Db", 'rb1905')  # 设置使用的历史数据库
    engine.setStartDate('20181001')                 # 设置回测用的数据起始日期
    engine.setEndDate('20190220')                   # 设置回测用的数据起始日期

    # 配置回测引擎参数
    engine.setSlippage(1)
    engine.setRate(1/100)
    engine.setSize(10)
    engine.setPriceTick(1)
    engine.setCapital(1000000)

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

    returnDrawdownRatio = round(backresult['returnDrawdownRatio'],2)  #收益回撤比
    sharpeRatio= round(backresult['sharpeRatio'],2)                   #夏普比率
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
    LAMBDA = 160  # 设置每一代产生的子女数
    pop = toolbox.population(400)  # 设置族群里面的个体数量
    CXPB, MUTPB, NGEN = 0.5, 0.0, 60  # 分别为种群内部个体的交叉概率、变异概率、产生种群代数
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
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    toolbox.register("map", pool.map)
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
    MU = 40  # 设置每一代选择的个体数
    LAMBDA = 160  # 设置每一代产生的子女数
    pop = toolbox.population(400)  # 设置族群里面的个体数量
    CXPB, MUTPB, NGEN = 0.5, 0.00, 60  # 分别为种群内部个体的交叉概率、变异概率、产生种群代数
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
    
    
    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(pop, 10)
    for i in best_ind:
        print("best_ind",i)
        print("best_value",i.fitness.values)

