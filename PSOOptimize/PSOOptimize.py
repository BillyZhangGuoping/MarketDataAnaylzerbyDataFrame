# encoding: UTF-8

"""
展示如何实现PSO粒子群优化VNPY策略参数
"""
from __future__ import division
from __future__ import print_function
import operator
import random
import numpy
from deap import base
from deap import creator
from deap import tools
from vnpy.trader.app.ctaStrategy.ctaBacktesting import BacktestingEngine, MINUTE_DB_NAME, OptimizationSetting
from vnpy.trader.app.ctaStrategy.strategy.strategyBollChannel import BollChannelStrategy

def object_func(strategy_avgTuple):
    """
    本函数为优化目标函数，根据随机生成的策略参数，运行回测后自动返回1个结果指标：夏普比率
    这个是直接赋值GenticOptimize2V的
    """
    strategy_avg = strategy_avgTuple
    paraSet = strategy_avgTuple.parameterPackage
    symbol = paraSet["symbol"]
    strategy = paraSet["strategy"]


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

        sharpeRatio = round(backresult['sharpeRatio'], 3)
        totalResultCount = round(backresult['totalResult'],3)

    except Exception, e:
        print("Error: %s, %s" %(str(Exception),str(e)))
        sharpeRatio = 0

    return sharpeRatio,




class PSOOptimize(object):
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
            "symbol":self.symbol
        }

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
                   pmin = list, pmax = list,smin=list, smax=list, parameterPackage = dict, best=None)

    def particle_generate(self):
        """
        生产particle粒子，根据传入设置的起始值，终止值随机生成位置,和位置最大最小值，根据步进生成速度,和速度最大最小值
        """
        position_list = []
        speed_list = []
        pmin_list = []
        pmax_list = []
        smin_list = []
        smax_list = []
        for key, value in self.parameterlist.items():
            if isinstance(value, tuple):
                if len(value) == 3:
                    position_list.append({key:random.randrange(value[0], value[1])})
                    pmin_list.append(value[0])
                    pmax_list.append(value[1])
                    speed_list.append(random.uniform(-value[2], value[2]))
                    smin_list.append(-value[2])
                    smax_list.append(value[2])
                else:
                    print("Paramerte list incorrect")
            else:
                print("Paramerte list incorrect")

        particle = creator.Particle(position_list)
        particle.speed = speed_list
        particle.pmin = pmin_list
        particle.pmax = pmax_list
        particle.smin = smin_list
        particle.smax = smax_list
        particle.parameterPackage = self.parameterPackage
        return particle

    def updateParticle(self,part, best, phi1, phi2):
        """
        根据粒子群最佳位置best，去更新粒子part的位置和速度，
        速度公式：
        v[] = v[] + c1 * rand() * (pbest[] - present[]) + c2 * rand() * (gbest[] - present[])
        位置公式：
        present[] = persent[] + v[]
        """
        u1 = (random.uniform(0, phi1) for _ in range(len(part)))
        u2 = (random.uniform(0, phi2) for _ in range(len(part)))

        v_u1 = map(operator.mul, u1, map(self.sub, part.best, part))# c1 * rand() * (pbest[] - present[])
        v_u2 = map(operator.mul, u2, map(self.sub, best, part)) # c2 * rand() * (gbest[] - present[])
        part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))

        for i, speed in enumerate(part.speed):
            if speed < part.smin:
                part.speed[i] = part.smin[i]
            elif speed > part.smax:
                part.speed[i] = part.smax[i]
        #返回现在位置，如果原来是整数，返回也是整数，否则这是浮点数; 如果超过上下限，用上下限数值
        for i,item in enumerate(part):
            if isinstance(item.values()[0],int):
                positionV = round(item.values()[0] + part.speed[i])
            else:
                positionV = item.values()[0] + part.speed[i]
            if positionV <= part.pmin[i] :
                part[i][item.keys()[0]] = part.pmin[i]
            elif positionV >= part.pmax[i]:
                part[i][item.keys()[0]] = part.pmax[i]
            else:
                part[i][item.keys()[0]] = positionV

    def sub(self,a,b):
       return a.values()[0] - b.values()[0]

    def optimize(self):
        toolbox = base.Toolbox()
        toolbox.register("particle", self.particle_generate)
        toolbox.register("population", tools.initRepeat, list, toolbox.particle)
        toolbox.register("update", self.updateParticle, phi1=2.0, phi2=2.0)
        toolbox.register("evaluate", object_func)

        pop = toolbox.population(n=20) #粒子群有5个粒子
        GEN = 10 #更新一千次
        best = None
        for g in range(GEN):
            for part in pop: #每次更新，计算粒子群中最优参数，并把最优值写入best
                part.fitness.values = toolbox.evaluate(part)
                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
            for part in pop: #更新粒子位置
                toolbox.update(part, best)

            # Gather all the fitnesses in one list and print the stats
        return pop, best


if __name__ == "__main__":

    Strategy = BollChannelStrategy

    Symbol = {
            "vtSymbol": 'rb1905',
            "StartDate": "20181001",
            "EndDate": "20190101",
            "Slippage": 2,
            "Size": 10,
            "Rate": 2 / 10000,
            "Capital": 10000
        }

    Parameterlist = {
        'bollWindow':(5,80,2),
        'bollDev':(2.0,8.0,1.0),
        'cciWindow':(5,80,2),
        'atrWindow': (5,80,2),
        'slMultiplier':(2.0,8.0,1.0),
    }

    parameterPackage = {
        "symbol": Symbol,
        "parameterlist": Parameterlist,
        "strategy": Strategy
    }
    PSO = PSOOptimize(Strategy, Symbol, Parameterlist)
    pop,best = PSO.optimize()
    print ("best para: %s, result:%s" %(best,best.fitness.values))
    print(pop[:20])

    print("-- End of (successful) %s evolution --", Symbol["vtSymbol"])
