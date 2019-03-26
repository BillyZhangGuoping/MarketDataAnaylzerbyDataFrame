#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
https://deap.readthedocs.io/en/master/overview.html
"""

# Types
from deap import base, creator
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialization
import random
from deap import tools

IND_SIZE = 10  # 种群数

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
# 调用randon.random为每一个基因编码编码创建 随机初始值 也就是范围[0,1]
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
import multiprocessing

pool = multiprocessing.Pool()
toolbox.register("map", pool.map)

# Operators
# difine evaluate function
# Note that a comma is a must
def evaluate(individual):
    return sum(individual),


# use tools in deap to creat our application
toolbox.register("mate", tools.cxTwoPoint) # mate:交叉
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1) # mutate : 变异
toolbox.register("select", tools.selTournament, tournsize=3) # select : 选择保留的最佳个体
toolbox.register("evaluate", evaluate)  # commit our evaluate


# Algorithms
def main():
    pop = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    # Evaluate the entire population
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = toolbox.map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring


    return pop


if __name__ == "__main__":
    # t1 = time.clock()
    pop = main()
    best_ind = tools.selBest(pop, 3)
    for i in best_ind:
        print("best_ind",i)
        print("best_value",i.fitness.values)

    # t2 = time.clock()

    # print(t2-t1)