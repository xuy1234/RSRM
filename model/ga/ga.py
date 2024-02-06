import operator
import random

from deap import gp, algorithms
from deap import tools
from deap import creator
from deap import base
from model.exps.exp_tree_node import Expression
from model.ga.agent import Agent as Game_GA
import model.ga.utils as utils

run = False


class GAPipeline:
    def __init__(self, config_s):
        """
        初始化遗传算法
        """
        global run
        if not run:
            run = True  # 初始化遗传工具箱
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        self.exp_dict = config_s.exp_dict
        toolbox = base.Toolbox()
        self.config_s = config_s
        self.agent = Game_GA(toolbox=toolbox, config_s=config_s)
        var_num = sum([exp.child == 0 for num, exp in self.exp_dict.items()])
        pset = gp.PrimitiveSet("MAIN", var_num)
        var_count = 0
        """
        初始化遗传算法tokens
        """
        for num, exp in self.exp_dict.items():
            if not isinstance(exp, Expression): continue
            if exp.child == 0:
                pset.renameArguments(**{f'ARG{var_count}': f"exp{num}"})
                var_count += 1
            else:
                pset.addPrimitive(exp.func, exp.child, name=f"exp{num}")
        """
        初始化遗传算法工具箱
        """
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("evaluate", self.agent.fitness)
        toolbox.register("select", tools.selTournament, tournsize=config_s.gp.tournsize)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", utils.multi_mutate, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config_s.gp.max_height))
        toolbox.decorate("mutate",
                         gp.staticLimit(key=operator.attrgetter("height"), max_value=config_s.gp.max_height))
        self.toolbox = toolbox
        self.pset = pset

    def ga_play(self, pop_init):
        """
        遗传算法运行类
        :param pop_init: 部分初始种群
        :return: 遗传算法结果
        """
        hof = tools.HallOfFame(20)
        pops = pop_init
        pop = self.config_s.gp.pops
        if len(pops) >= pop // 2: pops = random.sample(pops, pop // 2)
        pops = [creator.Individual(utils.tokens_to_DEAP(p, self.pset)) for p in pops]
        pops += self.toolbox.population(n=pop - len(pops))
        pop, log = algorithms.eaSimple(pops, self.toolbox, self.config_s.gp.cxpb, self.config_s.gp.mutpb,
                                       self.config_s.gp.times, halloffame=hof, verbose=False)
        return [utils.DEAP_to_tokens(tokens) for tokens in hof]
