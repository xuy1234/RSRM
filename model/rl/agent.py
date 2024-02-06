import copy
import random
from collections import defaultdict

from model.config import Config
from model.exps.expTree import FloorTree
from model.exps.calculator import cal_expression


class Agent:
    def __init__(self, config_s: Config):
        self.config_s = config_s
        self.expression_dict = config_s.exp_dict
        self.max_parameter = config_s.mcts.max_const
        self.generate_dict = defaultdict(int)
        self.max_height = config_s.mcts.max_height
        self.max_token = config_s.mcts.max_token
        self.discount = config_s.mcts.token_discount
        self.tree = FloorTree()
        self.expressions = []
        self.exp_last = ""

    def get_exps_full(self, num=None):
        """
        获取效果最好的num个表达式和value
        """
        if num is None:
            num = self.config_s.mcts.max_exp_num
        ans = sorted(self.expressions, key=lambda x: -x[0])[:num]
        self.expressions = []
        return ans

    def get_exps(self, num=None):
        """
        获取效果最好的num个表达式
        """
        if num is None:
            num = self.config_s.mcts.max_exp_num
        ans = sorted(self.expressions, key=lambda x: -x[0])[:num]
        tol = [i[1] for i in ans]
        self.expressions = []
        return tol

    def reward(self, tree=None, reward=True):
        """
        计算语法树当前表达式reward
        """
        try:
            if not tree:
                tree = self.tree
            if not tree.is_full():
                raise TimeoutError
            pre = tree.token_list_pre
            l = len(pre)
            if l <= 5:
                return 0 if reward else 1e999
            symbols = tree.get_exp()
            if self.exp_last:
                symbols = f"{self.exp_last}({symbols})"
            ans_tol = cal_expression(symbols, self.config_s)
            val = self.discount ** l / (1 + ans_tol)
            if pre not in [i[1] for i in self.expressions]:
                self.expressions.append((ans_tol, pre))
        except TimeoutError:
            val = 0
            ans_tol = 1e999
        if reward: return val
        return ans_tol

    def reset(self):
        """
        重置语法树
        """
        self.clear()
        return [0], self.unavailable()

    def unavailable(self):
        """
        计算当前不可选的token集合
        """
        self.tree.trim()
        exps = []
        ans = []
        if self.tree.depth() > self.max_height:
            return list(range(len(self.expression_dict)))
        if self.tree.triangle_count > 0 or self.tree.depth() <= 0:
            exps.extend(["Cos", "Sin"])
        if self.tree.head_token == "Exp":
            exps.append('Log')
        if self.tree.head_token == "Log":
            exps.append('Exp')
        if self.tree.const_num == self.max_parameter:
            exps.append("C")
        for i, j in self.expression_dict.items():
            if i not in ans and j.type_name in exps:
                ans.append(i)
        return ans

    def predict(self):
        """
        随机填满整棵语法树计算期望reward
        """
        tol = range(len(self.expression_dict))
        tree = copy.deepcopy(self.tree)
        tree, self.tree = self.tree, tree
        while not self.tree.is_full():
            action = random.choice(list(set(tol) - set(self.unavailable())))
            self.add_token(action)
        reward = self.reward()
        self.tree = tree
        return reward

    def change_exp(self, expr):
        """
        修改当前模式
        """
        self.exp_last = expr

    def add_token(self, token):
        """
        添加新的token
        """
        self.tree.add_exp(self.expression_dict[token])

    def step(self, token):
        """
        添加新的token，返回reward，是否结束，不可选的子节点
        """
        self.add_token(token)
        if self.tree.is_full():
            return self.reward(), True, []
        un = self.unavailable()
        if len(un) == len(self.expression_dict):
            return -1, True, []
        else:
            return - 0.1, False, un

    def clear(self):
        """
        清空语法树
        """
        self.tree = FloorTree()
