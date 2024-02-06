from model.exps.expTree import PreTree
from model.exps.calculator import cal_expression
import model.ga.utils as utils
class Agent:
    def __init__(self, toolbox, config_s):
        """
        遗传算法实现类
        """
        self.config_s = config_s
        self.expression_dict = config_s.exp_dict
        self.max_parameter = config_s.gp.max_const
        self.discount = config_s.gp.token_discount
        self.toolbox = toolbox
        self.exp_last = ""

    def change_exp(self, expr):
        """
        更换新的模式
        :param expr: 模式
        """
        self.exp_last = expr

    def fitness(self, individual):
        """
        计算individual的适应度
        """
        try:
            tree = PreTree()
            token_list = utils.DEAP_to_tokens(individual)
            if len(token_list) <= 5:
                return 1e999,
            for token in token_list:
                if token not in self.available(tree):
                    return 1e999,
                tree.add_exp(self.expression_dict[token])
            symbols = tree.get_exp()
            if self.exp_last: symbols = f"{self.exp_last}({symbols})"
            ans = cal_expression(symbols, self.config_s)
            val = self.discount ** (-len(individual)) * ans  # 计算适应度 越小越好
            return val,
        except TimeoutError:
            pass
        return 1e999,

    def available(self, tree):
        """
        判断树的可填节点
        :param tree: 语法树
        """
        exps = []
        ans = list(self.expression_dict.keys())
        if tree.triangle_count > 0:
            exps.extend(["Cos", "Sin"])
        if tree.head_token == "Exp":
            exps.append('Log')
        if tree.head_token == "Log":
            exps.append('Exp')
        if tree.const_num == self.max_parameter:
            exps.append("C")
        for i, j in self.expression_dict.items():
            if i in ans and j.type_name in exps:
                ans.remove(i)
        return ans
