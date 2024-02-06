from typing import Tuple

from sympy import expand, sympify

from model.config import Config
from model.exps.calculator import prune_poly_c
from model.exps.expTree import PreTree
from model.rl.agent import Agent


def get_expresssion_and_reward(game: Agent, tokens: Tuple[str], config_s: Config):
    """
    计算子序列的完整表达式和reward
    :param config_s: config file
    :param tokens: 需要计算结果子序列
    :return:
    """
    exp = PreTree()
    for token in tokens: exp.add_exp(config_s.exp_dict[token])
    symbols = exp.get_exp()
    if symbols.count('C'): symbols = prune_poly_c(symbols)
    return tokens, game.reward(tree=exp, reward=False), str(expand(sympify(symbols)))


def copy_game(game: Agent, action_set):
    """
    用于复制语法树
    :param game: 需要复制的语法树
    :param action_set: 当前语法树的token序列
    :return: 跟game参数一样的语法树
    """
    ans = Agent(
        config_s=game.config_s
    )
    ans.expressions = game.get_exps_full()
    for token in action_set:
        ans.add_token(token)
    if game.exp_last:
        ans.exp_last = game.exp_last
    return ans
