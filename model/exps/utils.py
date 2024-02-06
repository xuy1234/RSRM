import _thread
import math
import threading
from contextlib import contextmanager
from typing import Dict

import numpy as np
import sympy as sp

from model.exps.exp_tree_node import Expression


def get_expression(strs) -> Expression:
    """
    获取对应表达式的expression
    :param strs: 表达式字符串
    :return: 对应的expression
    """
    exp_dict = {
        "Id": Expression(1, np.array, lambda x: x, lambda x: f"{x}"),
        "Add": Expression(2, np.add, lambda x, y: x + y, lambda x, y: f"({x})+({y})"),
        "Sub": Expression(2, np.subtract, lambda x, y: x - y, lambda x, y: f"({x})-({y})"),
        "Mul": Expression(2, np.multiply, lambda x, y: x * y, lambda x, y: f"({x})*({y})"),
        "Div": Expression(2, np.divide, lambda x, y: x / y, lambda x, y: f"({x})/({y})"),
        "Dec": Expression(1, lambda x: x - 1, lambda x: x - 1, lambda x: f"({x})+1"),
        "Inc": Expression(1, lambda x: x + 1, lambda x: x + 1, lambda x: f"({x})-1"),
        "Neg": Expression(1, np.negative, lambda x: -x, lambda x: f"-({x})"),
        "Log": Expression(1, np.log, sp.log, lambda x: f"log({x})"),
        "Sin": Expression(1, np.sin, sp.sin, lambda x: f"sin({x})"),
        "Cos": Expression(1, np.cos, sp.cos, lambda x: f"cos({x})"),
        "Asin": Expression(1, np.arcsin, sp.asin, lambda x: f"arcsin({x})"),
        "Atan": Expression(1, np.arctan, sp.atan, lambda x: f"arctan({x})"),
        "Exp": Expression(1, np.exp, sp.exp, lambda x: f"exp({x})"),
        "Sqrt": Expression(1, np.sqrt, sp.sqrt, lambda x: f"({x})**0.5"),
        "N2": Expression(1, np.square, lambda x: x * x, lambda x: f"({x})**2"),
        "Pi": Expression(0, np.pi, math.pi, 'pi'),
        "One": Expression(0, 1, 1, '1'),
    }
    if strs in exp_dict:
        return exp_dict[strs]
    return Expression(0, None if strs == 'C' else int(strs[1:]), sp.symbols(strs), strs)


def expression_dict(tokens, num_of_var, const) -> Dict[int, Expression]:
    """
    创建表达式字典key是index,value是Expression类 参数数目，numpy计算表达式,sympy计算表达式，字符计算表达式
    :return: 表达式字典
    """

    def generate_expression_dict(expression_list) -> Dict[int, Expression]:
        exp_dict = {}
        for i, expression in enumerate(expression_list):
            exp = get_expression(expression)
            exp.type = i
            exp_dict[i] = exp
            exp.type_name = expression
        return exp_dict

    return generate_expression_dict(
        [f'X{i}' for i in range(1, 1 + num_of_var)] +
        (["C"] if const else []) +
        tokens
    )


class FinishException(Exception):
    """
    得到正确表达式抛出的异常
    """
    pass


@contextmanager
def time_limit(seconds: int, msg=''):
    """
    计时类 在seconds后抛出TimeoutError错误
    """
    if len(threading.enumerate()) >= 1:
        for th in threading.enumerate():
            if th.name.count('Timer') > 0:
                th._stop()
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    try:
        timer.start()
        yield
    except KeyboardInterrupt:
        raise TimeoutError(msg)
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()
