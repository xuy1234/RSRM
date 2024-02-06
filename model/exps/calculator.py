import math
import warnings
from typing import Tuple

import numpy as np
import sympy as sp
from scipy.optimize import minimize

from model.config import Config
from model.exps.utils import time_limit, FinishException


def process_symbol_with_C(symbols: str, c: np.ndarray) -> str:
    """
    将参数占位符替换成真实参数
    :param symbols: 表达式
    :param c: 参数
    :return: 转换后的表达式
    """
    for idx, val in enumerate(c):
        symbols = symbols.replace(f"C{idx + 1}", str(val))
    return symbols


def cal_expression_single(symbols: str, x: np.ndarray, t: np.ndarray, c: np.ndarray) -> float:
    """
    计算一次表达式求值并计算误差rmse
    :param symbols: 目标表达式
    :param x: 自变量
    :param t: 结果
    :param c: 参数
    :return: RMSE of function or 1e999 if error occurs
    """
    from numpy import inf, seterr
    zoo = inf
    seterr(all="ignore")
    I = complex(0, 1)
    for idx, val in enumerate(x):
        locals()[f'X{idx + 1}'] = val
    with warnings.catch_warnings(record=False) as caught_warnings:
        try:
            target = process_symbol_with_C(symbols, c)
            cal = eval(target)
            ans = float(np.linalg.norm(cal - t, 1) ** 2 / t.shape[0])
            if math.isinf(ans) or math.isnan(ans) or caught_warnings:
                return 1e999
        except OverflowError:
            return 1e999
        except ValueError:
            return 1e999
        except NameError as e:
            return 1e999
        except ArithmeticError:
            return 1e999
    return ans


def prune_poly_c(eq: str) -> str:
    '''
    将复杂参数替换为简单参数
    '''
    for i in range(5):
        eq_l = eq
        c_poly = ['C**' + str(i) + ".5" for i in range(1, 4)]
        c_poly += ['C**' + str(i) + ".25" for i in range(1, 4)]
        c_poly += ['C**' + str(i) for i in range(1, 4)]
        c_poly += [' ' + str(i) + "*C" for i in range(1, 4)]
        for c in c_poly:
            if c in eq: eq = eq.replace(c, 'C')
    for _ in range(5):
        for _ in range(5):
            eq = eq.replace('arcsin(C)', 'C')
            eq = eq.replace('arccos(C)', 'C')
            eq = eq.replace('sin(C)', 'C')
            eq = eq.replace('cos(C)', 'C')
            eq = eq.replace('sqrt(C)', 'C')
            eq = eq.replace('log(C)', 'C')
            eq = eq.replace('exp(C)', 'C')
        eq = eq.replace('C*C', 'C')
        eq = str(sp.sympify(eq))
        if eq == eq_l: break
    return eq


def cal_one(symbols: str, x: np.ndarray, t: np.ndarray, config_s: Config) -> Tuple[float, str]:
    """
    计算表达式求值自动寻找参数并计算误差rmse
    :param symbols: 目标表达式
    :param x: 自变量
    :param t: 结果
    :return: 误差
    """
    symbols = str(sp.sympify(symbols))
    if symbols.count('zoo') or symbols.count('nan'):
        return 1e999, symbols
    c_len = symbols.count('C')
    if c_len == 0:
        return cal_expression_single(symbols, x, t, []), symbols
    try:
        symbols = prune_poly_c(symbols)
    except Exception as e:
        return 1e999, symbols
    c_len = symbols.count('C')
    if c_len == 0:
        return cal_expression_single(symbols, x, t, []), symbols
    symbols = symbols.replace('C', 'PPP')
    for i in range(1, c_len + 1):
        symbols = symbols.replace('PPP', f'C{i}', 1)
    if config_s.const_optimize:
        x0 = np.random.randn(c_len)
        if cal_expression_single(symbols, x, t, x0) > 1e900:
            return 1e999, process_symbol_with_C(symbols, x0)
        x_ans = minimize(lambda c: cal_expression_single(symbols, x, t, c),
                         x0=x0, method='Powell', tol=1e-6, options={'maxiter': 10})
        x0 = x_ans.x
    else:
        x0 = np.ones(c_len)
    val = cal_expression_single(symbols, x, t, x0)

    return val, process_symbol_with_C(symbols, x0)


def cal_expression(symbols: str, config_s: Config, t_limit=1) -> float:
    warnings.filterwarnings('ignore')
    config_s.symbol_tol_num += 1
    try:
        with time_limit(t_limit):
            v, s = cal_one(symbols, config_s.x, config_s.t, config_s)
            if v > config_s.best_exp[1]:
                return v
            v_, s_ = cal_one(s, config_s.x_, config_s.t_, config_s)
            if config_s.best_exp[1] > 1e-10 + v_ + v:
                config_s.best_exp = s_, v_ + v
            if v_ + v <= config_s.reward_end_threshold:
                raise FinishException
            return v
    except TimeoutError:
        pass
    except RuntimeError:
        pass
    return 1e999
