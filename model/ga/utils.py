import numpy as np
from deap import gp


def DEAP_to_tokens(individual):
    """
    Convert individual to tokens.

    Parameters
    ----------

    individual : gp.PrimitiveTree
        The DEAP individual.

    Returns
    -------

    tokens : np.array
        The tokens corresponding to the individual.
    """

    tokens = np.array([i.name[3:] for i in individual], dtype=np.int32)
    return tokens


def tokens_to_DEAP(tokens, pset):
    """
    Convert DSO tokens into DEAP individual.

    Parameters
    ----------
    tokens : np.ndarray
        Tokens corresponding to the individual.

    pset : gp.PrimitiveSet
        Primitive set upon which to build the individual.

    Returns
    _______
    individual : gp.PrimitiveTree
        The DEAP individual.
    """
    plist = [pset.mapping[f"exp{t}"] for t in tokens]
    individual = gp.PrimitiveTree(plist)
    return individual


def multi_mutate(individual, expr, pset):
    """Randomly select one of four types of mutation."""
    v = np.random.randint(0, 4)
    if v == 0:
        individual = gp.mutUniform(individual, expr, pset)
    elif v == 1:
        individual = gp.mutNodeReplacement(individual, pset)
    elif v == 2:
        individual = gp.mutInsert(individual, pset)
    elif v == 3:
        individual = gp.mutShrink(individual)

    return individual


def pre_to_floor(token_list, expr_dict):
    """
    先序遍历转层序遍历
    :param token_list: 先序遍历
    :param expr_dict: 表达式字典
    """
    son = [[] for _ in token_list]
    stack = []
    for idx, token in enumerate(token_list):
        while stack and len(son[stack[-1][0]]) == expr_dict[stack[-1][1]].child:
            stack.pop()
        if stack:
            son[stack[-1][0]].append(idx)
        stack.append((idx, token))
    ans = []
    queue = [0]
    for i in queue:
        ans.append(token_list[i])
        for s in son[i]:
            queue.append(s)
    return ans


def floor_to_pre(token_list, expr_dict):
    """
    层序遍历转先序遍历
    :param token_list: 层序遍历
    :param expr_dict: 表达式字典
    """
    son = [[] for _ in token_list]
    queue = []
    for idx, token in enumerate(token_list):
        while queue and len(son[queue[0][0]]) == expr_dict[queue[0][1]].child:
            queue.pop(0)
        if queue:
            son[queue[0][0]].append(idx)
        queue.append((idx, token))
    ans = []
    stack = [0]
    while stack:
        i = stack[-1]
        stack.pop()
        ans.append(token_list[i])
        for s in son[i]:
            stack.append(s)
    return ans
