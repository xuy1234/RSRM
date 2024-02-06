from typing import List
from model.exps.exp_tree_node import TreeNode
import abc


class TreeBase(abc.ABC):
    def __init__(self):
        super().__init__()
        self.root = None
        self.node_stack: List[TreeNode] = []
        self.const_num = 0
        self.token_num = 0
        self.triangle_count_tol = 0
        self.max_depth = 0

    @property
    def token_list_pre(self) -> list:
        """
        :return:先序遍历结果
        """
        if not self.root: return []
        return self.root.traverse()

    def current_state(self):
        """
        :return: 语法树转换为token序列的结果
        """
        return [] if not self.root else self.root.traverse()

    def depth(self):
        """
        :return: 树的当前深度
        """
        if not self.node_stack: return 0
        return self.node_stack[self.first_list_pos()].depth

    def trim(self):
        while self.node_stack and self.node_stack[self.first_list_pos()].is_full():
            self.max_depth = max(self.max_depth, self.node_stack[self.first_list_pos()].depth)
            self.node_stack.pop(self.first_list_pos())

    def is_full(self):
        self.trim()
        return self.root and not self.node_stack

    def get_exp(self):
        """
        :return: 语法树转换为表达式的结果
        """
        if not self.root: return ""
        return str(self.root.exp_weak())

    @abc.abstractmethod
    def first_list_pos(self) -> int:
        pass

    def add_exp(self, exp):
        """
        层序添加新的节点
        :param exp:
        """
        if self.is_full():
            raise RuntimeError(f"{self.token_list_pre} {exp.type_name}")
        self.trim()
        self.token_num += 1
        if exp.type_name == "C":
            self.const_num += 1
            node = TreeNode(exp, self.const_num)
        else:
            node = TreeNode(exp, None)
        if not self.root:
            self.root = node
            self.node_stack.append(node)
            return
        self.triangle_count_tol += node.triangle_count
        self.node_stack[self.first_list_pos()].add_child(node)
        self.node_stack.append(node)

    def cal(self, X, Y, C):
        return self.root.cal(X, Y, C)

    @property
    def head_token(self):
        self.trim()
        if not self.root: return ""
        return self.node_stack[self.first_list_pos()].exp.type_name

    def traverse(self):
        lst = []
        self.root.traverse_with_list(lst=lst)
        return lst

    @property
    def triangle_count(self):
        self.trim()
        if not self.node_stack: return 0
        return self.node_stack[self.first_list_pos()].triangle_count


class FloorTree(TreeBase):
    """
    中序添加元素的语法树，用于MCTS部分
    """

    def first_list_pos(self) -> int:
        return 0


class PreTree(TreeBase):
    """
    前序添加元素的语法树，用于遗传算法
    """

    def first_list_pos(self) -> int:
        return -1
