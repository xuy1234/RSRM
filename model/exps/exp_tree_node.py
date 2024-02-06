class Expression:
    def __init__(self, child: int, func, sym_fun, sym2_func):
        # 参数数目，numpy计算表达式,sympy计算表达式，字符计算表达式
        self.child = child
        self.func = func
        self.type = self.type_name = None
        self.sym_fun = sym_fun
        self.sym2_fun = sym2_func


class TreeNode:
    """
    语法树的节点
    """

    def __init__(self, exp: Expression, val=None):
        self._exp = exp
        self._childs = []
        self._val = val
        self._last_num = exp.child
        self.depth = 1
        self.parent = None
        self.triangle_count = 1 if exp.type_name in ("Cos", "Sin") else 0

    def func_name(self):
        return self._exp.type_name

    def add_child(self, child):
        """
        添加子节点
        """
        self._childs.append(child)
        child.parent = self
        self._last_num -= 1
        child.depth = self.depth + 1
        child.triangle_count += self.triangle_count

    def is_full(self):
        """
        :return: 是否已满
        """
        return not self._last_num

    def traverse(self):
        """
        :return: 递归计算token序列
        """
        ans = [self._exp.type]
        for child in self._childs: ans.extend(child.traverse())
        return ans

    def exp_weak(self):
        """
        :return: 递归计算符号表达式
        """
        if self._exp.child:
            return self._exp.sym2_fun(*[child.exp_weak() for child in self._childs])
        return self._exp.sym2_fun

    def traverse_with_list(self, lst):
        """
        :return: 递归计算子树的符号表达式
        """
        ans = [self._exp.type]
        for child in self._childs: ans.extend(child.traverse_with_list(ans))
        if 1 < len(ans) <= 10: lst.append(ans)
        return ans

    @property
    def exp(self):
        return self._exp
