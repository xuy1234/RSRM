import random


class TreeNode:
    def __init__(self, father):
        self.father = father
        self.sons = None
        self.max_v = -1e9
        self.times = 0

    def update(self, v):
        now = self
        while now:
            now.max_v = max(now.max_v, v)
            now = now.father

    def expand(self, sons):
        self.sons = {
            son: TreeNode(self) for son in sons
        }

    def choose_max(self):
        return max(self.sons.items(), key=lambda x: x[1].max_v)[0]

    def choose_zero(self, n0):
        st = [i[0] for i in self.sons.items() if i[1].times < n0]
        if not st: return -1
        return random.choice(st)


class SearchTree:
    def __init__(self):
        self.root = TreeNode(None)
        self.now = self.root
        self.n0 = 10

    def empty(self):
        return self.now.sons is None

    def expand(self, sons):
        self.now.expand(sons)

    def choose(self):
        return self.now.choose_max()

    def update(self, v):
        self.now.update(v)

    def clear(self):
        self.now = self.root

    def step(self, a):
        self.now = self.now.sons[a]
        self.now.times += 1

    def get_status(self):
        return self.now

    def set_status(self, now):
        self.now = now

    def choose_zero(self):
        return self.now.choose_zero(self.n0)
