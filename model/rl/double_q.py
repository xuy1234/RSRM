import random
import numpy as np

from model.config import Config


class DoubleQLearningNode:
    """
    double q learning节点
    """

    def __init__(self, actions):
        self.children = {}
        self.value_a = np.zeros(actions)
        self.value_b = np.zeros(actions)
        self.actions = actions

    def choose_max(self, dis):
        """
        :param dis: 不使用的子节点index
        :return:最高value节点
        """
        value = self.value_b + self.value_a
        value[dis] = -1
        mx = value == np.max(value)
        action = np.random.choice(range(self.actions), p=mx / np.sum(mx))
        return action

    def choose_softmax(self, dis):
        """
        :param dis: 不使用的子节点index
        :return: 按照softmax value从节点取样
        """
        value = self.value_b + self.value_a
        value[dis] = -1e9
        value -= np.max(value)
        soft = np.exp(value)
        action = np.random.choice(range(self.actions), p=soft / np.sum(soft))
        return action

    def softmax_possibility(self, dis):
        """
        :param dis: 不使用的子节点index
        :return: softmax value
        """
        value = self.value_b + self.value_a
        value -= np.max(value)
        soft = np.exp(value)
        soft[dis] = 0
        soft /= np.sum(soft)
        return soft

    def check_exist(self, action):
        if action not in self.children:
            self.children[action] = DoubleQLearningNode(self.actions)

    def learn(self, gamma, lr, a, r, dead):
        """
        学习
        :param gamma: 衰减率
        :param lr: 学习率
        :param a: 子节点index
        :param r: 奖励
        :param dead: 是否结束
        """
        self.check_exist(a)
        if np.random.uniform() <= 0.5:
            q_predict = self.value_a[a]
            if not dead:
                q_target = r + gamma * np.max(self.children[a].value_b)  # next state is not terminal
            else:
                q_target = r  # next state is terminal
            self.value_a[a] += lr * (q_target - q_predict)  # update
        else:
            q_predict = self.value_b[a]
            if not dead:
                q_target = r + gamma * np.max(self.children[a].value_a)  # next state is not terminal
            else:
                q_target = r  # next state is terminal
            self.value_b[a] += lr * (q_target - q_predict)  # update


class DoubleQLearningTable:
    def __init__(self, actions, config_s: Config):
        self.actions = actions
        self.lr = config_s.mcts.q_learning_rate
        self.gamma = config_s.mcts.q_learning_discount
        self.epsilon = config_s.mcts.q_learning_epsilon
        self.root = DoubleQLearningNode(actions)
        self.q_table = self.root

    def choose_action(self, dis):
        # action selection
        if np.random.uniform() < self.epsilon:
            action = self.q_table.choose_softmax(dis)
        else:
            # choose random action
            st = set(range(self.actions)) - set(dis)
            action = random.sample(list(st), 1)[0]
        return action

    def learn(self, a, r, dead):
        self.q_table.learn(self.gamma, self.lr, a, r, dead)

    def get_status(self):
        return self.q_table

    def set_status(self, q):
        self.q_table = q

    def clear(self):
        self.q_table = self.root

    def step(self, a):
        self.q_table.check_exist(a)
        self.q_table = self.q_table.children[a]

    def possibility(self, dis):
        p = self.q_table.softmax_possibility(dis)
        return p
