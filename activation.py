import abc
import math
from abc import ABC

import numpy as np


class Activation:

    def __init__(self, strategy):
        self.strategy = strategy

    def activate(self, x_):
        self.strategy.activate(x_)

    def output(self, x_):
        self.strategy.activate(x_)
        return self.strategy.output()


    def derivative(self, x_):
        self.strategy.activate(x_)
        return self.strategy.delta()


class Strategy(metaclass=abc.ABCMeta):
    y: 'numpy array'
    y = None
    d_y = None
    @abc.abstractmethod
    def activate(self, x_: 'numpy array'):
        pass


    def output(self):
        return self.y

    def delta(self):
        return self.d_y


class Sigmoid(Strategy, ABC):
    # Implement the algorithm using the strategy interface
    def activate(self, x_):
        self.y = np.zeros(len(x_))
        self.y = 1 / (1 + np.exp(-x_))
        self.d_y = self.y * (1 - self.y)


class ReLU(Strategy, ABC):
    # Implementation
    def activate(self, x_):
        self.y = np.zeros(len(x_))
        self.d_y = np.zeros(len(x_))
        for i in range(len(x_)):
            self.y[i] = 0 if x_[i] < 0 else x_[i]
            self.d_y[i] = 0 if x_[i] < 0 else 1

def test():
    print("Sigmoid")
    sigmoid = Sigmoid()
    activation1 = Activation(sigmoid)
    print(activation1.output(np.array([0.8305, -0.526])))


    print("ReLU")
    relu = ReLU()
    activation2 = Activation(relu)
    print(activation2.output(np.ones(1)))


if __name__ == '__main__':
    test()
