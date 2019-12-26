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


class Strategy(metaclass=abc.ABCMeta):
    y: 'numpy array'
    y = None

    @abc.abstractmethod
    def activate(self, x_: 'numpy array'):
        pass


    def output(self):
        return self.y


class Sigmoid(Strategy, ABC):
    # Implement the algorithm using the strategy interface
    def activate(self, x_):
        self.y = np.zeros(len(x_))
        self.y = 1 / (1 + np.exp(-x_))



class ReLU(Strategy, ABC):
    # Implementation
    def activate(self, x_):
        self.y = np.zeros(len(x_))
        for i in range(len(x_)):
            self.y[i] = 0 if x_[i] < 0 else 1


def test():
    sigmoid = Sigmoid()
    activation1 = Activation(sigmoid)
    activation1.activate(np.ones(5))
    print("Sigmoid")
    print(activation1.strategy.y)
    print("ReLU")
    relu = ReLU()
    activation2 = Activation(relu)
    activation2.activate(np.ones(5))
    print(activation2.strategy.y)


if __name__ == '__main__':
    test()
