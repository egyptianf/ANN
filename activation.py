import abc
import math


class Activation:

    def __init__(self, strategy):
        self.strategy = strategy

    def activate(self, x_):
        self.strategy.activate(x_)


class Strategy(metaclass=abc.ABCMeta):
    y = 0

    @abc.abstractmethod
    def activate(self, x_):
        pass


class Sigmoid(Strategy):
    # Implement the algorithm using the strategy interface
    def activate(self, x_):
        self.y = 1 / (1 + math.exp(-x_))


class ReLU(Strategy):
    # Implementation
    def activate(self, x_):
        self.y = 0 if x_ < 0 else x_


def main():
    sigmoid = Sigmoid()
    activation = Activation(sigmoid)
    activation.activate(5)
    print(activation.strategy.y)


if __name__=='__main__':
    print("hi")
    main()
