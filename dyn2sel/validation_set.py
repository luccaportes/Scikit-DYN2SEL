import numpy as np


class ValidationSet:
    def __init__(self, max_size=100, min_size=1):
        self.max_size = max_size
        self.min_size = min_size
        self.buffer_X = []
        self.buffer_y = []
        self.index = 0

    def add_instances(self, X, y):
        for x_i, y_i in zip(X, y):
            if len(self.buffer_X) < self.max_size:
                self.buffer_X.append(x_i)
                self.buffer_y.append(y_i)
            else:
                self.buffer_X = self.buffer_X[1:] + [x_i]
                self.buffer_y = self.buffer_y[1:] + [y_i]

    def remove_instance(self, index=0):
        self.buffer_X.pop(index)
        self.buffer_y.pop(index)

    def clear(self):
        self.buffer_X = []
        self.buffer_y = []

    def replace_set(self, X, y):
        self.buffer_X = X
        self.buffer_y = y
