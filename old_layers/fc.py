import numpy as np
from functools import reduce
import math

class FullyConnect(object):
    def __init__(self, shape, output_num=2):
        self.input_shape = shape
        self.output_shape = [output_num]
        input_len = reduce(lambda x, y: x * y, shape)

        self.weights = np.random.standard_normal((input_len, output_num))/100
        self.bias = np.random.standard_normal(output_num)/100

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def forward(self, x):
        self.x = x.flatten()
        output = np.dot(self.x, self.weights)+self.bias
        return output


    def gradient(self, eta):
        col_x = self.x[:, np.newaxis]
        eta = eta[:, np.newaxis].T
        # print col_x.shape,eta.shape
        self.w_gradient += np.dot(col_x, eta)
        self.b_gradient += eta.reshape(self.bias.shape)
        next_eta = np.dot(eta, self.weights.T)
        next_eta = np.reshape(next_eta, self.input_shape)

        return next_eta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias
        # zero gradient
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)


if __name__ == "__main__":
    img = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    fc = FullyConnect(img, 2)

    out = fc.forward()

    fc.gradient(np.array([1, -2]))

    print fc.w_gradient
    print fc.b_gradient

    fc.backward()
    print fc.weights
