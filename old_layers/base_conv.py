import numpy as np
from functools import reduce
import math

class Conv2D(object):
    def __init__(self, shape, output_channels, ksize=3, stride=1):
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        self.stride = stride
        self.ksize = ksize

        weights_scale = reduce(lambda x, y: x * y, shape) / self.output_channels
        self.weights = np.random.standard_normal((ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale

        self.eta = np.zeros((shape[0], (shape[1] - ksize + 1) / self.stride, (shape[1] - ksize + 1) / self.stride,
             self.output_channels))
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

        if (shape[1] - ksize) % stride != 0:
            print 'input tensor width can\'t fit stride'
        if (shape[2] - ksize) % stride != 0:
            print 'input tensor height can\'t fit stride'

    def forward(self, x):
        col_weights = self.weights.reshape([-1, self.output_channels])
        self.col_image = im2col(x, self.ksize, self.stride)
        conv_out = np.dot(self.col_image, col_weights) + self.bias
        output_data = conv_out.reshape(self.eta.shape)
        return output_data

    def gradient(self, eta):
        self.eta = eta

        col_eta = np.reshape(eta, [-1, self.output_channels])
        self.w_gradient += np.dot(self.col_image.T, col_eta).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_eta, axis=0)

        # deconv of padded eta with flippd kernel to get next_eta
        pad_eta = np.pad(self.eta, (
            (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize  - 1, self.ksize - 1), (0, 0)),
                         'constant', constant_values=0)

        col_pad_eta = im2col(pad_eta, self.ksize, self.stride)
        flip_weights = np.flipud(np.fliplr(self.weights))
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)


def im2col(image, ksize, stride):
    # image is a 4d tensor([batchsize, width ,height, channel])
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)

    return image_col


if __name__ == "__main__":
    img = np.random.standard_normal((1, 32, 32, 3))
    conv = Conv2D(img, 12, 3, 1)
    next = conv.forward()
