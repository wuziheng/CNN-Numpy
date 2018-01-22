import numpy as np
import matplotlib.pyplot as plt
import cv2


class AvgPooling(object):
    def __init__(self, shape, ksize=2, stride=2):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channels = shape[-1]
        self.integral = np.zeros(shape)
        self.index = np.zeros(shape)

    def forward(self, x):
        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(x.shape[1]):
                    row_sum = 0
                    for j in range(x.shape[2]):
                        row_sum += x[b, i, j, c]
                        if i == 0:
                            self.integral[b, i, j, c] = row_sum
                        else:
                            self.integral[b, i, j, c] = self.integral[b, i - 1, j, c] + row_sum

        out = np.zeros([x.shape[0], x.shape[1] / self.stride, x.shape[2] / self.stride, self.output_channels],
                       dtype=float)

        # integral calculate pooling
        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        self.index[b, i:i + self.ksize, j:j + self.ksize, c] = 1
                        if i == 0 and j == 0:
                            out[b, i / self.stride, j / self.stride, c] = self.integral[
                                b, self.ksize - 1, self.ksize - 1, c]

                        elif i == 0:
                            out[b, i / self.stride, j / self.stride, c] = self.integral[b, 1, j + self.ksize - 1, c] - \
                                                                          self.integral[b, 1, j - 1, c]
                        elif j == 0:
                            out[b, i / self.stride, j / self.stride, c] = self.integral[b, i + self.ksize - 1, 1, c] - \
                                                                          self.integral[b, i - 1, 1, c]
                        else:
                            out[b, i / self.stride, j / self.stride, c] = self.integral[
                                                                              b, i + self.ksize - 1, j + self.ksize - 1, c] - \
                                                                          self.integral[
                                                                              b, i - 1, j + self.ksize - 1, c] - \
                                                                          self.integral[
                                                                              b, i + self.ksize - 1, j - 1, c] + \
                                                                          self.integral[b, i - 1, j - 1, c]

        out /= (self.ksize * self.ksize)
        return out

    def gradient(self, eta):
        # stride = ksize
        next_eta = np.repeat(eta, self.stride, axis=1)
        next_eta = np.repeat(next_eta, self.stride, axis=2)
        next_eta = next_eta*self.index
        return next_eta/(self.ksize*self.ksize)


class MaxPooling(object):
    def __init__(self, shape, ksize=2, stride=2):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channels = shape[-1]
        self.index = np.zeros(shape)

    def forward(self, x):
        out = np.zeros([x.shape[0], x.shape[1] / self.stride, x.shape[2] / self.stride, self.output_channels])

        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        out[b, i / self.stride, j / self.stride, c] = np.max(
                            x[b, i:i + self.ksize, j:j + self.ksize, c])
                        index = np.argmax(x[b, i:i + self.ksize, j:j + self.ksize, c])
                        self.index[b, i+index/self.stride, j + index % self.stride, c] = 1
        return out

    def gradient(self, eta):
        return np.repeat(np.repeat(eta, self.stride, axis=1), self.stride, axis=2) * self.index

if __name__ == "__main__":
    img = cv2.imread('test.jpg')
    img = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])

    pool = MaxPooling(img, 2, 2)
    img1 = pool.forward(img)
    img2 = pool.gradient(img1)
    print img[0,:,:,1]
    print img1[0,:,:,1]
    print img2[0,:,:,1]

    plt.imshow(img1[0])
    plt.show()
