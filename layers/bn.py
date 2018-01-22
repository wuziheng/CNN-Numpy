import numpy as np
import math

class BatchNorm(object):
    def __init__(self, shape):
        self.output_shape = shape
        self.batch_size = shape[0]
        self.input_data = np.zeros(shape)

        self.alpha = np.zeros(shape[-1])
        self.beta = np.zeros(shape[-1])
        self.a_gradient = np.zeros(shape[-1])
        self.b_gradient = np.zeros(shape[-1])

        self.moving_mean = np.zeros(shape[-1])
        self.moving_var = np.zeros(shape[-1])
        self.epsilon = 0.00001
        self.moving_decay = 0.997

    def forward(self, x, phase='train'):
        self.input_data = x
        self.mean = np.mean(x, axis=(0, 1, 2))
        self.var = self.batch_size / (self.batch_size - 1) * np.var(x,
                                                                    axis=(0, 1, 2)) if self.batch_size > 1 else np.var(
            x, axis=(0, 1, 2))

        # initialize shadow_variable with mean
        if np.sum(self.moving_mean) == 0 and np.sum(self.moving_var) == 0:
            self.moving_mean = self.mean
            self.moving_var = self.var
        # update shadow_variable with mean, var, moving_decay
        else:
            self.moving_mean = self.moving_decay * self.moving_mean  + (1 - self.moving_decay)*self.mean
            self.moving_var = self.moving_decay * self.moving_var + (1 - self.moving_decay)*self.var

        if phase == 'train':
            self.normed_x = (x - self.mean)/np.sqrt(self.var+self.epsilon)
        if phase == 'test':
            self.normed_x = (x - self.moving_mean)/np.sqrt(self.moving_var+self.epsilon)

        return self.normed_x*self.alpha+self.beta

    def gradient(self, eta):
        self.a_gradient = np.sum(eta * self.normed_x, axis=(0, 1, 2))
        self.b_gradient = np.sum(eta * self.normed_x, axis=(0, 1, 2))


        normed_x_gradient = eta * self.alpha
        var_gradient = np.sum(-1.0/2*normed_x_gradient*(self.input_data - self.mean)/(self.var+self.epsilon)**(3.0/2), axis=(0,1,2))
        mean_gradinet = np.sum(-1/np.sqrt(self.var+self.epsilon)*normed_x_gradient, axis=(0,1,2))

        x_gradient = normed_x_gradient*np.sqrt(self.var+self.epsilon)+2*(self.input_data-self.mean)*var_gradient/self.batch_size+mean_gradinet/self.batch_size

        return x_gradient

    def backward(self, alpha=0.0001):
        self.alpha -= alpha * self.a_gradient
        self.beta -= alpha * self.b_gradient

