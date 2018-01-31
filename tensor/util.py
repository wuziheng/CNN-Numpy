import numpy as np
import math


def initializer(shape, method):
    if method == 'const':
        return np.random.standard_normal(shape) / 100

    if method == 'None':
        return np.zeros(shape)

    if method == 'MSRA':
        weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / shape[-1])
        return np.random.standard_normal(shape) / weights_scale


def learning_rate_exponential_decay(learning_rate, global_step, decay_rate=0.1, decay_steps=5000):
    '''
    Applies exponential decay to learning rate
    decayed_learning_rate = learning_rate * decay_rate ^ (global_step/decay_steps)
    :return: learning rate decayed by step
    '''

    decayed_learning_rate = learning_rate * pow(decay_rate,float(global_step/decay_steps))
    return decayed_learning_rate
