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



