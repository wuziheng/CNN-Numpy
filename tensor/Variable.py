import numpy as np
from util import initializer
import math

if 'GLOBAL_VARIABLE_SCOPE' not in globals():
    # global GLOBAL_VARIABLE_SCOPE
    GLOBAL_VARIABLE_SCOPE = {}


class Variable(object):
    initial = 'MSRA'
    method = 'SGD'

    def __init__(self, shape=list, name=str, scope='', grad=True, learnable=False, init='MSRA'):
        if scope != '':
            self.scope = scope if scope[-1] == '/' else scope + '/'
            self.name = self.scope + name
        else:
            self.name = name
            self.scope = scope

        if self.name in GLOBAL_VARIABLE_SCOPE:
            raise Exception('Variable name: %s exists!' % self.name)
        else:
            GLOBAL_VARIABLE_SCOPE[self.name] = self

        for i in shape:
            if not isinstance(i, int):
                raise Exception("Variable name: %s shape is not list of int"%self.name)

        self.shape = shape
        self.data = initializer(shape, self.initial)

        self.child = []
        self.parent = []

        if grad:
            self.diff = np.zeros(self.shape)
            self.wait_bp = True
            self.learnable = learnable

    def eval(self):
        for operator in self.parent:
            GLOBAL_VARIABLE_SCOPE[operator].forward()
        self.wait_bp = True
        return self.data

    def diff_eval(self):
        if self.wait_bp:
            for operator in self.child:
                GLOBAL_VARIABLE_SCOPE[operator].backward()
            self.wait_bp = False
        else:
            pass

        return self.diff

    def apply_gradient(self, learning_rate=float, decay_rate=float, batch_size=1):
        self.data *= (1 - decay_rate)
        if self.method == 'SGD':
            learning_rate = learning_rate
            self.data -= (learning_rate*self.diff/batch_size)
            self.diff *= 0

        elif self.method == 'Momentum':
            self.mtmp = self.momentum * self.mtmp + self.diff/batch_size
            self.data -= learning_rate * self.mtmp
            self.diff *= 0

        elif self.method == 'NGA':
            self.mtmp = self.momentum * self.mtmp + self.diff / batch_size + self.momentum*(self.diff-self.lastdiff)/batch_size
            self.data -= learning_rate * self.mtmp
            self.lastdiff = self.diff
            self.diff *= 0

        elif self.method == 'Adam':
            self.t += 1
            learning_rate_t = learning_rate * math.sqrt(1 - pow(self.beta2, self.t)) / (1 - pow(self.beta1, self.t))
            self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * self.diff / batch_size
            self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * ((self.diff / batch_size) ** 2)
            self.data -= learning_rate_t * self.m_t / (self.v_t + self.epsilon) ** 0.5
            self.diff *= 0

        else:
            raise Exception('No apply_gradient method: %s'%self.method)

    def set_method_sgd(self):
        self.method = 'SGD'

    def set_method_momentum(self, momentum=0.9):
        self.method = 'Momentum'
        self.momentum = momentum
        self.mtmp = np.zeros(self.diff.shape)

    def set_method_nga(self,momentum=0.9):
        self.method = 'NGA'
        self.lastdiff = np.zeros(self.diff.shape)
        self.momentum= momentum
        self.mtmp = np.zeros(self.diff.shape)

    def set_method_adam(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.method = 'Adam'
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_t = np.zeros(self.diff.shape)
        self.v_t = np.zeros(self.diff.shape)
        self.t = 0



def get_by_name(name):
    if 'GLOBAL_VARIABLE_SCOPE' in globals():
        try:
            return GLOBAL_VARIABLE_SCOPE[name]
        except:
            raise Exception('GLOBAL_VARIABLE_SCOPE not include name: %s'%name)
    else:
        raise Exception('No Variable')


if __name__ == "__main__":
    shape = (3, 3, 12, 24)
    a = Variable(shape, 'a')
    print a.name

