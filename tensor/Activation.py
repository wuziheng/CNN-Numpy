import numpy as np
from Variable import Variable, GLOBAL_VARIABLE_SCOPE
from Operator import Operator


class Relu(Operator):
    def __init__(self, input_variable=Variable, name=str):
        self.input_variables = input_variable
        self.output_variables = Variable(self.input_variables.shape, name='out', scope=name)
        Operator.__init__(self, name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.output_variables.data = np.maximum(self.input_variables.data, 0)
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.input_variables.diff = self.output_variables.diff
            self.output_variables.diff[self.input_variables.data < 0] = 0
            self.wait_forward = True
            return


class LRelu(Operator):
    def __init__(self, input_variable=Variable, name=str, alpha = 0.01):
        self.input_variables = input_variable
        self.output_variables = Variable(self.input_variables.shape, name='out', scope=name)
        self.alpha = alpha
        Operator.__init__(self, name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.output_variables.data = np.maximum(self.input_variables.data, 0) + self.alpha * np.minimum(
                self.input_variables.data, 0)
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.input_variables.diff = self.output_variables.diff
            self.input_variables.diff[self.input_variables.data <= 0] *= self.alpha
            self.wait_forward = True

            return


class Sigmoid(Operator):
    def __init__(self, input_variable=Variable, name=str):
        self.input_variables = input_variable
        self.output_variables = Variable(self.input_variables.shape, name='out', scope=name)
        Operator.__init__(self, name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            # y = 1/(1+exp(-x))
            # print 'fuck:',self.input_variables.data
            self.output_variables.data = 1.0/(1.0+np.exp(-self.input_variables.data))
            # print 'fuck out:', self.output_variables.data
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            # eta_x = eta_y * (1-y) * y
            self.input_variables.diff = self.output_variables.data * (
            1 - self.output_variables.data) * self.output_variables.diff
            self.wait_forward = True
            return


class Tanh(Operator):
    def __init__(self, input_variable=Variable, name=str):
        self.input_variables = input_variable
        self.output_variables = Variable(self.input_variables.shape, name='out', scope=name)
        Operator.__init__(self, name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.output_variables.data = 2 * 1.0/(1.0+np.exp(-2*self.input_variables.data)) - 1
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.input_variables.diff = self.output_variables.diff * (1 - self.output_variables.data**2)
            self.wait_forward = True
            return


class Elu(Operator):
    def __init__(self, input_variable=Variable, name=str, alpha = 0.1):
        self.input_variables = input_variable
        self.output_variables = Variable(self.input_variables.shape, name='out', scope=name)
        self.alpha = alpha
        Operator.__init__(self, name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.output_variables.data = np.maximum(self.input_variables.data, 0) + self.alpha * (
            np.exp(np.minimum(self.input_variables.data, 0)) - 1)
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.input_variables.diff = self.output_variables.diff
            self.output_variables.diff[self.input_variables.data <= 0] *= (
            self.alpha * np.exp(self.input_variables.data[self.input_variables.data <= 0]))
            self.wait_forward = True
            return


class Prelu(Operator):
    def __init__(self, input_variable=Variable, name=str, alpha = 0.25):
        self.input_variables = input_variable
        self.output_variables = Variable(self.input_variables.shape, name='out', scope=name)
        self.alpha = alpha
        self.momentum  = 0.9
        self.eta = 1e-4
        Operator.__init__(self, name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.output_variables.data = np.maximum(self.input_variables.data, 0) + self.alpha * np.minimum(
                self.input_variables.data, 0)
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.alpha = self.momentum * self.alpha + self.eta * np.sum(np.minimum(self.input_variables.data, 0))
            self.input_variables.diff = self.output_variables.diff
            self.input_variables.diff[self.input_variables.data <= 0] *= self.alpha
            self.wait_forward = True

            return



if __name__ == "__main__":

    # check grad
    shape = ([10])
    a = Variable(shape, 'a')
    # print a.name
    # print 'a.data ', a.data
    # print 'a.diff ', a.diff

    test_layer = Prelu(a, 'Tanh')
    b = test_layer.output_variables

    epsilon = 1e-5

    a.data -= epsilon
    print 'a -eps ', a.data
    out1 = b.eval()

    # refresh graph
    b.wait_bp = False
    a.wait_bp = False
    test_layer.wait_forward = True

    a.data += 2*epsilon
    print 'a +eps ', a.data
    out2 = b.eval()

    # refresh graph
    b.wait_bp = False
    a.wait_bp = False
    test_layer.wait_forward = True

    a.data -= epsilon
    b.eval()

    b.diff = np.array([1.0]*10, dtype=float)
    print 'bp:        ', a.diff_eval()
    print 'grad_check:', (out2 - out1)/2/epsilon
