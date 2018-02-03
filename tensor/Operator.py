import numpy as np
from Variable import Variable, GLOBAL_VARIABLE_SCOPE


class Operator(object):
    def __init__(self, name, input_variables, output_variables):
        
        # init input check
        if GLOBAL_VARIABLE_SCOPE.has_key(name):
            raise Exception("Operator %s has exists !"%name)
        
        if not isinstance(input_variables, Variable) and not isinstance(input_variables[0], Variable):
            raise Exception("Operator %s 's input_variables is not instance(or list) of Variable!")

        if not isinstance(output_variables, Variable) and not isinstance(output_variables[0], Variable):
            raise Exception("Operator %s 's output_variables is not instance(or list) of Variable!")

        # register in GLOBAL_OP_SCOPE
        self.name = name
        GLOBAL_VARIABLE_SCOPE[self.name] = self
        
        self.child = []
        self.parent =[]
        
        # register for input Variable's child and output Variable's parents
        register_graph(input_variables, output_variables, self)

        self.wait_forward = True
        # self.wait_backward = not self.wait_forward

    def forward(self):
        pass
        # if self.wait_forward == True:
        #     1.check_parent_eval()
        #         for variable in self.parent:
        #             variable.eval()
        #     2.do forward_cal()
        #     3.set wait_forward()
        #         self.wait_forward = False
        # else:
        #     pass

    def backward(self):
        pass
        # if self.wait_forward == True:
        #     pass
        # else:
        #     1.check_child_diffeval()
        #         for variable in self.child:
        #             variable.diff_eval()
        #     2.do backward_cal()
        #     3.set wait forward()
        #         self.wait_forward=True
        #


class Conv2D(Operator):

    def __init__(self, kernel_shape=list, input_variable=Variable, name=str, stride=1, padding='SAME'):
        # kernel_shape = [ksize, ksize, input_channels, output_channels]
        for i in kernel_shape:
            if not isinstance(i, int):
                raise Exception("Operator Conv2D name: %s kernel shape is not list of int" % self.name)

        if not isinstance(input_variable, Variable):
            raise Exception("Operator Conv2D name: %s's input_variable is not instance of Variable" % name)

        if len(input_variable.shape)!=4:
            raise Exception("Operator Conv2D name: %s's input_variable's shape != 4d Variable!" % name)

        self.ksize = kernel_shape[0]
        self.stride = stride
        self.output_num = kernel_shape[-1]
        self.padding = padding
        self.col_image = []

        self.weights = Variable(kernel_shape, scope=name, name='weights',learnable=True)
        self.bias = Variable([self.output_num], scope=name, name='bias', learnable=True)
        self.batch_size = input_variable.shape[0]

        if self.padding == 'SAME':
            _output_shape = [self.batch_size, input_variable.shape[1] / stride, input_variable.shape[2] / stride,
                             self.output_num]
        if self.padding == 'VALID':
            _output_shape = [self.batch_size, (input_variable.shape[1] - self.ksize + 1) / stride,
                             (input_variable.shape[2] - self.ksize + 1) / stride, self.output_num]

        self.output_variables = Variable(_output_shape, name='out', scope=name)  # .name
        self.input_variables = input_variable
        Operator.__init__(self, name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self._conv(self.input_variables, self.output_variables, self.weights.data, self.bias.data)
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
            self._deconv(self.input_variables, self.output_variables, self.weights, self.bias)
            self.wait_forward = True
            return

    def _deconv(self, input=Variable, output=Variable, weights=Variable, bias=Variable):
        col_eta = np.reshape(output.diff, [self.batch_size, -1, self.output_num])
        for i in range(self.batch_size):
            weights.diff += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
        bias.diff += np.sum(col_eta, axis=(0, 1))

        # deconv of padded eta with flippd kernel to get next_eta
        if self.padding == 'VALID':
            pad_eta = np.pad(output.diff, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                             'constant', constant_values=0)

        if self.padding == 'SAME':
            pad_eta = np.pad(output.diff, (
                (0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                             'constant', constant_values=0)

        col_pad_eta = np.array([im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batch_size)])
        flip_weights = np.flipud(np.fliplr(weights.data))
        col_flip_weights = flip_weights.reshape([-1, weights.shape[2]])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, input.shape)
        input.diff = next_eta
        return

    def _conv(self, input=Variable, output=Variable, weights=np.ndarray, bias=np.ndarray):
        # reshape weights to col
        col_weights = weights.reshape(-1, self.output_num)

        # padding input_img according to method
        if self.padding == 'SAME':
            batch_img = np.pad(input.data, (
                (0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                               'constant', constant_values=0)
        else:
            batch_img = input.data

        # malloc tmp output_data
        conv_out = np.zeros(output.data.shape)

        self.col_image = []
        # do dot for every image in batch by im2col dot col_weight
        for i in range(self.batch_size):
            img_i = batch_img[i][np.newaxis, :]
            col_image_i = im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(col_image_i, col_weights) + bias, output.data[0].shape)
            self.col_image.append(col_image_i)
        self.col_image = np.array(self.col_image)

        output.data = conv_out
        return


class MaxPooling(Operator):
    def __init__(self, ksize=2, input_variable=Variable, name=str, stride=2):

        if not isinstance(input_variable, Variable):
            raise Exception("Operator Conv2D name: %s's input_variable is not instance of Variable" % name)


        self.ksize = ksize
        self.stride = stride
        self.batch_size = input_variable.shape[0]
        self.output_channels = input_variable.shape[-1]
        self.index = np.zeros(input_variable.shape)

        self.input_variables = input_variable
        _output_shape = [self.batch_size, input_variable.shape[2] / stride, input_variable.shape[2] / stride,
                         self.output_channels]
        self.output_variables = Variable(_output_shape, name='out', scope=name)
        Operator.__init__(self, name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self._pool()
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
            self.input_variables.diff = np.repeat(np.repeat(self.output_variables.diff, self.stride, axis=1),
                                                  self.stride, axis=2) * self.index
            self.wait_forward = True
            return

    def _pool(self):
        _out = np.zeros(self.output_variables.shape)
        for b in range(self.input_variables.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, self.input_variables.shape[1], self.stride):
                    for j in range(0, self.input_variables.shape[2], self.stride):
                        _out[b, i / self.stride, j / self.stride, c] = np.max(
                            self.input_variables.data[b, i:i + self.ksize, j:j + self.ksize, c])
                        index = np.argmax(self.input_variables.data[b, i:i + self.ksize, j:j + self.ksize, c])
                        self.index[b, i+index/self.stride, j + index % self.stride, c] = 1
        self.output_variables.data = _out
        return


class FullyConnect(Operator):
    def __init__(self, output_num, input_variable=Variable, name=str):
        if not isinstance(input_variable, Variable):
            raise Exception("Operator Conv2D name: %s's input_variable is not instance of Variable" % name)

        self.batch_size = input_variable.shape[0]
        input_len = reduce(lambda x, y: x * y, input_variable.shape[1:])
        self.output_num = output_num
        self.weights = Variable([input_len, self.output_num], name='weights', scope=name, init='const' ,learnable=True)
        self.bias = Variable([self.output_num], name='bias', scope=name, init='const',learnable=True)

        self.output_variables = Variable([self.batch_size, self.output_num], name='out', scope=name)
        self.input_variables = input_variable
        Operator.__init__(self, name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:

            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.flatten_x = self.input_variables.data.reshape([self.batch_size, -1])
            self.output_variables.data = np.dot(self.flatten_x, self.weights.data)+self.bias.data
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

            for i in range(self.batch_size):
                col_x = self.flatten_x[i][:, np.newaxis]
                diff_i = self.output_variables.diff[i][:, np.newaxis].T
                self.weights.diff += np.dot(col_x, diff_i)
                self.bias.diff += diff_i.reshape(self.bias.shape)
            next_diff = np.dot(self.output_variables.diff, self.weights.data.T)
            self.input_variables.diff = np.reshape(next_diff, self.input_variables.shape)

            self.wait_forward = True
            return


class SoftmaxLoss(Operator):
    def __init__(self, predict = Variable, label=Variable, name=str):
        self.batch_size = predict.shape[0]
        self.input_variables = [predict, label]
        self.loss = Variable([1], name='loss', scope=name, init='None')
        self.prediction = Variable(predict.shape, name='prediction', scope=name)
        self.softmax = np.zeros(self.prediction.shape)

        self.output_variables = [self.loss, self.prediction]
        Operator.__init__(self, name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()

            predict = self.input_variables[0].data
            label = self.input_variables[1].data

            self.prediction.data = self.predict(predict)

            self.loss.data = 0
            for i in range(self.batch_size):
                self.loss.data += np.log(np.sum(np.exp(predict[i]))) - predict[i, label[i]]

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
            self.input_variables[0].diff = self.softmax.copy()
            for i in range(self.batch_size):
                self.input_variables[0].diff[i, self.input_variables[1].data[i]] -= 1
            self.wait_forward = True
            return

    def predict(self, prediction):
        exp_prediction = np.zeros(prediction.shape)
        self.softmax = np.zeros(prediction.shape)
        for i in range(self.batch_size):
            prediction[i, :] -= np.max(prediction[i, :])
            exp_prediction[i] = np.exp(prediction[i])
            self.softmax[i] = exp_prediction[i]/np.sum(exp_prediction[i])
        return self.softmax


class DropOut(Operator):
    def __init__(self, name, phase, input_variable=Variable, prob=0.5):
        self.input_variables = input_variable
        self.output_variables = Variable(shape=input_variable.shape, scope=name, name='out')
        self.prob = prob
        self.phase = phase
        self.index = np.ones(input_variable.shape)

        Operator.__init__(self, name, self.input_variables, self.output_variables)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            if self.phase == 'train':
                self.index = np.random.random(self.input_variables.shape) < self.prob
                self.output_variables.data = self.input_variables.data * self.index
                self.output_variables.data /= self.prob
            elif self.phase == 'test':
                self.output_variables.data = self.input_variables.data
            else:
                raise Exception('Operator %s phase is not in test or train'% self.name)

            self.wait_forward=False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            if self.phase == 'train':
                self.input_variables.diff = self.output_variables.diff * self.index / self.prob
            elif self.phase == 'test':
                self.output_variables.diff = self.input_variables.diff
            else:
                raise Exception('Operator %s phase is not in test or train'% self.name)

            self.wait_forward = True
            return


def register_graph(input_variable, output_variable, operator):
    if isinstance(input_variable,Variable) and isinstance(output_variable, Variable):
        input_variable.child.append(operator.name)
        output_variable.parent.append(operator.name)
        operator.parent.append(input_variable.name)
        operator.child.append(output_variable.name)

    elif isinstance(input_variable, Variable) and len(output_variable)>1:
        for output in output_variable:
            input_variable.child.append(operator.name)
            output.parent.append(operator.name)
            operator.parent.append(input_variable.name)
            operator.child.append(output.name)

    elif isinstance(output_variable, Variable) and len(input_variable)>1:
        for _input in input_variable:
            _input.child.append(operator.name)
            output_variable.parent.append(operator.name)
            operator.parent.append(_input.name)
            operator.child.append(output_variable.name)

    elif len(output_variable)> 1 and len(input_variable)> 1:
        for _input in input_variable:
            _input.child.append(operator.name)
            operator.parent.append(_input.name)
        for output in output_variable:
            output.parent.append(operator.name)
            operator.child.append(output.name)

    else:
        raise Exception('Operator name %s input,output list error'% operator.name)


def im2col(image, ksize, stride):
    # image is a 4d tensor([batchsize, width ,height, channel])
    # print image.shape
    # print image
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])

            image_col.append(col)
    image_col = np.array(image_col)

    return image_col


if __name__ == "__main__":
    A = Variable((2,2,3,3),'A')
    B = Variable((3,3,4,4),'B')
    print np.zeros((1,1))