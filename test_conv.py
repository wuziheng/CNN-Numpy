import tensor.Variable as var
import tensor.Operator as op

from layers.base_conv import Conv2D
from layers.pooling import MaxPooling
from layers.fc import FullyConnect
from layers.relu import Relu
from layers.softmax import Softmax

import cv2
import numpy as np

img = cv2.imread('layers/test.jpg')
img = img[np.newaxis, :]

a = var.Variable((1, 128, 128, 3), 'a')
label = var.Variable([1, 1], 'label')
import random
label.data = np.array([random.randint(1,9)])
label.data = label.data.astype(int)

conv1_out = op.Conv2D((3, 3, 3, 3), input_variable=a, name='conv1',padding='VALID').output_variables
relu1_out = op.Relu(input_variable=conv1_out, name='relu1').output_variables
pool1_out = op.MaxPooling(ksize=2, input_variable=relu1_out, name='pool1').output_variables
fc1_out = op.FullyConnect(output_num=10, input_variable=pool1_out, name='fc1').output_variables
sf_out = op.SoftmaxLoss(predict=fc1_out,label=label,name='sf').loss

new_conv1 = op.GLOBAL_VARIABLE_SCOPE['conv1']
new_fc1 = op.GLOBAL_VARIABLE_SCOPE['fc1']


conv1 = Conv2D([1, 128, 128, 3], 3, 3, 1,method='VALID')
relu1 = Relu(conv1.output_shape)
pool1 = MaxPooling(conv1.output_shape)
fc1 = FullyConnect(pool1.output_shape,10)
sf = Softmax(fc1.output_shape)

conv1.weights = new_conv1.weights.data
conv1.bias = new_conv1.bias.data
fc1.weights = new_fc1.weights.data
fc1.bias = new_fc1.bias.data


out = sf.cal_loss(fc1.forward(pool1.forward(relu1.forward(conv1.forward(img)))), label.data)
sf.gradient()
eta = conv1.gradient(relu1.gradient(pool1.gradient(fc1.gradient(sf.eta))))


#
# new train op
# give value and forward
a.data = img
new_out = sf_out.eval()

# give diff and backward
new_eta = a.diff_eval()
print new_eta-eta


for k in var.GLOBAL_VARIABLE_SCOPE:
    s = var.GLOBAL_VARIABLE_SCOPE[k]
    if isinstance(s,var.Variable) and s.learnable:
        print s.name, s.parent, s.child
