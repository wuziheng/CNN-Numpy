import tensor.Variable as var
import tensor.Operator as op

from layers.base_conv import Conv2D
import cv2
import numpy as np

img = cv2.imread('tensor/test.jpg')
img = img[np.newaxis, :]

a = var.Variable((1, 128, 128, 3), 'A')
x = op.Conv2D((3, 3, 3, 3), input_variables=a, name='conv1').output_variables
conv1 = Conv2D((1, 128, 128, 3), 3, 3, 1, method='SAME')
#
print var.GLOBAL_VARIABLE_SCOPE['conv1'].weights.shape, conv1.weights.shape
print var.GLOBAL_VARIABLE_SCOPE['conv1'].bias.shape, conv1.bias.shape

var.GLOBAL_VARIABLE_SCOPE['conv1'].weights.data =conv1.weights # np.ones(var.GLOBAL_VARIABLE_SCOPE['conv1'].weights.shape)
var.GLOBAL_VARIABLE_SCOPE['conv1'].bias.data = conv1.bias # np.zeros(var.GLOBAL_VARIABLE_SCOPE['conv1'].bias.shape)

new_conv1 = var.GLOBAL_VARIABLE_SCOPE['conv1']

a.data=img

new_out = x.eval()
out = conv1.forward(img)
out_1 = out.copy() -1
# print new_out-out

eta = conv1.gradient(out_1 - out)
x.diff = (out_1-out)
new_eta = a.diff_eval()
