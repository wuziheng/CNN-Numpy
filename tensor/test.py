import Variable as var
import Operator as op

import cv2
import numpy as np

img = cv2.imread('test.jpg')
img = img[np.newaxis, :]

a = var.Variable((1, 128, 128, 3), 'A')
x = op.Conv2D((3, 3, 3, 3), input_variables=a, name='conv1').output_variables

var.GLOBAL_VARIABLE_SCOPE['conv1'].weights.data = np.ones(var.GLOBAL_VARIABLE_SCOPE['conv1'].weights.shape)
var.GLOBAL_VARIABLE_SCOPE['conv1'].bias.data = np.zeros(var.GLOBAL_VARIABLE_SCOPE['conv1'].bias.shape)




