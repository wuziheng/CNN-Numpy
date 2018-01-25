import numpy as np
import tensor.Variable as var
import tensor.Operator as op

import time
import struct
from glob import glob


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    images_path = glob('./%s/%s*3-ubyte' % (path, kind))[0]
    labels_path = glob('./%s/%s*1-ubyte' % (path, kind))[0]

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def inference(x, output_num):
    conv1_out = op.Conv2D((5, 5, 1, 12), input_variable=x, name='conv1', padding='VALID').output_variables
    relu1_out = op.Relu(input_variable=conv1_out, name='relu1').output_variables
    pool1_out = op.MaxPooling(ksize=2, input_variable=relu1_out, name='pool1').output_variables

    conv2_out = op.Conv2D((3, 3, 12, 24), input_variable=pool1_out, name='conv2').output_variables
    relu2_out = op.Relu(input_variable=conv2_out, name='relu2').output_variables
    pool2_out = op.MaxPooling(ksize=2, input_variable=relu2_out, name='pool2').output_variables

    fc_out = op.FullyConnect(output_num=output_num, input_variable=pool2_out, name='fc').output_variables
    return fc_out


batch_size = 64
img_placeholder = var.Variable((batch_size, 28, 28, 1), 'input')
label_placeholder = var.Variable([batch_size, 1], 'label')

# set train_op
prediction = inference(img_placeholder, 10)
sf = op.SoftmaxLoss(prediction, label_placeholder, 'sf')


images, labels = load_mnist('./data/mnist')
test_images, test_labels = load_mnist('./data/mnist', 't10k')


for epoch in range(20):
    learning_rate = 1e-5

    batch_loss = 0
    batch_acc = 0
    val_acc = 0
    val_loss = 0

    # train
    train_acc = 0
    train_loss = 0

    for i in range(images.shape[0] / batch_size):
        # feed
        img_placeholder.data = images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
        label_placeholder.data = labels[i * batch_size:(i + 1) * batch_size]

        # forward
        _loss = sf.loss.eval()
        _prediction = sf.softmax
        batch_loss += _loss
        train_loss += _loss

        for j in range(batch_size):
            if np.argmax(_prediction[j]) == label_placeholder.data[j]:
                batch_acc += 1
                train_acc += 1

        # backward
        img_placeholder.diff_eval()

        for k in var.GLOBAL_VARIABLE_SCOPE:
            s = var.GLOBAL_VARIABLE_SCOPE[k]
            if isinstance(s, var.Variable) and s.learnable:
                s.apply_gradient(learning_rate=learning_rate, decay_rate=0.0004)
            if isinstance(s, var.Variable):
                s.diff = np.zeros(s.shape)


        if i % 50 == 0:
            print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + \
                      "  epoch: %d ,  batch: %5d , avg_batch_acc: %.4f  avg_batch_loss: %.4f  learning_rate %f" % (epoch,
                                                                                                 i, batch_acc / float(
                          batch_size), batch_loss / batch_size, learning_rate)

        batch_loss = 0
        batch_acc = 0


    print time.strftime("%Y-%m-%d %H:%M:%S",
                            time.localtime()) + "  epoch: %5d , train_acc: %.4f  avg_train_loss: %.4f" % (
            epoch, train_acc / float(images.shape[0]), train_loss / images.shape[0])

    # validation
    for i in range(test_images.shape[0] / batch_size):
        img_placeholder.data = test_images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
        label_placeholder.data = test_labels[i * batch_size:(i + 1) * batch_size]

        _loss = sf.loss.eval()
        _prediction = sf.prediction.eval()

        val_loss += _loss

        for j in range(batch_size):
            if np.argmax(sf.softmax[j]) == label_placeholder.data[j]:
                val_acc += 1

    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  epoch: %5d , val_acc: %.4f  avg_val_loss: %.4f" % (
        epoch, val_acc / float(test_images.shape[0]), val_loss / test_images.shape[0])