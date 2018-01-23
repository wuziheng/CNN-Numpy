import numpy as np
from layers.base_conv import Conv2D
from layers.fc import FullyConnect
from layers.pooling import MaxPooling, AvgPooling
from layers.softmax import Softmax
from layers.relu import Relu


import time
import struct
from glob import glob


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    images_path = glob('./%s/%s*3-ubyte' % (path, kind))[0]
    labels_path = glob('./%s/%s*1-ubyte' % (path, kind))[0]

    # labels_path = os.path.join(path,
    #                            '%s-labels-idx1-ubyte'
    #                            % kind)
    # images_path = os.path.join(path,
    #                            '%s-images-idx3-ubyte'
    #                            % kind)
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


images, labels = load_mnist('./data/mnist')
test_images, test_labels = load_mnist('./data/mnist', 't10k')

# img = images[0].reshape([1,28,28,1])
# # plt.imshow(img[0,:,:,0])
# # plt.show()
#
# conv1 = conv2d(img.shape, 12, 5, 1)
# conv1_out = conv1.forward(img)
# print 'conv1_out: ', conv1_out.shape
#
# pool1 = max_pooling(conv1_out.shape)
# pool1_out = pool1.forward(conv1_out)
# print 'pool1_out: ', pool1_out.shape
#
# conv2 = conv2d(pool1_out.shape, 24, 3, 1)
# conv2_out = conv2.forward(pool1_out)
# print 'conv2_out: ', conv2_out.shape
#
# pool2 = max_pooling(conv2_out.shape)
# pool2_out = pool2.forward(conv2_out)
# print 'pool2_out: ', pool2_out.shape
#
# fc = fullyconnect(pool2_out.shape, 10)
# fc_out = fc.forward(pool2_out)
# print 'fc_out:', fc_out.shape
#
# sf = softmax(fc_out.shape)
# sf.cal_loss(fc_out, 5)
# sf.gradient()
# print '------------------------------------'
# print 'softmax_eta: ', sf.eta.shape
#
# fc_eta = fc.gradient(sf.eta)
# print 'fc_eta: ', fc_eta.shape
#
# pool2_eta = pool2.gradient(fc_eta)
# print 'pool2_eta: ', pool2_eta.shape
#
# conv2_eta = conv2.gradient(pool2_eta)
# print 'conv2_eta: ', conv2_eta.shape
#
# pool1_eta = pool1.gradient(conv2_eta)
# print 'pool1_eta: ', pool1_eta.shape
#
# conv1_eta = conv1.gradient(pool1_eta)
# print 'conv1_eta', conv1_eta.shape

batch_size = 64

conv1 = Conv2D([batch_size, 28, 28, 1], 12, 5, 1)
relu1 = Relu(conv1.output_shape)
pool1 = MaxPooling(relu1.output_shape)
conv2 = Conv2D(pool1.output_shape, 24, 3, 1)
relu2 = Relu(conv2.output_shape)
pool2 = MaxPooling(relu2.output_shape)
fc = FullyConnect(pool2.output_shape, 10)
sf = Softmax(fc.output_shape)


# train_loss_record = []
# train_acc_record = []
# val_loss_record = []
# val_acc_record = []

for epoch in range(20):
    # if epoch < 5:
    #     learning_rate = 0.00001
    # elif epoch < 10:
    #     learning_rate = 0.000001
    # else:
    #     learning_rate = 0.0000001

    learning_rate = 1e-5

    batch_loss = 0
    batch_acc = 0
    val_acc = 0
    val_loss = 0

    # train
    train_acc = 0
    train_loss = 0
    for i in range(images.shape[0] / batch_size):
        img = images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
        label = labels[i * batch_size:(i + 1) * batch_size]
        conv1_out = relu1.forward(conv1.forward(img))
        pool1_out = pool1.forward(conv1_out)
        conv2_out = relu2.forward(conv2.forward(pool1_out))
        pool2_out = pool2.forward(conv2_out)
        fc_out = fc.forward(pool2_out)
        # print i, 'fc_out', fc_out
        batch_loss += sf.cal_loss(fc_out, np.array(label))
        train_loss += sf.cal_loss(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sf.softmax[j]) == label[j]:
                batch_acc += 1
                train_acc += 1

        sf.gradient()
        conv1.gradient(relu1.gradient(pool1.gradient(
            conv2.gradient(relu2.gradient(pool2.gradient(
                fc.gradient(sf.eta)))))))

        if i % 1 == 0:
            fc.backward(alpha=learning_rate, weight_decay=0.0004)
            conv2.backward(alpha=learning_rate, weight_decay=0.0004)
            conv1.backward(alpha=learning_rate, weight_decay=0.0004)


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
        img = test_images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
        label = test_labels[i * batch_size:(i + 1) * batch_size]
        conv1_out = relu1.forward(conv1.forward(img))
        pool1_out = pool1.forward(conv1_out)
        conv2_out = relu2.forward(conv2.forward(pool1_out))
        pool2_out = pool2.forward(conv2_out)
        fc_out = fc.forward(pool2_out)
        val_loss += sf.cal_loss(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sf.softmax[j]) == label[j]:
                val_acc += 1

    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  epoch: %5d , val_acc: %.4f  avg_val_loss: %.4f" % (
        epoch, val_acc / float(test_images.shape[0]), val_loss / test_images.shape[0])

