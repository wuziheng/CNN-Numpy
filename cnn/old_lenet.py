import numpy as np
from old_layers.base_conv import Conv2D
#from layers.base_conv import Conv2D
from old_layers.fc import FullyConnect
from old_layers.pooling import MaxPooling, AvgPooling
from old_layers.softmax import Softmax
from old_layers.relu import relu

import time
import struct
from glob import glob

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    images_path = glob('./%s/%s*3-ubyte'%(path,kind))[0]
    labels_path = glob('./%s/%s*1-ubyte'%(path,kind))[0]

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

images, labels = load_mnist('./data')
test_images, test_labels = load_mnist('./data', 't10k')

conv1 = Conv2D([1, 28, 28, 1], 12, 5, 1)
relu1 = relu([1,24,24,12])
pool1 = MaxPooling([1, 24, 24, 12])
conv2 = Conv2D([1, 12, 12, 12], 24, 3, 1)
relu2 = relu([1,10,10,24])
pool2 = MaxPooling([1, 10, 10, 24])
fc = FullyConnect([1, 5, 5, 24], 10)
sf = Softmax([10])


for epoch in range(10):
    if epoch <4:
        learning_rate = 0.00001
    elif epoch < 6:
        learning_rate = 0.000001
    else:
        learning_rate = 0.0000001

    batch_size = 32
    batch_loss = 0
    batch_acc = 0
    val_acc = 0
    val_loss = 0

    # train
    for i in range(images.shape[0]):
        img = images[i].reshape([1,28,28,1])
        label = labels[i]

        conv1_out = relu1.forward(conv1.forward(img))
        pool1_out = pool1.forward(conv1_out)
        conv2_out = relu2.forward(conv2.forward(pool1_out))
        pool2_out = pool2.forward(conv2_out)
        fc_out = fc.forward(pool2_out)
        batch_loss += sf.cal_loss(fc_out, label)

        if np.argmax(sf.softmax) == label:
            batch_acc+=1

        sf.gradient()
        conv1.gradient(
            relu1.gradient(pool1.gradient(conv2.gradient(relu2.gradient(pool2.gradient(fc.gradient(sf.eta)))))))

        if i % batch_size == 0:
            fc.backward(alpha=learning_rate)
            conv2.backward(alpha=learning_rate)
            conv1.backward(alpha=learning_rate)

            if i % 2000 == 0:
                print time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + \
                      "  epoch: %d ,  batch: %5d , avg_batch_acc: %.4f  avg_batch_loss: %.4f  learning_rate: %f" % (epoch,
                    i / batch_size, batch_acc / float(batch_size), batch_loss / batch_size, learning_rate)

            batch_loss = 0
            batch_acc = 0

    # validation
    for i in range(test_images.shape[0]):
        img = test_images[i].reshape([1, 28, 28, 1])
        label = test_labels[i]
        conv1_out = relu1.forward(conv1.forward(img))
        pool1_out = pool1.forward(conv1_out)
        conv2_out = relu2.forward(conv2.forward(pool1_out))
        pool2_out = pool2.forward(conv2_out)
        fc_out = fc.forward(pool2_out)
        val_loss += sf.cal_loss(fc_out, label)
        if np.argmax(sf.softmax) == label:
            val_acc += 1

    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  epoch: %5d , val_acc: %.4f  avg_val_loss: %.4f" % (
    epoch, val_acc / float(test_images.shape[0]), val_loss / test_images.shape[0])
