import numpy as np
import layers.base_conv as conv
import old_layers.base_conv as old_conv


if __name__ == "__main__":
    img = np.random.standard_normal((1, 32, 32, 3))

    conv1 = conv.Conv2D(img.shape, 12, 3, 1)
    conv2 = old_conv.Conv2D(img.shape, 12, 3, 1)

    conv1.weights = conv2.weights
    conv1.bias = conv2.bias

    next1 = conv1.forward(img)
    next2 = conv2.forward(img)

    print next1-next2
    next3 = next1.copy()
    next3+=1

    grad1 = conv1.gradient(next3-next1)
    grad2 = conv2.gradient(next3-next1)


