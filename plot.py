import matplotlib.pyplot as plt
import numpy as np


def saveplot(collection, title ,savename, moving_alpha = 0.9):
    x = np.arange(0, len(collection)-1)
    collection = [collection[i]*moving_alpha+collection[i+1]*(1-moving_alpha) for i in range(len(collection)-1)]
    plt.plot(x,collection)
    plt.title('%s - iteration curve'%title)
    plt.xlabel('iteration')
    plt.ylabel(title)
    plt.savefig(savename)
    return

if __name__ == "__main__":
    f = open('../record.txt').readlines()
    acc = []
    loss = []
    for line in f:
        try:
            acc.append(float(line.split(',')[2].split(' ')[2]))
            loss.append(float(line.split(',')[2].split(' ')[5]))
        except:
            pass
    print acc
    print loss
    saveplot(acc,'acc','fig/acc.jpg')