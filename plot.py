import matplotlib.pyplot as plt
import numpy as np


def saveplot(collection, title, savename, moving_alpha = 0.9):
    x = np.arange(0, len(collection)-1)
    collection = [collection[i]*moving_alpha+collection[i+1]*(1-moving_alpha) for i in range(len(collection)-1)]
    plt.plot(x, collection)
    plt.title('%s - iteration curve'%title)
    plt.xlabel('iteration')
    plt.ylabel(title)
    plt.savefig(savename)
    plt.close('all')
    return

def plot(collection, title, moving_alpha = 0.9):
    x = np.arange(0, len(collection)-1)
    collection = [collection[i]*moving_alpha+collection[i+1]*(1-moving_alpha) for i in range(len(collection)-1)]
    plt.plot(x, collection)
    plt.title('%s - iteration curve'%title)
    plt.xlabel('iteration')
    plt.ylim(0,1)
    plt.ylabel(title)
    plt.show()


if __name__ == "__main__":
    from glob import glob
    filelist = glob('logs/*.txt')
    print filelist
    moving_alpha = 0.9

    plt.figure(figsize=(4,3))


    for file in filelist:
        name = file.split('/')[-1].split('.')[0]
        print name
        f = open(file).readlines()
        acc = []
        loss = []
        for line in f:
            try:
                acc.append(float(line.split(',')[2].split(' ')[2]))
                loss.append(float(line.split(',')[2].split(' ')[5]))
            except:
                pass
        loss = loss[0:101]
        loss = [loss[i] * moving_alpha + loss[i + 1] * (1 - moving_alpha) for i in
                      range(len(loss) - 1)]
        acc = acc[0:101]
        acc = [acc[i] * moving_alpha + acc[i + 1] * (1 - moving_alpha) for i in
                      range(len(acc) - 1)]
        x = range(0,len(acc))
        plt.plot(x,acc,label=name.split('_')[1])

    plt.xlabel('iteration(50batch)')
    plt.ylabel('acc')
    # plt.ylim(0,1)
    plt.title('acc-iteration curve')
    plt.legend(loc='lower right')
    plt.savefig('fig/method-acc.jpg',dpi=100)
    plt.show()