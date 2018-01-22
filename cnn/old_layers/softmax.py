import numpy as np

class Softmax(object):
    def __init__(self, shape):
        self.softmax = np.zeros(shape[0])
        self.eta = np.zeros(shape[0])

    def cal_loss(self, prediction, label):
        self.label = label
        self.prediction = prediction

        self.predict(prediction)
        self.loss = np.log(np.sum(np.exp(prediction))) - prediction[label]
        return self.loss

    def predict(self, prediction):
        prediction = prediction - np.max(prediction)
        exp_prediction = np.exp(prediction)
        self.softmax = exp_prediction/np.sum(exp_prediction)
        return self.softmax

    def gradient(self):
        self.eta = self.softmax.copy()
        self.eta[self.label] -= 1
        return self.eta
