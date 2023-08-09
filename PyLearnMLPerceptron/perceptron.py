import numpy as np
import sklearn.metrics as metrics

class Perceptron:
    def __init__(self, η1, η2, epochs):
        self.epochs = epochs
        self.W = np.random.rand(1, 2)
        self.b = np.random.rand(1, 1)
        self.losses_train = None
        self.losses_test = None
        self.η1 = η1
        self.η2 = η2
        self.ws = None
        self.bs = None

    def fit(self, X_train, Y_train):
        self.losses_train = []
        self.losses_test = []
        self.ws = np.zeros((self.epochs, X_train.shape[1]))
        self.bs = np.zeros(self.epochs)
        loss_train = 0

        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]):
                x = X_train[i]
                y = Y_train[i]

                y_pred = np.sum(x * self.W, axis=1) + self.b
                error = y - y_pred

                self.W += self.η1 * error * x  
                self.b += self.η2 * error

            Y_pred_train = np.sum(X_train * self.W, axis=1) + self.b
            loss_train = np.mean((Y_train - Y_pred_train) ** 2)
            self.losses_train.append(loss_train)
            self.ws[epoch,:] = self.W
            self.bs[epoch] = self.b

        return loss_train

    def predict(self, X_test):
        return np.sum(X_test * self.W, axis=1) + self.b

    def evaluate(self, X_test, Y_test):
        return metrics.mean_absolute_error(self.predict(X_test), Y_test)
        