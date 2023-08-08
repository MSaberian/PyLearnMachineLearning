import numpy as np
import sklearn.metrics as metrics

class Perceptron:
    def __init__(self, η1, η2, epochs):
        self.epochs = epochs
        self.W = np.random.rand(1, 1)
        self.b = np.random.rand(1, 1)
        self.losses_train = None
        self.losses_test = None
        self.η1 = η1
        self.η2 = η2

    def fit(self, X_train, Y_train):
        self.losses_train = []
        self.losses_test = []
        loss_train = 0

        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]):
                x = X_train[i]
                y = Y_train[i]

                y_pred = x * self.W + self.b
                error = y - y_pred

                self.W += self.η1 * error * x  
                self.b += self.η2 * error

            Y_pred_train = X_train * self.W + self.b
            loss_train = np.mean((Y_train - Y_pred_train) ** 2)
            self.losses_train.append(loss_train)

            # Y_pred_test = X_test * self.W + self.b
            # loss_test = np.mean((Y_test - Y_pred_test) ** 2)
            # self.losses_test.append(loss_test)
        return loss_train

    def predict(self, X_test):
        return X_test * self.W + self.b

    def evaluate(self, X_test, Y_test):
        return metrics.mean_absolute_error(self.predict(X_test), Y_test)
        