import numpy as np
from tqdm import tqdm

class MLP:
    def __init__(self, MLPstructure):
        self.D_in = MLPstructure.D_in
        self.H1 = MLPstructure.H1
        self.H2 = MLPstructure.H2
        self.D_out = MLPstructure.D_out
        self.η = MLPstructure.η
        self.W1 = None
        self.W2 = None
        self.W3 = None
        self.B1 = None
        self.B2 = None
        self.B3 = None
        self.Losss_train = []
        self.losss_test = []
        self.accs_train = []
        self.accs_test = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps)

    def cross_entropy_error(self, Y_pred, Y_gt):
        delta = 1e-7
        return -np.sum(Y_gt * np.log(Y_pred + delta))

    def root_mean_squired_error(self, Y_pred, Y_gt):
        return np.sqrt(np.mean((Y_pred - Y_gt) ** 2))

    def predict(self, x):
        # layer 1
        net1 = x.T @ self.W1 + self.B1
        out1 = self.sigmoid(net1)

        # layer 2
        net2 = out1 @ self.W2 + self.B2 
        out2 = self.sigmoid(net2)

        # layer 3
        net3 = out2 @ self.W3 + self.B3
        out3 = self.softmax(net3)

        return out3

    def fit(self, X_train, Y_train, X_test, Y_test, epochs=200):

        self.W1, self.W2, self.W3 = np.random.randn(self.D_in, self.H1), np.random.randn(self.H1, self.H2), np.random.randn(self.H2, self.D_out)
        self.B1, self.B2, self.B3 = np.random.randn(self.H1), np.random.randn(self.H2), np.random.randn(self.D_out)

        self.Losss_train = []
        self.losss_test = []
        self.accs_train = []
        self.accs_test = []

        for epoch in tqdm(range(epochs)):

            Y_pred = []
            for x, y in zip(X_train, Y_train):

                # forward
                x = x.reshape(-1, 1)

                # layer 1
                net1 = x.T @ self.W1 + self.B1
                out1 = self.sigmoid(net1)

                # layer 2
                net2 = out1 @ self.W2 + self.B2
                out2 = self.sigmoid(net2)

                # layer 3
                net3 = out2 @ self.W3 + self.B3
                out3 = self.softmax(net3)

                y_pred = out3
                Y_pred.append(y_pred.T)

                # back propagation

                # layer 3
                error = -2 * (y - y_pred)
                grad_W3 = out2.T @ error
                grad_B3 = error

                # layer 2
                error = error @ self.W3.T * out2 * (1 - out2)
                grad_W2 = out1.T @ error
                grad_B2 = error

                # layer 1
                error = error @ self.W2.T * out1 * (1 - out1)
                grad_W1 = x @ error
                grad_B1 = error

                # update

                # layer 1
                self.W1 = self.W1 - self.η * grad_W1
                self.B1 = self.B1 - self.η * grad_B1
                
                # layer 2
                self.W2 = self.W2 - self.η * grad_W2
                self.B2 = self.B2 - self.η * grad_B2

                # layer 3
                self.W3 = self.W3 - self.η * grad_W3
                self.B3 = self.B3 - self.η * grad_B3

            Y_pred = np.array(Y_pred).reshape(-1, 10)
            loss_train = self.root_mean_squired_error(Y_pred, Y_train)
            acc_train = np.mean(np.argmax(Y_pred, axis=1) == np.argmax(Y_train, axis=1))
            
            # test

            Y_pred = []
            for x, y in zip(X_test, Y_test):

                # forward
                x = x.reshape(-1, 1)

                # layer 1
                net1 = x.T @ self.W1 + self.B1
                out1 = self.sigmoid(net1)

                # layer 2
                net2 = out1 @ self.W2 + self.B2
                out2 = self.sigmoid(net2)

                # layer 3
                net3 = out2 @ self.W3 + self.B3
                out3 = self.softmax(net3)

                y_pred = out3
                Y_pred.append(y_pred.T)

            Y_pred = np.array(Y_pred).reshape(-1, 10)
            loss_test = self.root_mean_squired_error(Y_pred, Y_test)
            acc_test = np.mean(np.argmax(Y_pred, axis=1) == np.argmax(Y_test, axis=1))

            self.Losss_train.append(loss_train)
            self.losss_test.append(loss_test)
            self.accs_train.append(acc_train)
            self.accs_test.append(acc_test)
            # print('loss train:', loss_train, 'acc train:', acc_train)
            # print('loss test:', loss_test, 'acc test:', acc_test)

        print('train completed!')
