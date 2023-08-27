import numpy as np

class MLP:
    def __init__(self, MLPstructure):
        self.D_in = MLPstructure.D_in
        self.H1 = MLPstructure.H1
        self.H2 = MLPstructure.H2
        self.D_out = MLPstructure.D_out
        self.η = MLPstructure.η

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

    def fit(self, X_train, Y_train, X_test, Y_test, epochs=200):

        W1, W2, W3 = np.random.randn(self.D_in, self.H1), np.random.randn(self.H1, self.H2), np.random.randn(self.H2, self.D_out)
        B1, B2, B3 = np.random.randn(self.H1), np.random.randn(self.H2), np.random.randn(self.D_out)

        for epoch in range(epochs):

            Y_pred = []
            for x, y in zip(X_train, Y_train):

                # forward
                x = x.reshape(-1, 1)

                # layer 1
                net1 = x.T @ W1 + B1
                out1 = self.sigmoid(net1)

                # layer 2
                net2 = out1 @ W2 + B2
                out2 = self.sigmoid(net2)

                # layer 3
                net3 = out2 @ W3 + B3
                out3 = self.softmax(net3)

                y_pred = out3
                Y_pred.append(y_pred.T)

                # back propagation

                # layer 3
                error = -2 * (y - y_pred)
                grad_W3 = out2.T @ error
                grad_B3 = error

                # layer 2
                error = error @ W3.T * out2 * (1 - out2)
                grad_W2 = out1.T @ error
                grad_B2 = error

                # layer 1
                error = error @ W2.T * out1 * (1 - out1)
                grad_W1 = x @ error
                grad_B1 = error

                # update

                # layer 1
                W1 = W1 - self.η * grad_W1
                B1 = B1 - self.η * grad_B1
                
                # layer 2
                W2 = W2 - self.η * grad_W2
                B2 = B2 - self.η * grad_B2

                # layer 3
                W3 = W3 - self.η * grad_W3
                B3 = B3 - self.η * grad_B3

            Y_pred = np.array(Y_pred).reshape(-1, 10)
            loss_train = self.root_mean_squired_error(Y_pred, Y_train)
            acc_train = np.mean(np.argmax(Y_pred, axis=1) == np.argmax(Y_train, axis=1))
            
            # test

            Y_pred = []
            for x, y in zip(X_test, Y_test):

                # forward
                x = x.reshape(-1, 1)

                # layer 1
                net1 = x.T @ W1 + B1
                out1 = self.sigmoid(net1)

                # layer 2
                net2 = out1 @ W2 + B2
                out2 = self.sigmoid(net2)

                # layer 3
                net3 = out2 @ W3 + B3
                out3 = self.softmax(net3)

                y_pred = out3
                Y_pred.append(y_pred.T)

            Y_pred = np.array(Y_pred).reshape(-1, 10)
            loss_test = self.root_mean_squired_error(Y_pred, Y_test)
            acc_test = np.mean(np.argmax(Y_pred, axis=1) == np.argmax(Y_test, axis=1))

            print('loss train:', loss_train, 'acc train:', acc_train)
            print('loss test:', loss_test, 'acc test:', acc_test)

        print('train completed!')
