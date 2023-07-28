import numpy as np

class LLS:
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.w = None

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.X_train.T, self.X_train)), self.X_train.T), self.Y_train)
        # self.w = np.linalg.inv(self.X_train.T.dot(self.X_train)).dot(self.X_train.T).dot(self.Y_train)
        return self.w

    def predict(self, X):
        return self.w * X