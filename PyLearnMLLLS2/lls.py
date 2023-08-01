import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

class LinearLeastSquare:
    def __init__(self):
        ...

    def fit(self,X_train,Y_train):

        # self.w=np.matmul(np.matmul(np.linalg.inv(np.matmul(self.X_train.T , self.X_train)), self.X_train.T), self.Y_train)
        self.w=inv(X_train.T @ X_train)@ X_train.T @ Y_train
        return self.w

    def predict(self,X_test):
        Y_pred= X_test@ self.w
        return Y_pred
    
    def evaluate(self,Y_test,Y_pred,metric):
        if metric== "mae":
            loss=np.sum(np.abs(Y_test-Y_pred))/len(Y_test)
        elif metric == "mse":
            loss=mse=np.sum((Y_test-Y_pred)**2)/len(Y_test)
        return loss
