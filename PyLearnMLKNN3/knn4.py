from __future__ import print_function
import sys
import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        results = []
        for x in X:
            x_new = x.reshape(1,3)
            distances = np.sqrt(np.sum((self.X_train-x_new)**2, axis=1))
            dis = (self.X_train - x_new)**2
            disum = np.sum(dis, axis=1)
            disumsqr = np.sqrt(disum)
            nearest_neighbors = np.argsort(distances)[0:self.k]
            result = np.bincount(self.Y_train[nearest_neighbors])
            results.append(np.argmax(result))
            print(np.sort(distances)[0:self.k], nearest_neighbors, result, np.argmax(result))
            # print(f'Completed : {len(results)/len(X)*100:.2f}%', end='\r')
            # sys.stdout.flush()j
        print(results)
        return distances, 
        # return distances

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        Y_pred = np.asarray(Y_pred)[:,0].astype(dtype='int')
        accuracy = np.sum(Y_pred == Y) / len(Y)
        return accuracy

    # axis=1