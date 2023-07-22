import numpy as np
# from __future__ import print_function
import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def predict(self, X):
        Y_pred = []
        for x in X:
            distances = self.euclidean_distance(x, self.X_train)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.Y_train[k_indices]
            Y_pred.append(np.bincount(k_nearest_labels).argmax())
            print(f'Completed : {len(Y_pred)/len(X)*100:.2f}%', end='\r')
            sys.stdout.flush()
        return np.array(Y_pred)

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        accuracy = np.sum(Y_pred == Y) / len(Y)
        return accuracy

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    knn_mamad = KNN(k=3)
    knn_mamad.fit(X_train, Y_train)
    accuracy = knn_mamad.evaluate(X_test, Y_test)
    print('Accuracy: ', accuracy)

    knn_sklearn = KNeighborsClassifier(n_neighbors=3)
    knn_sklearn.fit(X_train, Y_train)
    accuracy = knn_sklearn.score(X_test, Y_test)
    print('Accuracy: ', accuracy)