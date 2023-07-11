import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def euclidean_distance(self, x1, x2):
        
        return np.sqrt(np.sum((x1-x2)**2))

    def predict(self, X):
        results = []
        for x in X:
            # distances = []
            # distances = self.euclidean_distance(X_train, x)
            distances = np.sqrt(np.sum((X_train-x)**2))
            # for x_train in self.X_train:
            #     d = self.euclidean_distance(x_train, x)
            #     distances.append(d)

            nearest_neighbors = np.argsort(distances)[0:self.k]
            result = np.bincount(self.Y_train[nearest_neighbors])
            results.append([np.argmax(result), max(np.sort(distances)[0:self.k])])
        return results

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        Y_pred = np.asarray(Y_pred)[:,0].astype(dtype='int')
        accuracy = np.sum(Y_pred == Y) / len(Y)
        return accuracy