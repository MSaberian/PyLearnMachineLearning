import random

def train_test_split(X, Y, test_size, shuffle=False):
    if len(X) != len(Y):
        raise ValueError('length of X_train and Y_train is not equal.')
    
    if shuffle:
        zipped = list(zip(X, Y))
        random.shuffle(zipped)
        X, Y = zip(*zipped)

    length = len(X)
    cut_point = int(length * test_size)
    X_test = X[:cut_point]
    Y_test = Y[:cut_point]
    X_train = X[cut_point:]
    Y_train = Y[cut_point:]

    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    list1 = [6, 4, 8, 9, 10]
    list2 = [1, 2, 3, 4, 5]
    X_train, X_test, Y_train, Y_test = train_test_split(list1, list2, 0.2, shuffle=True)
    print(X_train)
    print(X_test)
    print(Y_train)
    print(Y_test)