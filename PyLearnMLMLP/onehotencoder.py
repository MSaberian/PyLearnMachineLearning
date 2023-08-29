from sklearn.preprocessing import OneHotEncoder
import numpy as np

def mosaOneHotEncoder(x):
    len = x.size
    result = np.zeros((len,len))
    for i in range(len):
        result[i,x[i]] = 1
    return result

if __name__ == '__main__':
    sample = np.array([0, 3, 4, 1, 5, 2]).reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(sample)
    result = enc.transform(sample).toarray()
    print(result)
    print(mosaOneHotEncoder(sample))