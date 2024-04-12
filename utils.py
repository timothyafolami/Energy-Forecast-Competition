import numpy as np
def create_dataset(X, y, look_back=3):
    dataX, dataY = [], []
    # making sure the data is in the right shape and it is a numpy array
    X = np.array(X)
    y = np.array(y)
    for i in range(len(X)-look_back-1):
        a = X[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(y[i + look_back])
    return np.array(dataX), np.array(dataY)

