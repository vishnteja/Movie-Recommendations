'''
This file contains all the useful functions required by other
files.
- Train, Test Split
- RMSE & Spearmann Correlation
'''

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error as mse
import pandas as pd


def readMatrix(path):
    df = pd.read_csv(path)
    n_users = df.user_id.unique().shape[0]
    n_items = 27277 + 1

    urm = np.zeros(shape=(n_users, n_items), dtype=np.float32)
    for row in df.itertuples():
        urm[int(row[1]) - 1, int(row[3])] = float(row[2])

    return urm


def train_test_split(matrix, trate=0.1):
    test = np.zeros(matrix.shape)
    train = matrix.copy()
    for user in range(matrix.shape[0]):
        nzero_cols = matrix[user, :].nonzero()[0]
        # if nremove is zero then it defaults to 1
        nzero = nzero_cols.shape[0]
        nremove = int(np.around(np.multiply(nzero, trate)))
        if(nremove == 0):
            nremove = 1
        # print(nzero, nremove)
        test_ratings = np.random.choice(matrix[user, :].nonzero()[0],
                                        size=nremove,
                                        replace=False)
        train[user, test_ratings] = 0.
        for i in test_ratings:
            test[user, i] = matrix[user, i]

    # test = sp.csr_matrix(test)
    # Test and training are truly disjoint
    assert(((test + train) == matrix).all() == True), "Not Mutually Exclusive"
    return train, test


def time_waste(pred):
    lis = []

    for x in np.nditer(pred):
        lis.append(float(x))

    pred = np.array(lis)

    return pred


def rmse(pred, actual):
    if not pred.all:
        raise ValueError('Prediction List is Empty')

    pred = pred[actual.nonzero()].flatten()

    actual = actual[actual.nonzero()].flatten()

    return np.sqrt(mse(time_waste(pred), actual))


def precision_topk(pred, actual, k, mark=3.5):
    if not pred.all:
        raise ValueError('Prediciont List is Empty')

    pred = pred.copy()
    actual = actual.copy()
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()

    pred = time_waste(pred)
    correct = 0
    for i in range(k):
        maxi = pred.argmax()
        if(actual[maxi] > mark):
            correct += 1
        # Set it to max negative value
        pred[maxi] = -10000

    return float(correct / k)


def spearman_corr(pred, actual):
    if not pred.all:
        raise ValueError('Prediction List is Empty')

    pred = pred.copy()
    val = 0
    for user in pred:
        user = user[actual.nonzero()]
        user_act = actual[actual.nonzero()]

        n = user.size
        ind = np.array(np.arange(0, n))
        _, ind = np.sort(np.array([user, ind]))

        sumi = 0
        for i in ind:
            sumi += (user[i] - user_act[i])**2
        sumi = sumi * 6

        val += sumi / (n * (n**2 - 1))
        val = 1 - val

    val = val / n

    return val
