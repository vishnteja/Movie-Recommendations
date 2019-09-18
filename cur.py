import numpy as np 
import pandas as pd
from scipy import sparse as sp
import util
import time

def select_cols(mat, k):
    # prob 1d array of probabilities of all columns
    # prob = mat.T.dot(mat)
    # prob = np.array(np.diagonal(prob))
    # denom = np.abs(prob).sum(axis = 0)
    # prob = prob/denom

    # C = np.zeros((mat.shape[0], k))
    # ind_cols = np.arange(0, prob.size)
    # c_ind = []
    # for i in range(k):
    #     rand_sel = np.random.choice(ind_cols, 1, p=prob)
    #     c_ind.append(rand_sel[0])
    #     C[:, i] = mat[:, rand_sel[0]]
    #     C[:, i] = C[:, i]/np.sqrt(k*prob[rand_sel[0]])
    
    col_sum = np.square(mat).sum(axis=0)
    total_sum = col_sum.sum()
    col_prob = col_sum / total_sum
    col_ind = np.random.choice(a=np.arange(mat.shape[1]), p=col_prob, replace=True, size=k)
    C = mat[:, col_ind]/np.array([np.sqrt(k*col_prob[col_ind]), ]*mat.shape[0])
    return C, col_ind

def select_rows(mat, k):

    # prob = mat.dot(mat.T)
    # prob = np.array(np.diagonal(prob))
    # denom = np.abs(prob).sum(axis=0)
    # prob = prob/denom
    # R = np.zeros((k, mat.shape[1]))
    # ind_rows = np.arange(0, prob.size)
    # r_ind = []
    # for i in range(k):
    #     rand_sel = np.random.choice(ind_rows, 1, p=prob)
    #     r_ind.append(rand_sel[0])
    #     R[i, :] = mat[rand_sel[0], :]
    #     R[i, :] = R[i, :]/np.sqrt(k*prob[rand_sel[0]])
    # r_ind = np.array(r_ind)

    row_sum = np.square(mat).sum(axis=1)
    total_sum = row_sum.sum()
    row_prob = row_sum / total_sum 
    row_ind = np.random.choice(a=np.arange(mat.shape[0]), p=row_prob, replace=True, size=k)
    R = mat[row_ind, :]/np.array([np.sqrt(k*row_prob[row_ind]), ]*mat.shape[1]).T
    return R, row_ind
    
def pseudoInverse(W, reduce = False, energy=0.9):
    # U = WP (W+)

    # W = X.Z.YT
    X, Z, YT = np.linalg.svd(W)
    
    # Z+ = reciprocal(Z)
    if(reduce==True):
        div = Z.sum()
        print(Z.sum())
        var = 0
        i = 0
        while(var/div<energy):
            var += Z[i] 
            i += 1
        Z = Z[:i].copy()
        X = X[:, :i]
        YT = YT[:i, :]

    ZP = np.reciprocal(Z)
    ZP = sp.spdiags(ZP, 0, ZP.size, ZP.size)
    ZP = ZP@ZP

    # W+ = Y.Z+.XT
    XT = X.T
    Y = YT.T
    
    # W+ = Y.Z+.XT
    WP = Y@ZP
    WP = WP@XT

    return WP

def main():
    # Read Matrix
    mat = util.readMatrix('./data/mod_data/ratings_mod_1000.csv')
    
    # Calculate number of rows or cols to select
    k = np.linalg.matrix_rank(mat)
    k = 4*k
    
    train, test = util.train_test_split(mat)
    
    user_bias = train.sum(1)/(train != 0).sum(1)
    user_bias_1d = user_bias.copy()
    
    user_bias = np.diag(user_bias)
    train_ones = train.copy()
    train_ones[train_ones!=0] = 1
    user_bias = user_bias@train_ones
    train -= user_bias
    
    start_time = time.time()

    C, col_ind = select_cols(train, k)
    R, _ = select_rows(train, k)
    W = R[:, col_ind]

    print("CUR Normal")
    U = pseudoInverse(W, reduce=True, energy=0.999)
    Pred = C@U@R
    Pred += user_bias_1d[:, np.newaxis]
    
    end_time = time.time()

    flatp = Pred[test.nonzero()].flatten()
    flatt = test[test.nonzero()].flatten()
    print(flatp)
    print(flatt)

    print(util.precision_topk(Pred, test, 25))
    print(util.rmse(Pred, test))
    # print(util.correlation_coefficient(flatp, flatt))
    print(end_time-start_time)

    print("CUR With Energy 90%")
    
    start_time = time.time()
    U = pseudoInverse(W, reduce=True, energy=0.9)
    Pred = C@U@R
    Pred += user_bias_1d[:, np.newaxis]
    end_time = time.time()

    flatp = Pred[test.nonzero()].flatten()
    flatt = test[test.nonzero()].flatten()
    print(flatp)
    print(flatt)
    print(end_time-start_time)

    print("Finished Execution")

if __name__ == "__main__":
    main()