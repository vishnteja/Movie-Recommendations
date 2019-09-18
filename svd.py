import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as ssl
import csv
from scipy import spatial
# from sparsesvd import sparsesvd
import math as mt
import warnings
import util_SVD as util
import colab
warnings.filterwarnings('ignore')

# constants defining the dimensions of our User Rating Matrix (URM)
MAX_PID = 27278
MAX_UID = 1000

FILE_NAME = 'ratings_mod_1000.csv'


def readUrm():
    # urm_unsplit = np.zeros(shape=(MAX_UID + 1, MAX_PID), dtype=np.float32)
    # with open(FILE_NAME) as trainFile:
    #     urmReader = csv.reader(trainFile, delimiter=',')
    #     count = 0
    #     for row in urmReader:

    #         if int(row[2]) <= MAX_PID and int(row[0]) <= MAX_UID:
    #             urm_unsplit[int(row[0]), int(row[2])] = float(row[1])
    #             count += 1
    #         # else:
    #         #     break

    # print(f'Line Count {count}')
    # print(urm)

    urm_unsplit = util.readMatrix(FILE_NAME)
    urm, test_matrix = util.train_test_split(urm_unsplit)

    user_bias = urm.sum(1) / (urm != 0).sum(1)
    # user_bias[0] = 0

    user_bias_1d = user_bias.copy()
    # urm = (urm[urm.nonzero()] - user_bias[:, np.newaxis]).copy()
    # print(user_bias)
    user_bias = np.diag(user_bias)
    # print(user_bias)

    urm_ones = urm.copy()
    urm_ones[urm_ones != 0] = 1
    # print(urm_ones)

    user_bias = user_bias @ urm_ones
    # print(user_bias)
    # print(urm)
    urm = urm - user_bias
    # print(urm)

    K = np.linalg.matrix_rank(urm)
    urm = csc_matrix(urm, dtype=np.float32)
    # print('asdsa')
    # print('URM')
    # print(urm)
    # print('sadasd')

    return user_bias_1d, urm, test_matrix, K


def computeSVD(urm, K):

    U, s, Vt = getSVD(urm, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        # S[i, i] = mt.sqrt(s[i])
        S[i, i] = s[i]

    U = csc_matrix(np.transpose(U), dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)

    return U, S, Vt


def getSVD(urm, K):

    # SPARSE MATRIX CODE

    A = urm
    AT = A.transpose()

    ATA = AT @ A
    # print(ATA)
    eigs, V = ssl.eigs(ATA, K)
    # print(eigs)
    eig_vals = []

    for x in eigs:
        if x != 0:
            eig_vals.append(x.real)
    eig_vals = np.array(eig_vals, dtype=np.float32)
    eig_vals[::-1].sort()

    V = V.astype(np.float32)

    VT = V.transpose()
    eig_vals = np.sqrt(eig_vals)

    # print(eig_vals)

    S = np.diag(eig_vals)

    Si = np.linalg.inv(S)

    A = A.todense()
    U = A @ V
    U = U @ Si
    U = U.transpose()
    # U = np.negative(U)
    # V = np.negative(V)

    return U, eig_vals, VT


def computeEstimatedRatings(urm, U, S, Vt, user_bias, K):
    rightTerm = S * Vt
    # print(U.shape)
    # similarity = colab.predict_topk_nobias(U)

    estimatedRatings = U * rightTerm
    estimatedRatings = estimatedRatings + user_bias[:, np.newaxis]

    return estimatedRatings


def main():

    np.set_printoptions(suppress=True)
    print('Reading Data Set..')
    user_bias, urm, test_matrix, K = readUrm()

    # K = 50
    # print(K)
    print('Computing SVD...')
    U, S, Vt = computeSVD(urm, K)
    print('Getting Predictions')
    uTest = computeEstimatedRatings(
        urm, U, S, Vt, user_bias, K)

    # print(uTest)

    print(util.rmse(uTest, test_matrix))
    print(util.precision_topk(uTest, test_matrix, 25))


if __name__ == '__main__':
    main()
