import numpy as np
import time
import util

def findSimilarity(ratings, epsilon=1e-7):
    # epsilon -> small number for handling dived-by-zero errors
    similarity = ratings.dot(ratings.transpose())
    similarity += epsilon
    normalised = np.array([np.sqrt(np.diagonal(similarity))])
    return (similarity / normalised / normalised.T)

def predict_topk(ratings, similarity, k=40):
    predicted = np.zeros(ratings.shape)
    for i in range(ratings.shape[0]):
        top_k_users = [np.argsort(similarity[:, i])[:-k-1:-1]]
        for j in range(ratings.shape[1]):
            predicted[i, j] = similarity[i, :][tuple(top_k_users)].dot(ratings[:, j][tuple(top_k_users)])
            predicted[i, j] /= np.sum(np.abs(similarity[i, :][tuple(top_k_users)]))
    return predicted

def predict_topk_nobias(ratings, k=40):
    predicted = np.zeros(ratings.shape)
    train = ratings.copy()

    #ratings = (ratings - bias[:, np.newaxis]).copy()
    # ratings = (ratings.transpose() - bias).transpose().copy()


    user_bias = train.sum(1) / (train != 0).sum(1)
    user_bias_1d = user_bias.copy()

    user_bias = np.diag(user_bias)
    train_ones = train.copy()
    train_ones[train_ones != 0] = 1
    user_bias = user_bias @ train_ones
    train -= user_bias

    similarity = findSimilarity(train)

    for i in range(ratings.shape[0]):
        top_k_users = [np.argsort(similarity[1:, i])[:-k-1:-1]]
        for j in range(ratings.shape[1]):
            predicted[i, j] = similarity[i, :][tuple(top_k_users)].dot(ratings[:, j][tuple(top_k_users)])
            predicted[i, j] /= np.sum(np.abs(similarity[i, :][tuple(top_k_users)]))
    predicted = (predicted.transpose() + user_bias_1d -1.2).transpose()
    return predicted

def main():

    matrix = util.readMatrix('./data/mod_data/ratings_mod_1000.csv')
    train, test = util.train_test_split(matrix)

    # Dense Matrix
    usim = findSimilarity(train)

    start_time = time.time()
    #Normal CF
    upred = predict_topk(train, usim)
    np.around(upred, decimals=3)
    upred_save = upred[test.nonzero()].flatten()
    np.savetxt('CFpredwithout.csv', upred_save, delimiter=',')
    print('User Based with bias RMSE : '+str(util.rmse(upred, test)))
    print('User based with bias Precision top k: ' + str(util.precision_topk(upred, test, 25)))
    #print('User based with bias SC: ' + str(util.spearman_corr(upred, test)))
    print('Took ' + str(time.time() - start_time))

    next_time = time.time()
    # CF with Baseline
    upred1 = predict_topk_nobias(train)
    np.around(upred1, decimals=3)
    upred1_save = upred1[test.nonzero()].flatten()
    test_save = test[test.nonzero()].flatten()
    np.savetxt('CFpred.csv', upred1_save, delimiter=',')
    np.savetxt('CFtest.csv', test_save, delimiter=',')
    print('User Based without bias RMSE : '+str(util.rmse(upred1, test)))
    print('User based without bias Precision top k: ' + str(util.precision_topk(upred1, test, 25)))
    #print('User based without bias SC: ' + str(util.spearman_corr(upred1, test)))
    print('Took ' + str(time.time() - next_time))

if __name__ == '__main__':
    main()