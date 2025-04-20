import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from L2 import WeightedStackedEnsemble
from u_mReadData import *

class newCCclass:
    def __init__(self):
        self.classifiers = []
        self.CC_cfl_test = []
        self.all_clf_train = []
        self.all_clf_test = []
        self.basic_result = []
        self.circle_result = []
        self.num = 0


    def train(self, X, Y, list):

        L = Y.shape[1]
        N = X.shape[0]
        D_j_prime_x = []
        D_j_prime_y = []
        x_prime = np.array(X)
        for j in list:
            D_j_prime_x = x_prime
            D_j_prime_y = Y

            D_j_prime_x = np.array(D_j_prime_x)
            D_j_prime_y = np.array(D_j_prime_y)
            # train hj to predict binary relevance of yj
            # P(y=1∣X)= 1/ (1+a) a = e的-（wX+b）次方
            clf = LogisticRegression()
            clf.fit(D_j_prime_x, D_j_prime_y[:,j])

            x_prime = np.hstack((x_prime, Y[:, [j]]))
            self.classifiers.append(clf)
            self.all_clf_train.append(clf)
            self.num = self.num + 1



    def CC_test(self, X, Y):
        LT = Y.shape[1]
        y_hat = []
        # for j = 1,...,L
        for j in range(LT):
            D_j_prime_x = []
            # do x' ← [x1,...,xd , yˆ1,..., yˆj−1]

            if j == 0:
                x_prime = X
            else:
                cnm = np.array(y_hat)
                cnm = cnm.reshape(-1, 1)

                x_prime = np.column_stack((x_prime, cnm))
            D_j_prime_x.append(x_prime)

            D_j_prime_x = np.array(D_j_prime_x)
            p = D_j_prime_x[0]
            # do x' ← [x1,...,xd , yˆ1,..., yˆj−1]
            y_pred_j = self.classifiers[j].predict(p)
            y_hat = []
            y_hat.append(y_pred_j)
            y_hat = np.array(y_hat).flatten()
            y_hat = y_hat.tolist()
            # return y
            self.CC_cfl_test.append(y_hat)
        return np.array(self.CC_cfl_test).T
# 训练元学习器
    def CC_train_BR_train(self, X, Y):
        for i in range(Y.shape[1]):

            #        cfl = WeightedStackedEnsemble()
            cfl = WeightedStackedEnsemble()
            cfl.fit(X, Y[:, i])
            self.all_clf_test.append(cfl)
# 测试基学习器
    def test_BRC_test(self, X,r):

        y_hat = []
        x_prime = np.array(X)
        j = 0
        for i in range(r):

            D_j_prime_x = []
            # do x' ← [x1,...,xd , yˆ1,..., yˆj−1]

            if j == 0:
                x_prime = np.array(X)
                j = j+1
            else:
                cnm = np.array(y_hat)
                cnm = cnm.reshape(-1, 1)

                x_prime = np.column_stack((x_prime, cnm))
            D_j_prime_x.append(x_prime)
            D_j_prime_x = np.array(D_j_prime_x)
            p = D_j_prime_x[0]

            # do x' ← [x1,...,xd , yˆ1,..., yˆj−1]
            q = self.all_clf_train[i]
            y_pred_j = self.all_clf_train[i].predict(p)
            y_hat = []
            # return y
            y_hat.append(y_pred_j)
            y_hat = np.array(y_hat).flatten()
            y_hat = y_hat.tolist()

            self.basic_result.append(y_hat)

        return np.array(self.basic_result).T

    def BR_test_BRC_test(self, X):
        for clf in self.all_clf_test:
            self.circle_result.append(clf.predict(X))
        return np.array(self.circle_result).T


    def CCCLF_clear(self):
        self.classifiers = []

    def BRCLF_test_clear(self):
        self.CC_cfl_test = []





