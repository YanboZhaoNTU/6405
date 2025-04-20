import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from L2 import WeightedStackedEnsemble
from u_mReadData import *
class CCclass:
    def __init__(self):
        self.classifiers = []
        self.CC_cfl_test = []
        self.all_clf_train = []
        self.all_clf_test = []
        self.basic_result = []
        self.circle_result = []
        self.num = 0

    def train(self, X, Y):
        L = Y.shape[1]
        N = X.shape[0]
        D_j_prime_x = []
        D_j_prime_y = []

        for j in range(L):
            # D'j

            # for (x, y) ∈ D

                #do x' ← [x1,...,xd ,y1,...,yj−1]
            if j == 0:
                x_prime = np.array(X)
            else:
                x_prime = np.hstack((x_prime, Y[:,[j-1]]))
                # Dj' ← Dj ∪ (x' ,yj )
            D_j_prime_x = x_prime
            D_j_prime_y = Y

            D_j_prime_x = np.array(D_j_prime_x)
            D_j_prime_y = np.array(D_j_prime_y)
            # train hj to predict binary relevance of yj
            # P(y=1∣X)= 1/ (1+a) a = e的-（wX+b）次方
            clf = LogisticRegression()
            clf.fit(D_j_prime_x, D_j_prime_y[:,j])
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