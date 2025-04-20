import numpy as np
from sklearn.linear_model import LogisticRegression
from L2 import *


class BRclass:
    def __init__(self):
        self.BR_cfl_train = []
        self.BR_cfl_test = []
        self.all_clf_train = []
        self.all_clf_test = []

    def BRC_train(self,X, Y):
        for i in range(Y.shape[1]):
            cfl = LogisticRegression()
            cfl.fit(X, Y[:, i])
            self.BR_cfl_train.append(cfl)
            self.all_clf_train.append(cfl)

    def BRC_test(self,X):
        for clf in self.BR_cfl_train:
            self.BR_cfl_test.append(clf.predict(X))
        return np.array(self.BR_cfl_test).T

    def train_BRC_train(self,X, Y):
        for i in range(Y.shape[1]):
            print(Y.shape[1])
            #        cfl = WeightedStackedEnsemble()
            cfl = WeightedStackedEnsemble()
            cfl.fit(X, Y[:, i])
            self.all_clf_test.append(cfl)

    def test_BRC_test(self,X, star, end):
        for i in range(star, end):
            clf = self.all_clf_train[i]
            self.BR_cfl_train.append(clf.predict(X))
        return np.array(self.BR_cfl_train).T

    def test_train_BRC_test(self, X):
        for clf in self.all_clf_test:
            self.BR_cfl_test.append(clf.predict(X))
        return np.array(self.BR_cfl_test).T

    def BRCLF_clear(self):
        self.BR_cfl_train = []

    def BRCLF_test_clear(self):
        self.BR_cfl_test = []

    def get(self):
        return self.BR_cfl_train
