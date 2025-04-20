import numpy as np
from sklearn.linear_model import LogisticRegression

from L2 import WeightedStackedEnsemble

class mixclass:
    def __init__(self):
        self.name = 0
        self.all_clf_test = []
        self.circle_result = []

    def LP_train_BR_train(self, X, Y):
        for i in range(Y.shape[1]):

            #        cfl = WeightedStackedEnsemble()
            cfl = WeightedStackedEnsemble()
            cfl.fit(X, Y[:, i])
            self.all_clf_test.append(cfl)


    def BR_test_BRC_test(self, X):
        for clf in self.all_clf_test:
            self.circle_result.append(clf.predict(X))
        return np.array(self.circle_result).T