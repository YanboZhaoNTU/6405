import numpy as np
from sklearn.linear_model import LogisticRegression, SGDRegressor
from L2 import *
class BRclass:
    def __init__(self):
        self.BR_cfl_train = []
        self.BR_cfl_test = []
        self.all_clf_train = []
        self.all_clf_test = []
        self.L2=[]
        self.L2_result=[]

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

    def clear(self):
        self.BR_cfl_test = []


    def train_L2(self,X, Y):
        for i in range(Y.shape[1]):

            #        cfl = WeightedStackedEnsemble()
            cfl = SGDRegressor(loss='squared_loss',      # 对应普通平方误差
                     penalty=None,              # 不加正则
                     max_iter=1000,            # 最大迭代次数
                     tol=1e-4,                 # 收敛阈值
                     eta0=0.01,                # 学习率初始值
                     learning_rate='constant')
            cfl.fit(X, Y[:, i])
            self.L2.append(cfl)

    def test_L2(self, X):
        for clf in self.L2:
            self.L2_result.append(clf.predict(X))
        return np.array(self.L2_result).T