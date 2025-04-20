from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import random
import numpy as np
from u_mReadData import ReadData
from time import time
from u_evaluation import evaluate
from u_savedata import *
import warnings
warnings.filterwarnings('ignore')
from skmultilearn.problem_transform import LabelPowerset
from sklearn.semi_supervised import SelfTrainingClassifier

def base_cls(mod='svm'):
    if(mod=='svm'):
        return SVC(probability=True, tol=1e-4, cache_size=200, max_iter=1000)
    elif(mod=='sgd'):
        return SGDClassifier(loss='log_loss')
    elif(mod=='lr'):
        return LogisticRegression()
    elif(mod=='bayes'):
        return GaussianNB()
    elif(mod=='dt'):
        return DecisionTreeClassifier()
    elif(mod=='nn'):
        return MLPClassifier(tol=1e-4, max_iter=200)
    elif(mod=='forest'):
        return RandomForestClassifier()
    else:
        return None

def fill1(Y):
    Y = np.array(Y)
    for j in range(np.shape(Y)[1]):
        if(np.sum(Y[:,j])==0):
            Y[0][j] = 1
    return Y

def randorder(Q):
    return np.array(random.sample(range(Q),Q))

def balanceorder(Y):
    order = np.argsort(np.sum(Y, 0))[::-1]
    return order

class Baser():
    def __init__(self, basemode='lr'):
        self.learner = base_cls(basemode)
    def fit(self,X,y,ins_weight=[]):
        self.output = -1
        if(np.sum(y)==len(y)):
            self.output = 1
        elif(np.sum(y)==0):
            self.output = 0
        else:
            if(len(ins_weight)==0):
                self.learner.fit(X,y)
            else:
                self.learner.fit(X,y,ins_weight)
    def predict_proba(self, Xt):
        if(self.output==-1):
            return self.learner.predict_proba(Xt)
        else:
            return np.zeros((len(Xt),2))+self.output

class BR():
    def __init__(self):
        self.baseLearner = []
        self.Q = 0
    def train(self,X,Y,idxs=[]):
        self.Q = np.shape(Y)[1]
        for j in range(self.Q):
            singleLearner = Baser()
            if(len(idxs)==0):
                singleLearner.fit(X,Y[:,j])
            else:
                idx = np.argwhere(idxs[j]).flatten()
                singleLearner.fit(X[idx],Y[:,j][idx])
            self.baseLearner.append(singleLearner)
    def test(self,Xt):
        prediction = []
        for j in range(self.Q):
            prediction_a = self.baseLearner[j].predict_proba(Xt)[:,1]
            prediction.append(prediction_a)
        return np.array(np.transpose(prediction))
    def test_a(self,Xt,k):
        prediction_a = self.baseLearner[k].predict_proba(Xt)[:,1]
        return prediction_a

class CC():
    def __init__(self,order=[]):
        self.baseLearner = []
        self.num_label = 0
        self.order = order
    def train(self,X,Y):
        X_train = np.array(X)
        self.num_label = np.shape(Y)[1]
        if(len(self.order)==1):
            self.order = randorder(self.num_label)
        if(len(self.order)==0):
            self.order = balanceorder(Y)
        for j in self.order:
            singleLearner = Baser()
            singleLearner.fit(X_train,Y[:,j])
            self.baseLearner.append(singleLearner)
            X_train = np.hstack((X_train, Y[:,[j]]))
    def test(self,Xt):
        Xt_train = np.array(Xt)
        prediction= [[] for _ in range(self.num_label)]
        for i in range(len(self.order)):
            j = self.order[i]
            prediction_a = self.baseLearner[i].predict_proba(Xt_train)[:,1]
            prediction[j] = prediction_a
            prediction_a = np.reshape(prediction_a, (-1, 1))
            Xt_train = np.hstack((Xt_train, prediction_a))
        return np.transpose(prediction)

class LP():
    def train(self,X,Y):
        learner = LabelPowerset(classifier=base_cls('lr'))
        learner.fit(X,fill1(Y))
        self.learner = learner
    def test(self,Xt):
        return self.learner.predict_proba(Xt).todense()

if __name__ == '__main__':
    numdata = 1
    datasnames = ["Yeast","CAL500","CHD_49","Enron","Flags","Foodtruck","GnegativeGO","GpositiveGO","Image","Langlog"]
    rd = ReadData(datas=datasnames,genpath='DATA/')
    # rd = ReadData(datas=datasnames,genpath='data/')
    algname = 'ALG'

    '''k-fold'''
    for dataIdx in range(numdata):
        print(dataIdx)
        k_fold,X_all,Y_all = rd.readData_CV(dataIdx)
        for train, test in k_fold.split(X_all, Y_all):
            X = X_all[train]
            Y = Y_all[train]
            Xt = X_all[test]
            Yt = Y_all[test]
            print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
            start_time = time()
            learner = BR()
            learner.train(X,Y)
            mid_time = time()
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], algname, evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

    '''k-fold with 1 result'''
    n_fold = 10
    for dataIdx in range(numdata):
        print(dataIdx)
        tmp_rst = np.zeros(13)
        k_fold,X_all,Y_all = rd.readData_CV(dataIdx,n_fold)
        for train, test in k_fold.split(X_all, Y_all):
            X = X_all[train]
            Y = Y_all[train]
            Xt = X_all[test]
            Yt = Y_all[test]
            print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
            start_time = time()
            learner = CC()
            learner.train(X,Y)
            mid_time = time()
            prediction = learner.test(Xt)
            tmp_rst += np.array(np.append(evaluate(prediction, Yt),[mid_time-start_time, time()-mid_time]))
        saveResult(datasnames[dataIdx], algname, tmp_rst/n_fold)

    '''train-test'''
    for dataIdx in range(numdata):
        print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
        start_time = time()
        learner = BR()
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], algname, evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))