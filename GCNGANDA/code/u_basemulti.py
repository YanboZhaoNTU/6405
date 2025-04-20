import numpy as np
from u_mReadData import ReadData
from time import time
from u_evaluation import evaluate
from u_savedata import saveResult
import warnings
warnings.filterwarnings('ignore')
from skmultilearn.problem_transform import BinaryRelevance,ClassifierChain,LabelPowerset
from u_base import base_cls

if __name__ == '__main__':
    numdata = 1
    datasnames = ["Birds","CAL500","CHD_49","Enron","Flags","Foodtruck","GnegativeGO","GpositiveGO","Image","Langlog"]
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    # rd = ReadData(datas=datasnames,genpath='data/')
    algname = 'ALG'

    '''train-test'''
    for dataIdx in range(numdata):
        print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
        start_time = time()
        learner = BinaryRelevance(classifier=base_cls())
        learner.fit(X,Y)
        mid_time = time()
        prediction = learner.predict_proba(Xt).todense()
        saveResult(datasnames[dataIdx], 'BR', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

        start_time = time()
        learner = ClassifierChain(classifier=base_cls())
        learner.fit(X,Y)
        mid_time = time()
        prediction = learner.predict_proba(Xt).todense()
        saveResult(datasnames[dataIdx], 'CC', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

        start_time = time()
        learner = LabelPowerset(classifier=base_cls())
        learner.fit(X,Y)
        mid_time = time()
        prediction = learner.predict_proba(Xt).todense()
        saveResult(datasnames[dataIdx], 'LP', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
