from u_base import *
import warnings
warnings.filterwarnings('ignore')

class CC2():
    def __init__(self,num_neighbor=12):
        self.baseLearner = []
        self.k = num_neighbor
    def train(self,X,Y,order=[]):
        self.num_label = np.shape(Y)[1]
        self.order = order
        if(len(order)==1):
            self.order = randorder(self.num_label)
        if(len(order)==0):
            self.order = balanceorder(Y)
        X_train = np.array(X)
        Y_train = np.array(Y)
        for i in self.order:
            singleLearner = Baser(self.k)
            singleLearner.fit(X_train,Y_train[:,i])
            self.baseLearner.append(singleLearner)
            X_train = np.hstack((X_train, Y_train[:,[i]]))
    def test(self,Xt):
        X_test = np.array(Xt)
        prediction= np.zeros((len(Xt),self.num_label))
        for i in range(len(self.order)):
            prediction_a = self.baseLearner[i].predict_proba(X_test)
            prediction[:,i] = prediction_a
            prediction_a = np.reshape(prediction_a, (len(Xt), 1))
            X_test = np.hstack((X_test, prediction_a))
        return reorderY(prediction,self.order),prediction
    def test_a(self,Xt):
        # X_test = np.array(Xt)
        prediction= np.zeros((len(Xt),self.num_label))
        for meta in range(len(Xt)):
            thisinstance = np.array(Xt[meta])
            for i in range(len(self.order)):
                prediction_a = self.baseLearner[i].predict_proba([thisinstance])[0]
                prediction[meta,i] = prediction_a
                thisinstance = np.append(thisinstance,prediction_a)
        return reorderY(prediction,self.order),prediction
    def test1(self,Xt):
        X_test = np.array(Xt)
        prediction= np.zeros((len(Xt),self.num_label))
        for i in range(len(self.order)):
            prediction_a = self.baseLearner[i].predict_proba(X_test)
            prediction[:,i] = prediction_a
            prediction_a = np.round(np.reshape(prediction_a, (len(Xt), 1)))
            X_test = np.hstack((X_test, prediction_a))
        return reorderY(prediction,self.order)
    def test_TWCC(self,Xt):
        numtest = len(Xt)
        numlabel = self.num_label
        prediction_twcc = np.zeros((numtest,numlabel))
        counting = 0
        baselearner = self.baseLearner
        chainorder = self.order
        for i in range(numtest):
            thisinstance = np.array(Xt[i])
            count_a,predict_a = self.test_TWCC_a(thisinstance,chainorder,baselearner)
            counting = counting+count_a
            prediction_twcc[i] = predict_a
        print('counting==',counting)
        savemat([[counting]],file='counting')
        return np.array(prediction_twcc)
    def test_TWCC_a(self,testinstance,chainorder,baseLearner):
        numlabel = len(chainorder)
        prediction_twcc = np.zeros(numlabel)
        counting = 0
        thisinstance = np.array(testinstance)
        for j in range(numlabel):
        # for j in chainorder:
            prediction_ij = baseLearner[j].predict_proba([thisinstance])[0]
            p = np.abs(prediction_ij-0.5)+0.5
            if(p<2/3 and j<numlabel-1):
                tmpinstance1 = np.append(thisinstance, prediction_ij)
                tmpinstance2 = np.append(thisinstance, 1-prediction_ij)
                prediction_ijj1 = baseLearner[j+1].predict_proba([tmpinstance1])[0]
                prediction_ijj2 = baseLearner[j+1].predict_proba([tmpinstance2])[0]
                q1 = np.abs(prediction_ijj1-0.5)+0.5
                q2 = np.abs(prediction_ijj2-0.5)+0.5
                if(p*q1>=(1-p)*q2):
                    thisinstance = tmpinstance1
                    prediction_twcc[j] = prediction_ij
                else:
                    counting = counting+1
                    thisinstance = tmpinstance2
                    prediction_twcc[j] = 1-prediction_ij
                    if(prediction_ij==0.5):
                        prediction_twcc[j] = 0.49
            else:
                thisinstance = np.append(thisinstance, prediction_ij)
                prediction_twcc[j] = prediction_ij
        return counting,reorderY(np.array([prediction_twcc]),chainorder)

if __name__=="__main__":
    numdata = 7
    datasnames = ["HumanGO","Image","Medical","PlantGO","Tmc2007_500","Yeast","Yelp"]
    rd = ReadData(datas=datasnames,genpath='data/')
    for dataIdx in range(5,numdata):
        X,Y,Xt,Yt = rd.readData(dataIdx)
        order = balanceorder(Y)

        k = int(len(X)/2)
        mLearner = CC2(num_neighbor=k)
        start_time = time()
        mLearner.train(X,Y,order)
        mid_time = time()
        output,prediction = mLearner.test_a(Xt)
        resolveResult(datasnames[dataIdx], 'BCC', evaluate(output, Yt), (mid_time-start_time), (time()-mid_time))

        mid_time = time()
        prediction = mLearner.test_TWCC(Xt)
        resolveResult(datasnames[dataIdx], 'TWCC', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
