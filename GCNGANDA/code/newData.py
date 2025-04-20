from u_mReadData import *
from u_base import *
from u_evaluation import *

class data:
    def __init__(self):
        self.num = 0
        self.X_train = np.array([])
        self.Y_train = np.array([])
        self.X_test = np.array([])
        self.Y_test = np.array([])
        self.train_X_tr = np.array([])
        self.train_Y_tr = np.array([])
        self.train_X_te = np.array([])
        self.train_Y_te = np.array([])
        self.train_X_tr_list = []
        self.train_Y_tr_list = []
        self.train_X_te_list = []
        self.train_Y_te_list = []
        self.test_X_tr = np.array([])
        self.test_Y_tr = np.array([])
        self.test_X_te = np.array([])
        self.test_Y_te = np.array([])
        self.test_X_tr_list = []
        self.test_Y_tr_list = []
        self.test_X_te_list = []
        self.test_Y_te_list = []

    def readData(self, datasnames=None, route=None):
        if datasnames is None:
            datasnames = ["Yeast"]

        if route is None:
            route = 'data/'

        rd = ReadData(datas=datasnames, genpath='data/')
        X_train, Y_train, X_test, Y_test = rd.readData(0)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def getoriginTrainData(self):
        return self.X_train
    def getoriginTrainLabel(self):
        return self.Y_train
    def getoriginTestData(self):
        return self.X_test
    def getoriginTestLabel(self):
        return self.Y_test

    def bagging_train_Data(self,num = None):
        if num is None:
            num = 1629

        for i in range(1629):
            j = random.randint(0, 1628)
            self.train_X_tr_list.append(self.X_train[j])
            self.train_Y_tr_list.append(self.Y_train[j])
        for i in range(1629):
            j = random.randint(0, 1628)
            self.train_X_te_list.append(self.X_train[j])
            self.train_Y_te_list.append(self.Y_train[j])
        self.train_X_tr = np.array(self.train_X_tr_list)
        self.train_Y_tr = np.array(self.train_Y_tr_list)
        self.train_X_te = np.array(self.train_X_te_list)
        self.train_Y_te = np.array(self.train_Y_te_list)

    def bagging_test_Data(self,num = None):
        if num is None:
            num = 788
        for i in range(787):
            j = random.randint(0, 787)

            self.test_X_tr_list.append(self.X_test[j])
            self.test_Y_tr_list.append(self.Y_test[j])
        self.test_Y_tr_list = np.array(self.test_Y_tr_list)


    def TrainX(self):
        return self.train_X_tr

    def TrainY(self):
        return self.train_Y_tr

    def PredX(self):
        return self.train_X_te

    def PredY(self):
        return self.train_Y_te

    def TestX(self):
        return self.test_X_tr_list

    def TestY(self):
        return self.test_Y_tr_list

    def clean(self):
        self.num = 0
        self.X_train = np.array([])
        self.Y_train = np.array([])
        self.X_test = np.array([])
        self.Y_test = np.array([])
        self.train_X_tr = np.array([])
        self.train_Y_tr = np.array([])
        self.train_X_te = np.array([])
        self.train_Y_te = np.array([])
        self.train_X_tr_list = []
        self.train_Y_tr_list = []
        self.train_X_te_list = []
        self.train_Y_te_list = []
        self.test_X_tr = np.array([])
        self.test_Y_tr = np.array([])
        self.test_X_te = np.array([])
        self.test_Y_te = np.array([])
        self.test_X_tr_list = []
        self.test_Y_tr_list = []
        self.test_X_te_list = []
        self.test_Y_te_list = []


















