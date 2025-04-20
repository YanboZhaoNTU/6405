from u_mReadData import *
from u_base import *
import numpy as np
from u_evaluation import *
BR_cfl_train = []
BR_cfl_test = []
save_OR_D = []
save_OR_L = []
datasnames = ["Yeast"]

result = np.array([])
Y_tr = np.array([])
X_te = np.array([])
Y_te = np.array([])

X_tr_list = []
Y_tr_list = []
X_te_list = []
Y_te_list = []
#训练阶段基学习器
all_clf_train = []
#训练阶段元学习器
all_clf_test = []

#元学习器的X_train
O_tr_T_list = np.array([])
#元学习器的Y_train
O_tr_Y_list = np.array([])

real_label = np.array([])
rd = ReadData(datas=datasnames, genpath='data/')
X_train, Y_train, X_test, Y_test = rd.readData(0)
print(np.shape(X_train), np.shape(Y_train), np.shape(X_test), np.shape(Y_test))


# (1629, 103) (1629, 14) (788, 103) (788, 14)
def BRC_train(X, Y):
    for i in range(Y.shape[1]):
        cfl = LogisticRegression()
        cfl.fit(X, Y[:, i])
        BR_cfl_train.append(cfl)
        all_clf_train.append(cfl)


def BRC_test(X):
    for clf in BR_cfl_train:
        BR_cfl_test.append(clf.predict(X))
    return np.array(BR_cfl_test).T


def train_BRC_train(X, Y):
    for i in range(Y.shape[1]):
        print(Y.shape[1])
        cfl = LogisticRegression()
        cfl.fit(X, Y[:, i])
        BR_cfl_train.append(cfl)
        all_clf_test.append(cfl)
    print(Y.shape[1])





for h in range(3):

    X_tr_list = []
    Y_tr_list = []
    X_te_list = []
    Y_te_list = []
    for i in range(1629):
        j = random.randint(0, 1628)

        X_tr_list.append(X_train[j])
        Y_tr_list.append(Y_train[j])
    for i in range(1629):
        j = random.randint(0, 1628)
        X_te_list.append(X_train[j])
        Y_te_list.append(Y_train[j])

    X_tr = np.array(X_tr_list)

    Y_tr = np.array(Y_tr_list)
    X_te = np.array(X_te_list)
    Y_te = np.array(Y_te_list)
    BRC_train(X_tr, Y_tr)
    #    BRC_test(X_te)
    #    O_tr_T_list = np.append(O_tr_T_list, BRC_test(X_te))
    result = BRC_test(X_te)
    print(result.shape)
    # 清空基学习器的训练分类器
    BR_cfl_train = []

train_BRC_train(result, Y_te)

###############################################
####训练阶段结束了
###############################################

BR_cfl_train = []
BR_cfl_test = []
save_OR_D = []
save_OR_L = []


result = np.array([])
Y_tr = np.array([])
X_te = np.array([])
Y_te = np.array([])

X_tr_list = []
Y_tr_list = []
X_te_list = []
Y_te_list = []

test_clf_num = 0

test_result = np.array([])

def test_BRC_test(X, star,end):
    BR_cfl_train = []
    for i in range(star,end):

        clf = all_clf_train[i]
        BR_cfl_train.append(clf.predict(X))
    return np.array(BR_cfl_train).T





def test_train_BRC_test(X):
    for clf in all_clf_test:
        BR_cfl_test.append(clf.predict(X))
    return np.array(BR_cfl_test).T


for i in range(788):
    j = random.randint(0, 787)
    X_tr_list.append(X_test[j])
    Y_tr_list.append(Y_test[j])

for i in range(788):
    j = random.randint(0, 787)
    X_te_list.append(X_test[j])
    Y_te_list.append(Y_test[j])

real_label = Y_te
fr = np.zeros((788, 14))

X_tr = np.array(X_tr_list)
Y_tr = np.array(Y_tr_list)
X_te = np.array(X_te_list)
Y_te = np.array(Y_te_list)

for h in range(3):
    star = test_clf_num
    end = test_clf_num + 14
    test_clf_num = test_clf_num + 14

    test_result = test_BRC_test(X_tr,star,end)
    fr = fr+test_result
fr = fr/3

correct_matrix = (fr == Y_tr_list)
num_correct = np.sum(correct_matrix)
eva = evaluate(fr, Y_tr_list)
print(eva)
accuracy = num_correct / (788*14)

