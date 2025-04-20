import numpy as np
from u_base import *
from newLP import *
from newData import *
from newBR import *

data = data()
data.readData()

rt = np.array([])
rte = np.array([])
X_train = data.getoriginTrainData()
Y_train = data.getoriginTrainLabel()
X_test = data.getoriginTestData()
Y_test = data.getoriginTestLabel()
num = 0
BR = BRclass()
BR.BRC_train(X_train, Y_train)
tr_result = BR.BRC_test(X_train)
BR.clear()
te_result = BR.BRC_test(X_test)

rt = tr_result
rte = te_result

CC = CC()
CC.train(X_train, Y_train)
tr_result = CC.test(X_train)
te_result = CC.test(X_test)

rt = np.hstack([rt, tr_result])
rte = np.hstack([rte, te_result])

LP = LP()

LP.train(X_train, Y_train)
tr_result = LP.test(X_train)
te_result = LP.test(X_test)

rt = np.hstack([rt, tr_result])
rte = np.hstack([rte, te_result])

BRE = BRclass()
BRE.train_L2(rt,Y_train)
result = BRE.test_L2(rte)
eva = evaluate(result, Y_test)
print(eva)