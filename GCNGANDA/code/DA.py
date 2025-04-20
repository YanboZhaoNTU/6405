from sklearn.model_selection import train_test_split

from u_mReadData import *
from u_base import *
import numpy as np
from u_evaluation import *
from L2 import *
from weight import *
from BRclass import *
datasnames = ["20NG"]
rd = ReadData(datas=datasnames, genpath='data/')

X_train, Y_train, X_test, Y_test = rd.readData(0)
print(np.shape(X_train), np.shape(Y_train), np.shape(X_test), np.shape(Y_test))

X_train_label, X_temp, Y_train_label, y_temp = train_test_split(X_train, Y_train, test_size=0.8, random_state=42)
BRt = BRclass()
BRt.BRC_train(X_train_label, Y_train_label)
Plabel = BRt.BRC_test(X_temp)



Y_train_label = np.array(Y_train_label)
Plabel = np.array(Plabel)

print(np.shape(Y_train_label))
print(np.shape(Plabel))


Y_p_train = np.vstack([Y_train_label,Plabel])

TRt = BRclass()
TRt.BRC_train( X_train, Y_p_train)
test_result = TRt.BRC_test(X_test)
print(np.shape(test_result))
print(np.shape(Y_test))
eva = evaluate(test_result,Y_test)
print(eva)