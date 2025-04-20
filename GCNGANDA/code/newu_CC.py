from newData import *
from newCC import *

data = data()
data.readData()
CC = CCclass()
X_train = data.getoriginTrainData()
Y_train = data.getoriginTrainLabel()
X_test = data.getoriginTestData()
Y_test = data.getoriginTestLabel()
CC.train(X_train,Y_train)
result = CC.CC_test(X_test,Y_test)

eva = evaluate(result, Y_test)
print(eva)