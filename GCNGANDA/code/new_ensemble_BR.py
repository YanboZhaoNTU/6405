from newData import *
from newBR import *

data = data()
data.readData()
BR = BRclass()
X_train = data.getoriginTrainData()
Y_train = data.getoriginTrainLabel()
X_test = data.getoriginTestData()
Y_test = data.getoriginTestLabel()
BR.BRC_train(X_train,Y_train)
result = BR.BRC_test(X_test)

eva = evaluate(result, Y_test)
print(eva)