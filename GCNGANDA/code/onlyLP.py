from newData import *
from newLP import *
LP = data()
LP.readData()
X_train = LP.getoriginTrainData()
Y_train = LP.getoriginTrainLabel()
X_test = LP.getoriginTestData()
Y_test = LP.getoriginTestLabel()
LPt = LabelPowersetLogistic()
LPt.fit(X_train,Y_train)
LPt.shape_train(X_test)
result = LPt.predict(X_test)
eva = evaluate(result,Y_test)
print(eva)