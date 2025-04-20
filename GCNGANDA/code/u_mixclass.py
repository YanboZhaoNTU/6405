from mixclass import mixclass
from u_mReadData import *
from u_base import *
import numpy as np
from u_evaluation import *
from L2 import *
from weight import *
from BRclass import *
from LP import *
from CCclass import *

#导入数据
datasnames = ["Yeast"]
rd = ReadData(datas=datasnames, genpath='data/')
X_train, Y_train, X_test, Y_test = rd.readData(0)

result = np.array([])
Y_tr = np.array([])
X_te = np.array([])
Y_te = np.array([])
w_save = np.array([])
X_tr_list = []
Y_tr_list = []
X_te_list = []
Y_te_list = []

BRt = BRclass()
CCt = CCclass()
LPt = LabelPowersetLogistic()

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

BRt.BRC_train(X_tr, Y_tr)
result = BRt.BRC_test(X_te)
Y_or = Y_tr

BR_cfl_train = []
BRt.BRCLF_clear()



CCt.train(X_tr, Y_tr)
temp = CCt.CC_test(X_te,Y_te)
result = np.hstack([result,temp])
#Y_or = np.hstack([Y_or,Y_tr])


LPt.fit(X_tr, Y_tr)

temp = LPt.predict(X_te)
result = np.hstack([result,temp])
#Y_or = np.hstack([Y_or, Y_tr])
mc = mixclass()
mc.LP_train_BR_train(result, Y_te)
print(111)

# 测试阶段

BRt.BRCLF_clear()
BRt.BRCLF_test_clear()

Y_tr = np.array([])
X_te = np.array([])
Y_te = np.array([])
X_tr_list = []
Y_tr_list = []
X_te_list = []
Y_te_list = []
test_clf_num = 0
test_result = np.array([])
for i in range(788):
    j = random.randint(0, 787)

    X_tr_list.append(X_test[j])
    Y_tr_list.append(Y_test[j])

star = test_clf_num
end = test_clf_num + 14

X_tr = np.array(X_tr_list)
test_result = BRt.test_BRC_test(X_tr,star,end)


CCt.CCCLF_clear()
CCt.BRCLF_test_clear()




star = test_clf_num
end = test_clf_num + 14

X_tr = np.array(X_tr_list)
temp = CCt.test_BRC_test(X_tr,star,end)
test_result = np.hstack([test_result,temp])



test_clf_num = 0



star = test_clf_num
end = test_clf_num + 14
X_tr = np.array(X_tr_list)
temp = LPt.test_fit(X_tr,0)

test_result = np.hstack([test_result,temp])
real_label = np.array(Y_tr_list)
#real = np.hstack([real_label,real_label])
#real = np.hstack([real,real_label])
final_result = mc.BR_test_BRC_test(test_result)
eva = evaluate(final_result,real_label)
print(eva)
