import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import LabelPowerset
from sklearn.metrics import accuracy_score, f1_score
from newData import *
from u_base import *
data = data()
data.readData()

X_train = data.getoriginTrainData()
Y_train = data.getoriginTrainLabel()
X_test = data.getoriginTestData()
Y_test = data.getoriginTestLabel()

new = CC()


new.train(X_train,Y_train)

# 使用LabelPowerset包装一个基础分类器


# 训练模型


# 进行预测
y_pred = new.test(X_test)
eva = evaluate(y_pred, Y_test)
# 评估模型性能

print(eva)

