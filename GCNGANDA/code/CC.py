import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from u_mReadData import *


######################################

#错的

#####################################


datasnames = ["Yeast"]
rd = ReadData(datas=datasnames,genpath='data/')

num = 0
for dataIdx in range(0,1):

    # X,Y,Xt,Yt = rd.readData(dataIdx)
    k_fold,X_all,Y_all = rd.readData_CV(dataIdx)

    for train, test in k_fold.split(X_all, Y_all):
        #
        x_train = X_all[train]
        y_train = Y_all[train]

        Xt = X_all[test]
        Yt = Y_all[test]
#训练代码
        #保存分类器
        classifiers = []

        L = y_train.shape[1]
        N = x_train.shape[0]
        # TRAINING(D = {(x1, y1),...,(xN , yN )})
        # for j = 1,...,L
        # do |> the j th binary transformation and training
        for j in range(L):
            # D'j
            D_j_prime_x = []
            D_j_prime_y = []
            # for (x, y) ∈ D
            for i in range(N):
                #do x' ← [x1,...,xd ,y1,...,yj−1]
                if j == 0:
                    x_prime = x_train[i]
                else:
                    x_prime = np.concatenate((x_train[i], y_train[i,:j]))
                # Dj' ← Dj ∪ (x' ,yj )
                D_j_prime_x.append(x_prime)
                D_j_prime_y.append(y_train[i,j])

            D_j_prime_x = np.array(D_j_prime_x)
            D_j_prime_y = np.array(D_j_prime_y)
            # train hj to predict binary relevance of yj
            # P(y=1∣X)= 1/ (1+a) a = e的-（wX+b）次方
            clf = LogisticRegression()
            clf.fit(D_j_prime_x, D_j_prime_y)
            classifiers.append(clf)










        # 测试代码
        # classifiers |> global h = (h1,...,hL)
        x_test = Xt
        y_test = Yt



        LT = y_test.shape[1]


        NT = x_test.shape[0]
        # y ←[ˆy1,..., yˆL]
        y_hat = []

        # for j = 1,...,L
        for j in range(LT):
            D_j_prime_x = []
            # do x' ← [x1,...,xd , yˆ1,..., yˆj−1]
            for i in range(NT):
                if j == 0:
                    x_prime = x_test[i]
                else:
                    x_prime = np.concatenate((x_test[i], y_hat))
                D_j_prime_x.append(x_prime)

            D_j_prime_x = np.array(D_j_prime_x)

            # do x' ← [x1,...,xd , yˆ1,..., yˆj−1]
            y_pred_j = classifiers[j].predict(D_j_prime_x)[0]

            # return y
            y_hat.append(y_pred_j)
            y_hat = np.array(y_hat).flatten()
            y_hat = y_hat.tolist()





'''
        accuracies = []
        f1_scores = []

        for j in range(L):
            y_true = y_test[:,j]
            y_pred = y_hat[:,j]

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            accuracies.append(acc)
            f1_scores.append(f1)

        for j in range(L):
            print(f"标签 {j + 1}: 准确率 = {accuracies[j]:.2f}, F1 分数 = {f1_scores[j]:.2f}")

        avg_accuracy = np.mean(accuracies)
        avg_f1_score = np.mean(f1_scores)

'''



#



'''
D = []

for i in range(0,19):
    Dj = {}
     # do jth 二进制转换和训练
    for x,y in D:
        # do x' 映射 x1...xd, y1...yd
        Dj += Dj + x' yj
        # train hj to predict binary relevance of yi
    # hj Dj' {0,1}
'''