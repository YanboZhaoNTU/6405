import numpy as np


class WeightedStackedEnsemble:
    def __init__(self, lr=0.01, num_iter=1000, tol=1e-5):
        """
        lr: 学习率
        num_iter: 迭代次数
        tol: 判定“损失改善”是否足够大的阈值
        """
        self.lr = lr
        self.num_iter = num_iter
        self.tol = tol
        self.w = None

    def fit(self, S, y):
        n_samples, d = S.shape
        # 初始化权重
        self.w = np.zeros(d)

        # 计算初始的 "总平方误差"
        best_loss = np.sum((y - S.dot(self.w)) ** 2)

        for _ in range(self.num_iter):
            old_w = self.w.copy()
            old_loss = best_loss

            # 前向计算
            y_pred = S.dot(self.w)
            # 梯度: -(1/n)*S^T(y - S w)
            grad = -(2.0 / n_samples) * S.T.dot(y - y_pred)
            # 更新 w
            self.w -= self.lr * grad

            # 计算新误差(平方误差和)
            new_loss = np.sum((y - S.dot(self.w)) ** 2)
            # 若误差改善不足，则回退并停止
            if (old_loss - new_loss) < self.tol:
                self.w = old_w
                break
            else:
                best_loss = new_loss

    def predict(self, S):
        return S.dot(self.w)