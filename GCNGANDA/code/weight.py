import numpy as np
#小二乘闭式解求解
#假设 (S^T S) 可逆
class WeightedStackedEnsembleClosedForm:
    def __init__(self):
        self.w = None

    def fit(self, S, y):
        self.w = np.linalg.inv(S.T @ S) @ (S.T @ y)
    def predict(self, S):

        return S @ self.w