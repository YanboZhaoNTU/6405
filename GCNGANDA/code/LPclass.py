import numpy as np
from sklearn.linear_model import LogisticRegression

from L2 import WeightedStackedEnsemble


class LabelPowersetLogistic:
    """
    使用 Label Powerset 将多标签问题转换为单标签多分类问题，
    并调用 sklearn 的 LogisticRegression 作为基分类器。
    """

    def __init__(self, **lr_kwargs):
        """
        lr_kwargs: 可以传给 LogisticRegression 的超参数，如 C=1.0, penalty='l2' 等
                   例如: LabelPowersetLogistic(C=2.0, max_iter=200)
        """
        self.clf = LogisticRegression(**lr_kwargs)

        # 记录标签组合和ID之间的映射
        self._label_combo_to_id = {}  # dict: { tuple_of_labels -> class_id }
        self._id_to_label_combo = {}  # dict: { class_id -> tuple_of_labels }

        self._is_fitted = False
        self.train_classifier = []
        self.train_predict_result = np.empty((1629, 0))
        self.test_predict_result = np.empty((788, 0))
        self.all_clf_test = []
        self.circle_result = []

    def fit(self, X, Y):
        """
        训练模型
        X: (n_samples, n_features) 特征矩阵
        Y: (n_samples, n_labels)   多标签二进制矩阵
        """
        # 1) 将多标签矩阵转换为单标签 ID
        single_label_ids = self._fit_label_converter(Y)

        # 2) 训练逻辑回归
        self.clf.fit(X, single_label_ids)
        self.train_classifier.append(self.clf)
        self._is_fitted = True
        return self

    def predict(self, X):
        """
        对给定特征 X 进行多标签预测
        X: (n_samples, n_features)
        return: (n_samples, n_labels) 预测的多标签矩阵
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")

        # 1) 先预测单一类别 ID
        pred_label_ids = self.clf.predict(X)

        # 2) 将类别 ID 转回原多标签向量
        result = self._inverse_transform(pred_label_ids)
        self.train_predict_result = np.hstack([self.train_predict_result, result])


        return np.array(self.train_predict_result)


    def test_fit(self, X, i):
        """
        对给定特征 X 进行多标签预测
        X: (n_samples, n_features)
        return: (n_samples, n_labels) 预测的多标签矩阵
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")

        # 1) 先预测单一类别 ID
        pred_label_ids = self.train_classifier[i].predict(X)

        # 2) 将类别 ID 转回原多标签向量
        Y_pred = self._inverse_transform(pred_label_ids)
        self.test_predict_result = np.hstack([self.test_predict_result, Y_pred])
        return np.array(self.test_predict_result)

    def LP_train_BR_train(self, X, Y):
        for i in range(Y.shape[1]):
            print(Y.shape[1])
            #        cfl = WeightedStackedEnsemble()
            cfl = WeightedStackedEnsemble()
            cfl.fit(X, Y[:, i])
            self.all_clf_test.append(cfl)


    def BR_test_BRC_test(self, X):
        for clf in self.all_clf_test:
            self.circle_result.append(clf.predict(X))
        return np.array(self.circle_result).T



    def _fit_label_converter(self, Y):
        """
        创建训练集中的多标签 -> 单标签 ID 映射，并返回单标签ID数组。
        Y: (n_samples, n_labels)
        return: single_label_ids: (n_samples,)
        """
        n_samples = Y.shape[0]
        single_label_ids = np.zeros(n_samples, dtype=int)

        combo_map = {}
        curr_id = 0

        for i in range(n_samples):
            # 转成 tuple，方便作为字典键
            combo = tuple(Y[i].tolist())
            if combo not in combo_map:
                combo_map[combo] = curr_id
                curr_id += 1
            single_label_ids[i] = combo_map[combo]

        # 将 combo_map 反转为 id->combo
        self._label_combo_to_id = combo_map
        self._id_to_label_combo = {v: k for k, v in combo_map.items()}

        return single_label_ids

    def _inverse_transform(self, single_label_ids):
        """
        将预测得到的单标签 ID 转换回多标签二进制向量
        single_label_ids: (n_samples,)
        return: (n_samples, n_labels)
        """
        # 通过任意一个组合获取 n_labels
        any_combo = next(iter(self._label_combo_to_id))  # 取一个 key
        n_labels = len(any_combo)

        n_samples = len(single_label_ids)
        Y_out = np.zeros((n_samples, n_labels), dtype=int)

        for i in range(n_samples):
            cid = single_label_ids[i]
            if cid in self._id_to_label_combo:
                combo = self._id_to_label_combo[cid]
                Y_out[i, :] = combo
            else:
                # 如果预测到一个不在映射表中的新组合，可以自定义处理，这里全 0
                pass

        return Y_out
