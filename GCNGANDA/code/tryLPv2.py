#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#列出所有可能标签的组合，但是因为有些实例没有用，所以此代码无用
from u_mReadData import *
from u_evaluation import *
"""
label_powerset_logreg.py

演示将 (1629 x 14) 多标签矩阵转为单整数 ID（Label Powerset），
并使用 sklearn 的 LogisticRegression 进行多分类训练与预测。

包含:
1) generate_id_binary_map       -> 生成 int ↔ tuple(...) 的固定映射
2) map_label_matrix_to_ids      -> 多标签矩阵 -> 单标签 ID
3) map_ids_to_label_matrix      -> 单标签 ID -> 多标签矩阵(反向映射)

用法：直接运行该脚本，即可在终端看到示例输出。

2023-10 by ChatGPT
"""

import numpy as np
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def generate_id_binary_map(n):
    """
    生成并返回两个字典:
      1) id_to_binary: int -> tuple(0/1, ..., 0/1)
      2) binary_to_id: tuple(0/1, ..., 0/1) -> int

    按固定顺序 (字典序) 为每个 n 维二进制组合分配 ID。
    比如 n=2 时:
      (0, 0) -> 0
      (0, 1) -> 1
      (1, 0) -> 2
      (1, 1) -> 3
    """
    num = 0
    id_to_binary = {}
    binary_to_id = {}
    for idx, bits in enumerate(itertools.product([0, 1], repeat=n)):
        # bits 是一个长度为 n 的元组，例如 (0, 0, 1, ...)
        id_to_binary[idx] = bits
        binary_to_id[bits] = idx
        num += 1

    print(num)
    return id_to_binary, binary_to_id


def map_label_matrix_to_ids(Y, binary_to_id):
    """
    将形状 (n_samples, n_labels) 的二进制矩阵 Y，每行映射到一个整数 ID。

    参数:
      - Y: np.ndarray，形状 (n_samples, n_labels)，元素为 0/1
      - binary_to_id: dict，二进制组合 -> ID 的映射

    返回:
      - single_label_ids: np.ndarray，形状 (n_samples,) 的整数数组
    """
    n_samples, n_labels = Y.shape
    single_label_ids = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # 将第 i 行 (14 维) 转成 tuple，作为字典键
        combo = tuple(Y[i].tolist())
        if combo in binary_to_id:
            single_label_ids[i] = binary_to_id[combo]
        else:
            # 理论上不会发生（只要是 0/1，就在 2^n 里）
            raise ValueError(f"第{i}行的标签组合 {combo} 不在映射字典中!")

    return single_label_ids


def map_ids_to_label_matrix(ids, id_to_binary):
    """
    将单标签 ID 的一维数组 (shape: n_samples,) 反向映射回
    (n_samples, n_labels) 的二进制矩阵。

    参数:
      - ids: np.ndarray，形状 (n_samples,)
      - id_to_binary: dict, ID -> tuple(0/1, ..., 0/1)

    返回:
      - Y_out: np.ndarray，形状 (n_samples, n_labels)
    """
    results = []
    for cid in ids:
        # 通过 id_to_binary[cid] 得到 tuple(0/1, ..., 0/1)
        combo = id_to_binary[cid]
        results.append(combo)
    # 转为 numpy 数组，维度: (n_samples, n_labels)
    return np.array(results, dtype=int)


def main():

    datasnames = ["Yeast"]
    rd = ReadData(datas=datasnames, genpath='data/')
    X_train, Y_train, X_test, Y_test = rd.readData(0)
    # -------------------
    # 1) 准备示例数据
    # -------------------


    # -------------------
    # 2) 生成固定映射
    # -------------------
    # 对所有 14 维二进制向量(2^14=16384种) 按字典序 -> 唯一ID
    id_to_binary, binary_to_id = generate_id_binary_map(14)
    print("\n已生成 2^14={} 个二进制组合的映射。".format(len(id_to_binary)))

    # -------------------
    # 3) 标签多标签 -> 单标签 ID
    # -------------------
    y_ids = map_label_matrix_to_ids(Y_train, binary_to_id)
    print("映射后的 y_ids 形状:", y_ids.shape)  # (1629,)

    # -------------------
    # 4) 训练 LogisticRegression 多分类
    # -------------------

    clf = LogisticRegression(max_iter=1000, multi_class='auto')
    clf.fit(X_train, y_ids)

    # -------------------
    # 5) 预测及反向映射
    # -------------------
    y_pred = clf.predict(X_test)
    Y_pred_bin = map_ids_to_label_matrix(y_pred, id_to_binary)
    accuracy = np.mean(Y_pred_bin == Y_test)
    print("\nLogisticRegression 在测试集上的准确率:", accuracy)

    # 反向映射: ID -> 二进制
    # 生成 (n_test x 14) 的二进制矩阵

    eva = evaluate(Y_pred_bin, Y_test)
    print(eva)
    # 演示查看测试集前 5 条数据的预测结果



if __name__ == "__main__":
    main()
