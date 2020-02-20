# -*- coding: utf-8 -*-
# @Time    : 2020/2/19 19:44
# @Author  : Xav
# @File    : Model.py

import numpy as np
from collections import defaultdict


class TreeNode:
    def __init__(self):
        # 记录其子结点
        self.next = {}
        # 当前结点的决策用的特征
        self.feature = None
        # 当self.next为空时，代表这个结点时叶子结点
        # self.c 用来记录叶子结点的类标记
        self.c = None


class DecisionTree:
    def __init__(self, eplison=0.1):
        '''

        :param eplison: 阈值参数
        '''
        # label的类别数
        self.ck = None
        # 数据集的特征数
        self.feature = None
        # 阈值
        # 书中 5.3.1节算法中介绍，如果所有的特征的信息增益小于阈值
        # 则置T为但点数，将T中实例数最大的类Ck作为该类结点的类标记
        self.e = eplison
        self.tree = TreeNode()
        self.total_nodes = 0

    def _findmost(self, array):
        '''
        找到一个array中出现次数最多的值
        :param array: 一个list或者ndarray
        :return: 出现次数最多的值
        '''
        dict = defaultdict(int)
        for i in array:
            dict[i] += 1
        return max(dict, key=lambda k: dict[k])

    def _cal_h_d(self, data, ck):
        '''
        计算一个数据集D的经验熵
        :param data: 数据集D
        :param ck: 数据集中所有的标签
        :return: 经验熵
        '''
        h_d = 0
        size_data = len(data)
        for c in ck:
            # H(D) = -∑（size_ck/size_data)*log2(size_ck/size_data)
            size_ck = sum(data[:, -1] == c)
            h_d -= (size_ck / size_data) * np.log2((size_ck / size_data))
        return h_d

    def _cal_h_d_a(self, data, a):
        '''
        计算特征A对数据集D的经验条件熵
        :param data: 数据集D
        :param a: 特征A
        :return: 经验条件熵
        '''
        h_d_a = 0
        size_data = len(data)
        d_sub = self._data_cut(data, np.unique(data[:, a]), a)
        # H(D|A) = ∑ (size_di / size_data) * H(Di)
        for di in d_sub:
            size_di = len(di)
            h_d_a += (size_di / size_data) * self._cal_h_d(di, np.unique(di[:, -1]))
        return h_d_a

    def _data_cut(self, data, vals, a):
        '''
        将数据集按照特定特征的取值分为相对应的小的数据集
        :param data: 原数据集
        :param vals: 取值
        :param a: 特征
        :return: 小数据集集合
        '''
        datasets = []
        for val in vals:
            datasets.append(data[data[:, a] == val])
        return datasets

    def create_tree(self, feature, data, node):
        '''
        建造决策树
        :param feature: 可选的特征
        :param data: 当前数据集
        :param node: 当前决策树结点
        :return:
        '''
        self.total_nodes += 1
        print('node:'+str(self.total_nodes))
        current_ck = np.unique(data[:, -1])
        # 如果D中所有实例属于同一类Ck，则置T为单节点数，并将Ck作为该节点的类，返回T
        if len(current_ck) == 1:
            node.c = data[0, -1]
            return
        # 如果A为空集，则置T为单节点数，并将D中实例数最大的类Ck作为该节点的类，返回T
        if len(feature) == 0:
            node.c = self._findmost(data[:, -1])
            return
        # 计算数据集的经验熵
        h_d = self._cal_h_d(data, current_ck)
        # 计算各个可选特征对数据集的经验条件熵
        # 并计算信息增益
        max_g_d_a = float('-inf')
        a_g = float('-inf')
        for a in feature:
            print('calculating g(D,A) by feature:' + str(a))
            g_d_a = h_d - self._cal_h_d_a(data, a)
            # 选定信息增益最大的特征
            if max_g_d_a < g_d_a:
                max_g_d_a = g_d_a
                a_g = a
        # 书中 5.3.1节算法中介绍，如果所有的特征的信息增益小于阈值
        # 则置T为但点数，将T中实例数最大的类Ck作为该类结点的类标记
        if max_g_d_a < self.e:
            node.c = self._findmost(data[:, -1])
            return
        # 对Ag的每一可能值ai，依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的
        # 类作为标记，构建子节点，由节点及其子节点构成树T，返回T
        rest_features = feature.copy()
        rest_features.remove(a_g)
        a_g_vals = np.unique(data[:, a_g])
        d_sub = self._data_cut(data, a_g_vals, a_g)
        for idx, di in enumerate(d_sub):
            node.feature = a_g
            node.next[a_g_vals[idx]] = TreeNode()
            self.create_tree(rest_features, di, node.next[a_g_vals[idx]])

    def train(self, data, labels):
        """

        :param data: 训练数据
        :param labels: 训练标签
        :param epoch: 训练轮数，默认为50
        :return: 训练结束后的模型对训练数据的分类准确度
        """
        print('trainning')
        # 获得样本数量
        n = len(labels)
        self.ck = np.unique(labels)
        self.feature = len(data[0])
        feature = [i for i in range(self.feature)]
        data = np.hstack((data, np.expand_dims(labels, axis=1)))
        self.create_tree(feature[:], data, self.tree)

    def pred(self, data):
        """
        预测数据的标签
        :param data:
        :return:
        """
        current_node = self.tree
        while current_node.next:
            feature = current_node.feature
            current_node = current_node.next[data[feature]]
        prediction = current_node.c
        return prediction

    def test(self, data, labels):
        """

        :param data: 测试集数据
        :param labels: 测试集标签
        :return: 模型对测试集数据的预测准确率
        """
        n = len(labels)
        correct = 0
        for i in range(n):
            print('predicting:' + str(i))
            x = data[i]
            y = labels[i]
            if self.pred(x) == y:
                correct += 1
        # 返回准确率
        return correct / n
