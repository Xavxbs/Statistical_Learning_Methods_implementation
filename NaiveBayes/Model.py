# -*- coding: utf-8 -*-
# @Time    : 2020/2/18 21:58
# @Author  : Xav
# @File    : Model.py
from collections import Counter

import numpy as np
from numpy import ndarray, exp, pi, sqrt


class GaussianNB:
    """
    高斯朴素贝叶斯分类器
        prior: 先验概率
        avgs: 均值
        vars: 方差
        n_class: label的分类数
    """

    def __init__(self):
        self.prior = None
        self.avgs = None
        self.vars = None
        self.n_class = None

    @staticmethod
    def _get_prior(label: ndarray) -> ndarray:
        """
        计算先验概率
        :param label: target标签
        :return: 先验概率
        """
        cnt = Counter(label)
        prior = np.array([cnt[i] / len(label) for i in range(len(cnt))])
        return prior

    def _get_avgs(self, data: ndarray, label: ndarray) -> ndarray:
        '''
        计算均值
        :param data: 训练集数据
        :param label: 训练集标签
        :return: 返回方差
        '''
        return np.array([data[label == i].mean(axis=0) for i in range(self.n_class)])

    def _get_vars(self, data: ndarray, label: ndarray) -> ndarray:
        '''
        计算方差
        :param data: 训练集数据
        :param label: 训练集标签
        :return: 返回方差
        '''
        return np.array([data[label == i].var(axis=0)
                         for i in range(self.n_class)])

    def _get_posterior(self, row: ndarray) -> ndarray:
        '''
        对每个特征计算后验概率
        :param row: 特征
        :return: 后验概率连乘后的结果
        '''
        return (1 / sqrt(2 * pi * self.vars) * exp(
            -(row - self.avgs) ** 2 / (2 * self.vars))).prod(axis=1)

    def fit(self, data: ndarray, label: ndarray):
        '''
        训练模型
        :param data: 训练集数据
        :param label: 训练集标签
        :return:
        '''

        # 计算先验概率
        self.prior = self._get_prior(label)
        # 计算标签的分类个数
        self.n_class = len(self.prior)
        # 计算平均值
        self.avgs = self._get_avgs(data, label)
        # 计算方差
        self.vars = self._get_vars(data, label)
        # 计算训练集误差
        train_acc = self.test(data, label)
        print(train_acc)

    def predict_prob(self, data: ndarray) -> ndarray:
        '''
        预测各label的概率
        :param data: 测试集概率
        :return:
        '''

        # 计算各个类和属性的似然函数，相乘得到概率
        likelihood = np.apply_along_axis(self._get_posterior, axis=1, arr=data)
        probs = self.prior * likelihood
        # 标准化数据
        probs_sum = probs.sum(axis=1)
        # 这里可以直接返回probs，但是为了阐明原理还是写上了后半部分
        return probs / probs_sum[:, None]

    def test(self, data: ndarray, label: ndarray) -> ndarray:
        '''
        对测试集进行预测并得到正确率
        :param data:
        :param label:
        :return:
        '''
        # 选择概率最大的一类
        prid_label = self.predict_prob(data).argmax(axis=1)
        # 返回正确率
        return np.sum(prid_label == label) / len(label)


class NaiveBayes:
    def train(self, data, labels, epoch=50):
        """

        :param data: 训练数据
        :param labels: 训练标签
        :param epoch: 训练轮数，默认为50
        :return: 训练结束后的模型对训练数据的分类准确度
        """
        # 获得样本数量
        n = len(labels)
        self.feature_num = len(data[0])
        self.kc = np.unique(labels)
        self.class_nums = len(self.kc)
        # 将数据和标签合并方便运算
        data = np.hstack((data, np.expand_dims(labels, axis=1)))
        self.prior_prob = np.zeros((self.class_nums, 1))
        self.y_dict = {}
        self.x_dict = []
        self.x_val_nums = []
        self.x_val_unique = []
        # 生成label类的值与索引值的映射
        for i, class_val in enumerate(self.kc):
            self.y_dict[class_val] = i
        # 记录各个特征的不同值的数量的最大值
        # 用于在生成条件概率的矩阵时初始化大小
        max_unique_classes = 0
        for i in range(self.feature_num):
            xj_dict = {}
            # 各特征的可取值
            self.x_val_unique.append(np.unique(data[:, i]))
            # 各特征的不同值的数量
            self.x_val_nums.append(len(self.x_val_unique[-1]))
            for idx, xj_val in enumerate(self.x_val_unique[-1]):
                max_unique_classes = max(max_unique_classes, idx)
                xj_dict[xj_val] = idx
            self.x_dict.append(xj_dict.copy())
        # 计算各个标签的先验概率的极大似然估计
        for c in self.kc:
            self.prior_prob[self.y_dict[c]] = ((np.sum(data[:, -1:][np.where(data[:, -1:] == c)])) + 1) / (
                n + 1 * self.class_nums)
        # 计算各个特征的条件概率的极大似然估计
        self.cond_prob = np.zeros((self.class_nums, self.feature_num, max_unique_classes + 1))
        for data_slice in data:
            for i in range(self.feature_num):
                self.cond_prob[self.y_dict[data_slice[-1]]][i][self.x_dict[i][data_slice[i]]] += 1
        # 由于每个概率值很小（比如0.0001）若干个很小的概率值直接相乘，得到的结果会越来越小
        # 为了避免计算过程出现下溢，引入对数函数Log，在对数空间中进行计算
        for label in range(self.class_nums):
            for feature in range(self.feature_num):
                for val in range(self.x_val_nums[feature]):
                    self.cond_prob[label][feature][val] = np.log((self.cond_prob[label][feature][val] + 1) / (
                        self.prior_prob[label] * (n + 1 * self.class_nums) + 1 * self.x_val_nums[feature]))
        self.prior_prob = np.log(self.prior_prob)

    def pred(self, data):
        """
        预测数据的标签
        :param data: 待预测数据
        :return:
        """
        predictions = []
        for label in range(self.class_nums):
            prob = self.prior_prob[label]
            # 由于先验概率和条件概率都转换到了对数空间
            # 所以这里计算后验概率时，需要将先验概率和各个条件概率连加，并非连乘
            # 得到的也是后验概率的对数值，由于对数函数在（0，1）区间内单调递增
            # 所以可以直接比较后验概率的对数值
            for feature in range(self.feature_num):
                xj = data[feature]
                # 如果xj在现有的模型中
                if xj in self.x_val_unique[feature]:
                    prob += self.cond_prob[label][feature][self.x_dict[feature][xj]]
                # 我实验的时候使用的是鸢尾花数据集
                # 它的每个特征可以看为是连续数据，所以在训练集中是无法枚举完的，之后我又换用了高斯核的朴素贝叶斯分类器
                # 如果xj不在现有的模型中
                # 判断xj和哪一个现有的值更为接近
                else:
                    idx = -1
                    val_n = self.x_val_nums[feature]
                    for i in range(val_n):
                        if xj < self.x_val_unique[feature][i]:
                            idx = i
                            break
                    # 当特征值大于模型中已有的该特征的所有值时令特征值为最大
                    # 当特征值校于模型中已有的该特征的所有值时令特征值为最小
                    # 否则找到它最接近的已有的值
                    if idx == -1:
                        prob += self.cond_prob[label][feature][self.x_dict[feature][self.x_val_unique[feature][-1]]]
                    elif idx == 0:
                        prob += self.cond_prob[label][feature][self.x_dict[feature][self.x_val_unique[feature][0]]]
                    else:
                        idx = idx if (self.x_val_unique[feature][idx] - xj) <= (
                            xj - self.x_val_unique[feature][idx - 1]) else (idx - 1)

                        prob += self.cond_prob[label][feature][self.x_dict[feature][self.x_val_unique[feature][idx]]]
            predictions.append(prob)
        return self.kc[predictions.index(max(predictions))]

    def test(self, data, labels):
        """

        :param data: 测试集数据
        :param labels: 测试集标签
        :return: 模型对测试集数据的预测准确率
        """
        n = len(labels)
        wrongs = 0
        for i in range(n):
            x = data[i]
            y = labels[i]
            if self.pred(x) != y:
                wrongs += 1
        # 返回准确率
        return (n - wrongs) / n
