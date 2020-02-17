# -*- coding: utf-8 -*-
# @Time    : 2020/2/17 16:37
# @Author  : Xav
# @File    : Model.py
import numpy as np


class Perceptron:
    def __init__(self, dimention, learning_rate=0.001):
        """

        :param dimention: 数据维度
        :param learning_rate: 学习率
        """
        #数据的维度
        self.d = dimention
        #初始化参数w,b
        self.w = [0] * self.d
        self.b = 0
        #初始化学习率
        self.learning_rate = learning_rate

    def train(self, data, labels, epoch=50):
        """

        :param data: 训练数据
        :param labels: 训练标签
        :param epoch: 训练轮数，默认为50
        :return: 训练结束后的模型对训练数据的分类准确度
        """
        #获得样本数量
        n = len(labels)
        #将数据集转换为矩阵形式方便运算
        data = np.mat(data)
        labels = np.mat(labels).T
        for k in range(epoch):
            print('Epoch:' + str(k))
            for i in range(n):
                print('Trainning with data:' + str(i))
                x = data[i]
                y = labels[i]
                #感知机边界对样本的判定
                if -1 * y * (self.w * x.T + self.b) >= 0:
                    self.w = self.w + self.learning_rate * y * x
                    self.b = self.b + self.learning_rate * y
        #统计预测错误的样本数量
        wrongs = 0
        for i in range(len(data)):
            x = data[i]
            y = labels[i]
            if -1 * y * (self.w * x.T + self.b) >= 0:
                wrongs += 1
        #返回准确率
        return (n - wrongs) / n

    def test(self, data, labels):
        """

        :param data: 测试集数据
        :param labels: 测试集标签
        :return: 模型对测试集数据的预测准确率
        """
        n = len(labels)
        data = np.mat(data)
        labels = np.mat(labels).T
        wrongs = 0
        for i in range(n):
            x = data[i]
            y = labels[i]
            if -1 * y * (self.w * x.T + self.b) >= 0:
                wrongs += 1
        #返回准确率
        return (n - wrongs) / n
