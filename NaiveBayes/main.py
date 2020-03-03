# -*- coding: utf-8 -*-
# @Time    : 2020/2/18 21:58
# @Author  : Xav
# @File    : Model.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from Model import NaiveBayes, GaussianNB
import numpy as np

'''
使用scikit-learn内置的iris
使用的高斯朴素贝叶斯分类器
我先编写了朴素贝叶斯分类器，发现它对特征是连续变量的数据集的分类效果很差
之后将数据转化为高斯分布
这里参考了代码，对我的帮助非常大
https://github.com/tushushu/imylu/blob/master/imylu/probability_model/gaussian_nb.py
训练集数量 120
测试集数量 30
-------------------------------
训练集正确率：95.83%
测试集正确率：93.33%
'''
def load_data():
    '''
    Load iris dataset
    :return: data and labels
    '''
    raw_data = load_iris()
    data = raw_data.data
    labels = raw_data.target
    return data, labels


if __name__ == '__main__':
    #读取数据
    data, labels = load_data()
    #打乱数据并分割数据集
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=11)
    #创建分类器对象
    # model = NaiveBayes()
    model = GaussianNB()
    #开始训练
    model.fit(x_train, y_train)
    # #对测试集进行测试
    test_acc = model.test(x_test, y_test)
    print('test acc:' + str(test_acc))