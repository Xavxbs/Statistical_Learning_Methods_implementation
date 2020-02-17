# -*- coding: utf-8 -*-
# @Time    : 2020/2/17 16:10
# @Author  : Xav
# @File    : main.py

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from Model import Perceptron

'''
使用scikit-learn内置的乳腺癌数据集
训练集数量 454
测试集数量 114
-------------------------------
训练集正确率：61.98%
测试集正确率：65.79%
'''
def load_data():
    '''
    Load breast cancer dataset
    :return: data and labels
    '''
    raw_data = load_breast_cancer()
    data = raw_data.data
    labels = raw_data.target
    return data, labels


if __name__ == '__main__':
    #读取数据
    data, labels = load_data()
    #打乱数据并分割数据集
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=10)
    #创建感知机分类器对象
    model = Perceptron(len(x_train[0]))
    #训练感知机模型并开始训练
    train_acc = model.train(x_train, y_train)
    print('train acc:' + str(train_acc))
    #对测试集进行测试
    test_acc = model.test(x_test, y_test)
    print('test acc:' + str(test_acc))
