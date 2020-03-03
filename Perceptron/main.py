# -*- coding: utf-8 -*-
# @Time    : 2020/2/17 16:10
# @Author  : Xav
# @File    : main.py

import numpy as np
from sklearn.model_selection import train_test_split
from Model import Perceptron

'''
在部分代码中使用了额外的Mnist数据集
训练集56000
测试集14000
-------------------------------
训练集正确率：98.92%
测试集正确率：98.73%
'''


def load_data():
    '''
    加载文件
    :param fileName:要加载的文件路径
    :return: 数据集和标签集
    '''
    #存放数据及标记
    dataArr = []; labelArr = []
    #读取文件
    files = ['../data/mnist_train.csv','../data/mnist_test.csv']
    # files = ['../data/test.csv']
    for file_name in files:
        fr = open(file_name)
        #遍历文件中的每一行
        for line in fr.readlines():
            #获取当前行，并按“，”切割成字段放入列表中
            #strip：去掉每行字符串首尾指定的字符（默认空格或换行符）
            #split：按照指定的字符将字符串切割成每个字段，返回列表形式
            curLine = line.strip().split(',')
            #将每行中除标记外的数据放入数据集中（curLine[0]为标记信息）
            #在放入的同时将原先字符串形式的数据转换为整型
            #此外将数据进行了二值化处理，大于128的转换成1，小于的转换成1，方便后续计算
            dataArr.append([int(int(num) > 128) for num in curLine[1:]])
            #将标记信息放入标记集中
            #放入的同时将标记转换为整型
            labelArr.append(1 if int(curLine[0]) == 1 else -1)
    #返回数据集和标记
    return np.array(dataArr), np.array(labelArr)

if __name__ == '__main__':
    # 读取数据
    data, labels = load_data()
    # 打乱数据并分割数据集
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=10)
    # 创建感知机分类器对象
    model = Perceptron(len(x_train[0]))
    # 训练感知机模型并开始训练
    train_acc = model.train(x_train, y_train)
    print('train acc:' + str(train_acc))
    # 对测试集进行测试
    test_acc = model.test(x_test, y_test)
    print('test acc:' + str(test_acc))
