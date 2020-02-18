# -*- coding: utf-8 -*-
# @Time    : 2020/2/17 18:26
# @Author  : Xav
# @File    : main.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from Model import KNN

'''
k-近邻算法实现
这里设定最近参与多数表决的节点数量k为2
具体使用kd树实现了查找，适合训练实例数远大于空间维度时的查找，平均计算复杂度为O（logN）（N为维度）
当空间维度接近训练实例数时，kd树查找的效率会接近线性扫描
使用scikit-learn内置的鸢尾花数据集
训练集数量 480
测试集数量 120
-------------------------------
测试集正确率：69.33%
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
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, random_state=10)
    #设定最近参与多数表决的节点数量
    k = 2
    #创建k-近邻对象
    model = KNN(x_train,y_train,k)
    #对测试集进行测试
    tst = model.test(x_test, y_test)
    print('测试集准确率：' + str(tst))
