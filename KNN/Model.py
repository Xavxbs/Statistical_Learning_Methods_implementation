# -*- coding: utf-8 -*-
# @Time    : 2020/2/17 18:25
# @Author  : Xav
# @File    : Model.py

import numpy as np
from collections import defaultdict


class Kdtree:
    def __init__(self, x, y):
        """
        构建kd树的节点
        :param x: 数据
        :param y: 标签
        """
        self.val = x
        self.target = y
        # 左孩子
        self.left = None
        # 右孩子
        self.right = None


class KNN:
    def __init__(self, data, labels, k):
        """

        :param data: 训练集数据
        :param labels: 训练集标签
        :param k: 参与多数表决的节点数量
        """
        # 将数据和标签合为一个ndarry，方便排序
        self.data = np.hstack((data, np.expand_dims(labels, axis=1)))
        # 数据数量
        self.n = len(self.data)
        # 数据维度
        self.w = len(self.data[0]) - 1
        self.k = min(k, self.n)
        # 构建树根
        self.head = self.create_tree(0, len(data), 0)

    def create_tree(self, start, end, idx):
        """
        使用递归建造kd树
        :param start: 构建树的起始位置
        :param end: 构建树的结束位置
        :param idx: 对data数据排序所依照的维度
        :return: 返回树根
        """
        if start >= end:
            return None
        # 保证维度不超过最大值
        idx = idx % self.w
        # 按照指定维度的数据对所有数据进行排序
        self.data[start:end] = self.data[np.argsort(self.data[start:end, idx])]
        # 取中位数，并取其位置作为树根
        mid = start + (end - start) // 2
        head = Kdtree(self.data[mid][0:self.w], self.data[mid][self.w])
        # 递归生成左子树
        head.left = self.create_tree(start, mid, idx + 1)
        # 递归生成右子树
        head.right = self.create_tree(mid + 1, end, idx + 1)
        # 返回树根
        return head

    def trace(self, x, node, idx, stack):
        """
        将另一节点的区域加入栈中
        :param x: 需要预测的数据
        :param node: 当前需要判断的节点
        :param idx: 判断时需要的维度
        :param stack: 用于保存路径的栈
        """
        # 当节点不是叶子结点时
        while node.left is not None or node.right is not None:
            idx = idx % self.w
            # 判断下一个需要加入栈中的节点是当前节点的左孩子还是右孩子
            if x[idx] > node.val[idx]:
                if node.right is not None:
                    node = node.right
                else:
                    break
            else:
                if node.left is not None:
                    node = node.left
                else:
                    break
            # 将节点加入栈
            stack.append((node, idx))
            idx += 1

    def find_topk(self, x):
        """

        :param x: 测试集中的单条数据
        :return: 在kd树中查找空间中与x最接近的前k个结点
        """
        # 统计最短长度的数组，数组中有结点和结点与预测数据在空间中的距离
        length = [(Kdtree(0, 0), -1)]
        # 循环k个轮次
        for currentk in range(self.k):
            # current_node初始化为树根结点
            current_node = self.head
            # 用于记录当前轮最接近的结点和距离
            nearest = None
            # 用于记录路径的栈，查找过程是由底到顶的顺序，所以需要记录
            stack = [(current_node, 0)]
            # 按照制定维度对所有数据进行比较
            idx = 0
            nearest_length = float('inf')
            # 从根部出发，向下访问kd树，同时记录路径，当访问到kd树的叶子结点为止
            while current_node.left is not None or current_node.right is not None:
                idx = idx % self.w
                if x[idx] > current_node.val[idx]:
                    if current_node.right is not None:
                        current_node = current_node.right
                    else:
                        break
                else:
                    if current_node.left is not None:
                        current_node = current_node.left
                    else:
                        break
                # 记录路径
                stack.append((current_node, idx))
                # 切换到下一个维度
                idx += 1
            # 记录当前结点与测试数据在空间中的距离
            current_length = np.sqrt(np.sum(np.square(x - current_node.val)))
            # 与上一轮的最短距离比较，确保当前要找的到的结点是除之查找到的结点以外距离数据在空间中最近的点
            if current_length >= length[-1][1] and current_node not in length:
                # 记录当前结点信息，并更新本轮最短距离
                nearest = (current_node, current_length)
                nearest_length = min(nearest_length, current_length)
            # 弹出栈
            current_node, current_idx = stack.pop()
            while True:
                # 记录当前结点与测试数据在空间中的距离
                current_length = np.sqrt(np.sum(np.square(x - current_node.val)))
                # 判断是否是符合要求的最近的结点，如果是，就记录这个节点，然后向kd树的上层路径回退
                if current_length < nearest_length and current_length >= length[-1][1] and current_node not in length:
                    nearest_length = current_length
                    nearest = (current_node, nearest_length)
                    # 当栈空时停止
                    if not stack:
                        break
                    current_node, current_idx = stack.pop()
                else:
                    # 如过不是当前最近距离的点，则检查该结点的另一半区域是否与以目标点为球心，当前最短距离为半径的超球体相交
                    tmp_val = current_node.val
                    # 构造找到边界上距离目标点最近的点
                    tmp_val[current_idx] = x[current_idx]
                    current_length = np.sqrt(np.sum(np.square(x - tmp_val)))
                    # 如果另一区域与超球体相交
                    if current_length < nearest_length and current_length >= length[-1][1] and current_node not in length:
                        # 判断继续探索哪一半的区域
                        if current_node.val[current_idx] < x[current_idx]:
                            if current_node.left is not None:
                                stack.append((current_node.left, idx + 1))
                                self.trace(x, current_node.left, idx + 1, stack)
                                current_node, current_idx = current_node.left, idx + 1
                        else:
                            if current_node.right is not None:
                                stack.append((current_node.right, idx + 1))
                                self.trace(x, current_node.right, idx + 1, stack)
                                current_node, current_idx = current_node.right, idx + 1
                    else:
                        # 栈空时停止
                        if not stack:
                            break
                        current_node, current_idx = stack.pop()
            # 判断是否找到距离最近的点
            # 这里找到的topk中最短距离都是unique value
            if nearest:
                length.append(nearest)
            else:
                break
        return length[1:self.k + 1]

    def predict(self, x):
        """

        :param x: 测试集中的单条数据
        :return: 返回预测标签
        """
        # 与当前数据在空间中最接近的前k个点，这里距离使用的是欧式距离
        topk = self.find_topk(x)
        # 统计投票结果的字典
        dict = defaultdict(int)
        for item in topk:
            dict[item[0].target] += 1
        # 返回投票结果中投票最多的那个标签
        return max(dict, key=dict.get)

    def test(self, data, labels):
        """
        预测新数据并进行测试的入口函数
        :param data: 测试集数据
        :param labels: 测试集标签
        :return: 测试集准确率
        """
        n = len(data)
        # 统计正确数目
        corrects = 0
        for i in range(n):
            pred = self.predict(data[i])
            if pred == labels[i]:
                # 正确时正确数加一
                corrects += 1
        return corrects / n
