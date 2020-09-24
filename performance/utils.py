# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:44:29 2020

@author: linjianing
"""


import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


def caliv(mat):
    """计算IV值，输入矩阵."""
    warnings.filterwarnings('ignore')
    mat = np.where(mat == 0, 1e-5, mat)
    magc = mat.sum(axis=0)
    entropy = (mat[:, 1]/magc[1] - mat[:, 0]/magc[0])*np.log(
            (mat[:, 1]/magc[1]) / (mat[:, 0]/magc[0]))
    warnings.filterwarnings('default')
    return np.nansum(np.where((entropy == np.inf) | (entropy == -np.inf),
                              0, entropy))


def gen_ksseries(y, pred, pos_label=None):
    """生成KS及KS曲线相应元素.

    input
        y           实际flag
        pred        预测分数
        pos_label   标签被认为是响应，其他作为非响应
    """
    fpr, tpr, thrd = roc_curve(y, pred, pos_label=pos_label)
    eventNum = np.sum(y)
    allNum = len(y)
    nonEventNum = allNum - eventNum
    # KS对应的x点
    ksTile = (eventNum*tpr + nonEventNum*fpr)/allNum
    return tpr - fpr, ksTile


def calks(y, pred, pos_label=None):
    """计算ks值."""
    return np.max(gen_ksseries(y, pred, pos_label)[0])


def calauc(y, pred, pos_label=None):
    """计算auc值."""
    fpr, tpr, thrd = roc_curve(y, pred, pos_label)
    return auc(fpr, tpr)
