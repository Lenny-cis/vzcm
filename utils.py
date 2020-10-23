# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:38:28 2019

@author: linjn
"""

import numpy as np
import pandas as pd


def obtainNonRelativeFeats(corrDf, varsDic, thred=0.6):
    """
    处理共线性问题，通过比较相关性高的成对变量的IV值，挑选成对变量中的一个,两两相关的成对变量会形成一连串的变量集，在这些变量集中挑选一个IV最高的变量.

    input
        corrDf          相关系数矩阵
        varsDic         变量IV集
        thred           相关系数阈值
    """
    # 保留阈值以上的变量系数
    t_df = corrDf.copy()
    t_df = t_df[abs(t_df) > thred]
    if len(t_df) == 0:
        return list(corrDf.index)
    s = 1
    corrFeats = []
    # 选择一个变量开始遍历，初始化最大IV
    while s:
        initIdx = t_df.index[0]
        notnaIdx = list(t_df.index[t_df.loc[:, initIdx].notna()])[1:]
        maxIV = varsDic.loc[initIdx, 'IV']
        maxIdx = initIdx
        # 遍历所有与初始变量相关的变量的IV，比较IV值，取大者。
        for idx in notnaIdx:
            if varsDic.loc[idx, 'IV'] > maxIV:
                corrFeats.append(maxIdx)
                maxIV = varsDic.loc[idx, 'IV']
                maxIdx = idx
            # 若IV值小于初始变量的IV则剔除该变量相应的行和列
            else:
                corrFeats.append(idx)
                t_df.drop(idx, axis=1, inplace=True)
                t_df.drop(idx, inplace=True)
        # 遍历结束删除初始变量，继续遍历下一个变量
        t_df.drop(initIdx, axis=1, inplace=True)
        t_df.drop(initIdx, inplace=True)
        if len(t_df) == 0:
            s = 0
    return list(set(corrDf.index) - set(corrFeats))


def update_dict(dict1, dict2):
    dict1 = dict1.copy()
    for k, v in dict2.items():
        dic = dict1.get(k, {})
        dic.update(v)
    return dict1
