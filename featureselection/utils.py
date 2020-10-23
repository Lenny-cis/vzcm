# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:54:01 2020

@author: linjianing
"""


import pandas as pd
import numpy as np


def var_mrcr(ser):
    """计算变量缺失率和集中度."""
    m_R = ser.isna().sum() / len(ser)
    c_R = ser.value_counts().iloc[0] / ser.count()
    n_ = ser.nunique()
    return m_R, c_R, n_
