# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:01:01 2020

@author: linjianing
"""


import pandas as pd
from .utils import var_mrcr


def df_mrcr(df, y, desc_xs=None):
    """数据集特征初步描述."""
    if desc_xs is None:
        desc_xs = df.columns.difference([y])

    elif isinstance(desc_xs, str):
        desc_xs = [desc_xs]

    desc_dict = {}
    desc_xs = df.columns.intersection(desc_xs)
    for x in desc_xs:
        x_ser = df.loc[:, x]
        mr, cr, n_u = var_mrcr(x_ser)
        desc_dict.update({x: {'nonmissing_ratio': 1 - mr,
                              'nonconcentration_ratio': 1 - cr,
                              'nunique': n_u}})

    return desc_dict
