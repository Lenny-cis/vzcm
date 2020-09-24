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
        desc_xs = list(df.columns - set([y]))

    elif isinstance(desc_xs, str):
        desc_xs = [desc_xs]

    desc_dict = {}
    desc_xs = list(df.columns & set(desc_xs))
    for x in desc_xs:
        x_ser = df.loc[:, x]
        mr, cr = var_mrcr(x_ser)
        desc_dict.update({x: {'nonmissing_ratio': 1 - mr,
                              'nonconcentration_ratio': 1 - cr}})

    return desc_dict
