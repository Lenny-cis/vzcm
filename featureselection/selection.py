# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:31:20 2020

@author: linjianing
"""


import pandas as pd
from collections import OrderedDict
from .description import df_mrcr
from performance.var_perf import df_perf


slc_dic = OrderedDict({'nonmissing_ratio': 0.5,
                       'nonconcentration_ratio': 0.05,
                       'iv': 0.05})


def features_selection(df, y, thrd):
    """特征筛选."""
    mrcr = df_mrcr(df, y)
    perf = df_perf(df, y)
    slc_dict = mrcr
    slc_dict.update(perf)
    df_slc = pd.DataFrame.from_dict(slc_dict, orient='index')
    handle_dict = {}
    for th in thrd:
        pre_n = df_slc.shape[0]
        df_slc = df_slc.loc[df_slc.loc[:, th] >= thrd[th], :]
        left_n = df_slc.shape[0]
        handle_dict.update({th: {'pre': pre_n, 'handle': pre_n - left_n}})

    perf_dict = df_slc.to_dict(orient='index')
    return perf_dict, handle_dict
