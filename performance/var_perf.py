# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:17:00 2020

@author: linjianing
"""


import pandas as pd
from performance.utils import calks, calauc, caliv
from binning.utils import gen_cross_numeric, gen_cross_category


def var_perf(y, x, pos_label=None):
    """变量效果集合."""
    t_df = pd.DataFrame({'x': x, 'y': y})
    if x.values.dtype.kind in ['f', 'i', 'u']:
        cross, cut = gen_cross_numeric(t_df, 'x', 'y')

    else:
        cross, cut = gen_cross_category(t_df, 'x', 'y')

    iv = caliv(cross.values)
    # 计算ks及auc
    cross.loc[:, 'event_prop'] = cross.loc[:, 1] / cross.sum(axis=1)
    prop_dict = cross.loc[:, 'event_prop'].to_dict()
    if x.values.dtype.kind in ['f', 'i', 'u']:
        new_x = pd.cut(x, cut, labels=False).map(prop_dict)
    else:
        new_x = x.map(cut).map(prop_dict).astype(float)

    new_x.fillna(prop_dict.get(-1, 0), inplace=True)
    ks = calks(y, new_x, pos_label)
    auc = calauc(y, new_x, pos_label)
    return ks, auc, iv


def df_perf(df, y, perf_xs=None):
    """指定变量的效果集合."""
    if perf_xs is None:
        perf_xs = list(set(df.columns) - set([y]))

    elif isinstance(perf_xs, str):
        perf_xs = [perf_xs]

    perf_xs = list(set(df.columns) & set(perf_xs))
    y_ser = df.loc[:, y]
    perf_dict = {}
    for x in perf_xs:
        x_ser = df.loc[:, x]
        ks, auc, iv = var_perf(y_ser, x_ser)
        perf_dict.update({x: {'ks': ks, 'auc': auc, 'iv': iv}})

    return perf_dict
