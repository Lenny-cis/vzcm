# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:38:28 2019

@author: linjn
"""

import numpy as np
import pandas as pd


def is_shape_I(values):
    if np.array([values[i] < values[i+1]
                for i in range(len(values)-1)]).all():
        return True
    return False


def is_shape_D(values):
    if np.array([values[i] > values[i+1]
                for i in range(len(values)-1)]).all():
        return True
    return False


def is_shape_U(values):
    if not (is_shape_I(values) and is_shape_D(values)):
        knee = np.argmin(values)
        if is_shape_D(values[: knee+1]) and is_shape_I(values[knee:]):
            return True
    return False


def is_shape_A(self, values):
    if not (is_shape_I(values) and is_shape_D(values)):
        knee = np.argmax(values)
        if is_shape_I(values[: knee+1]) and is_shape_D(values[knee:]):
            return True
    return False


def gen_badrate(df):
    return df.values[:, 1]/df.values.sum(axis=1)


def slc_min_dist(df):
    R_margin = df.sum(axis=1)
    C_margin = df.sum(axis=0)
    n = df.sum().sum()
    A = df.div(R_margin, axis=0)
    R = R_margin/n
    C = C_margin/n
    dist = (A-A.shift()).dropna().applymap(np.square)\
        .div(C, axis=1).sum(axis=1)*(R*R.shift()/(R+R.shift())).dropna()
    return dist.idxmin()


def bad_rate_shape(df, I_min, U_min):
    n = len(df)
    badRate = gen_badrate(df[df.index != -1])
    if (n >= I_min):
        if is_shape_I(badRate):
            return 'I'
        elif is_shape_D(badRate):
            return 'D'
    if (n >= U_min):
        if is_shape_U(badRate):
            return 'U'
    return np.nan


def gen_cut(ser, n=10, mthd='eqqt', prec=5):
    rcut = list(sorted(ser.dropna().unique()))
    if len(rcut) == 1:
        return 'N_CUT ERROR'
    if mthd == 'eqqt':
        cut = list(pd.qcut(ser, n, retbins=True,
                           precision=prec, duplicates='drop')[1])
    elif mthd == 'eqdist':
        cut = list(pd.cut(ser, n, retbins=True,
                          precision=prec, duplicates='drop')[1])
    else:
        return 'MTHD ERROR'
    if len(rcut) < n:
        cut = rcut
    cut.insert(0, -np.inf)
    cut.append(np.inf)
    return cut


def gen_cut_cross(df, col_var, dep_var, n=10, mthd='eqqt', prec=5):
    t_df = df.copy(deep=True)
    cut = gen_cut(t_df[col_var], n=n, mthd=mthd, prec=prec)
    t_df[col_var] = pd.cut(t_df[col_var], cut,
                           precision=prec, duplicates='drop')
    cross = t_df.groupby([col_var, dep_var], as_index=False).size().unstack()
    allsize = t_df.groupby([dep_var]).size()
    na_cross = pd.DataFrame({0: np.nansum([allsize[0], -cross.sum()[0]]),
                             1: np.nansum([allsize[1], -cross.sum()[1]])},
                            index=[-1])
    cross.index = cross.index.astype('O')
    cross = cross.append(na_cross)
    cross.fillna(0, inplace=True)
    return cross


def cal_WOE_IV(df):
    cross = df.values
    col_margin = cross.sum(axis=0)
    row_margin = cross.sum(axis=1)
    event_rate = cross[:, 1]/row_margin
    event_prop = cross[:, 1]/col_margin[1]
    non_event_prop = cross[:, 0]/col_margin[0]
    WOE = np.log(np.where(event_prop == 0, 0.0005, event_prop)
                 / np.where(non_event_prop == 0, 0.0005, non_event_prop))
    WOE[event_rate == 0] = np.min(WOE[(event_rate != 0) & (df.index != -1)])
    WOE[event_rate == 1] = np.max(WOE[(event_rate != 1) & (df.index != -1)])
    WOE[df.index == -1] = max(WOE[df.index == -1], 0)
    IV = np.where(event_rate == 1, 0, (event_prop-non_event_prop)*WOE)
    return pd.DataFrame({'All': row_margin, 'eventRate': event_rate,
                         'WOE': WOE.round(4), 'IV': IV}, index=df.index),\
        IV.sum()


def merge_bin(df, idxlist):
    cross = df[df.index != -1].copy(deep=True).values
    idxs = [x for x in df.index if x != -1]
    for idx in idxlist:
        iidx = list(idxs).index(idx)
        cross[iidx] = cross[iidx-1: iidx+1].sum(axis=0)
        cross = np.delete(cross, iidx-1, axis=0)
        idxs[iidx] = pd.Interval(idxs[iidx-1].left, idxs[iidx].right)
        idxs.pop(iidx-1)
    cross = pd.DataFrame(cross, index=idxs).append(df[df.index == -1])
    return cross


def merge_PCT_zero(df, thrd_PCT=0.05, mthd='PCT'):
    cross = df[df.index != -1].copy(deep=True)
    s = 1
    while s:
        row_margin = cross.sum(axis=1)
        total = row_margin.sum()
        min_num = row_margin.min()
        if mthd.upper() == 'PCT':
            min_idx = row_margin.idxmin()
        else:
            zero_idxs = cross[(cross == 0).any(axis=1)].index
            if len(zero_idxs) >= 1:
                min_idx = zero_idxs[0]
                min_num = 0
            else:
                min_num = np.inf
        if min_num/total <= thrd_PCT:
            idxs = list(cross.index)
            min_idx_row = idxs.index(min_idx)
            sup_idx = idxs[min(len(cross)-1, min_idx_row+1)]
            inf_idx = idxs[max(0, min_idx_row-1)]
            if min_idx == idxs[0]:
                min_idx = idxs[1]
                cross = merge_bin(cross, [min_idx])
            elif min_idx == idxs[-1]:
                cross = merge_bin(cross, [min_idx])
            else:
                min_dist_idx = slc_min_dist(cross.loc[inf_idx: sup_idx])
                cross = merge_bin(cross, [min_dist_idx])
        else:
            s = 0
    return cross.append(df[df.index == -1])
