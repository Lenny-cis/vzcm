# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:46:41 2020

@author: linjianing
"""


import sys
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm


def make_tqdm_iterator(**kwargs):
    """产生tqdm进度条迭代器."""
    options = {
        "file": sys.stdout,
        "leave": True
    }
    options.update(kwargs)
    iterator = tqdm(**options)
    return iterator


def is_shape_I(values):
    """判断输入的列表/序列是否为单调递增."""
    if np.array([values[i] < values[i+1]
                for i in range(len(values)-1)]).all():
        return True
    return False


def is_shape_D(values):
    """判断输入的列表/序列是否为单调递减."""
    if np.array([values[i] > values[i+1]
                for i in range(len(values)-1)]).all():
        return True
    return False


def is_shape_U(values):
    """判断输入的列表/序列是否为先单调递减后单调递增."""
    if not (is_shape_I(values) and is_shape_D(values)):
        knee = np.argmin(values)
        if is_shape_D(values[: knee+1]) and is_shape_I(values[knee:]):
            return True
    return False


def is_shape_A(self, values):
    """判断输入的列表/序列是否为先单调递增后单调递减."""
    if not (is_shape_I(values) and is_shape_D(values)):
        knee = np.argmax(values)
        if is_shape_I(values[: knee+1]) and is_shape_D(values[knee:]):
            return True
    return False


def gen_badrate(df):
    """输入bin和[0, 1]的列联表，生成badrate."""
    return df.values[:, 1]/df.values.sum(axis=1)


def slc_min_dist(df):
    """
    选取最小距离.

    计算上下两个bin之间的距离，计算原理参考用惯量类比距离的wald法聚类计算方式
    """
    R_margin = df.sum(axis=1)
    C_margin = df.sum(axis=0)
    n = df.sum().sum()
    A = df.div(R_margin, axis=0)
    R = R_margin/n
    C = C_margin/n
    # 惯量类比距离
    dist = (A-A.shift()).dropna().applymap(np.square)\
        .div(C, axis=1).sum(axis=1)*(R*R.shift()/(R+R.shift())).dropna()
    return dist.idxmin()


def bad_rate_shape(df, I_min, U_min):
    """判断badrate的单调性，限制了单调的最小个数，U形的最小个数."""
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


def gen_cut(ser, n=10, mthd='eqqt'):
    """
    输入序列、切点个数、切分方式、精度生成切点.

    若序列中只有一个值，返回字符串"N_CUT ERROR"
    input
        ser         序列
        n           切点个数
        mthd        切分方式
            eqqt    等频
            eqdist  等距
            categ   有序分类变量
        prec        切分精度
    """
    rcut = list(sorted(ser.dropna().unique()))
    if len(rcut) <= 1:
        return 'N_CUT ERROR'

    # 有序分类变量切点为各个类别
    if mthd == 'categ':
        cut = rcut
        cut.insert(0, -np.inf)
        cut[-1] = np.inf

    # 连续变量切分
    else:
        if mthd == 'eqqt':
            cut = list(pd.qcut(ser, n, retbins=True,
                               duplicates='drop')[1].round(4))

        elif mthd == 'eqdist':
            cut = list(pd.cut(ser, n, retbins=True,
                              duplicates='drop')[1].round(4))

        else:
            return 'MTHD ERROR'

        cut[0] = -np.inf
        cut[-1] = np.inf
    return cut


def gen_cross(df, col_var, dep_var, cut, prec=5, vtype=None):
    """
    生成列联表.

    input
        df          原始数据，原值
        col_var     待切分变量
        dep_var     应变量
        cut         切点
        prec        精度
    """
    # 切分后返回bin[0, 1, ...]
    t_df = df.copy(deep=True)
    if vtype != 'category':
        t_df[col_var] = pd.cut(t_df[col_var], cut, precision=prec,
                               duplicates='drop', labels=False)
    # 生成列联表
    cross = t_df.groupby([col_var, dep_var]).size().unstack()
    allsize = t_df.groupby([dep_var]).size()
    # 生成缺失组，索引为-1
    na_cross = pd.DataFrame({0: np.nansum([allsize[0], -cross.sum()[0]]),
                             1: np.nansum([allsize[1], -cross.sum()[1]])},
                            index=[-1])
    cross = cross.append(na_cross)
    cross.fillna(0, inplace=True)
    return cross


def gen_cross_numeric(df, x, y, n=10, mthd='eqqt'):
    """生成数值字段的列联表."""
    t_df = df.copy(deep=True)
    cut = gen_cut(t_df.loc[:, x], n=n, mthd=mthd)
    if type(cut).__name__ != 'list':
        return None, cut
    # 切分后返回bin[0, 1, ...]
    t_df[x] = pd.cut(t_df[x], cut, labels=False, duplicates='drop')
    cross = t_df.groupby([x, y]).size().unstack()
    t_cut = [cut[int(x+1)] for x in cross.index]
    t_cut.insert(0, -np.inf)
    allsize = t_df.groupby([y]).size()
    na_cross = pd.DataFrame({0: np.nansum([allsize[0], -cross.sum()[0]]),
                             1: np.nansum([allsize[1], -cross.sum()[1]])},
                            index=[-1])
    cross.reset_index(inplace=True, drop=True)
    cross = cross.append(na_cross)
    cross.fillna(0, inplace=True)
    return cross, t_cut


def gen_cross_category(df, x, y):
    """生成分类字段的列联表."""
    t_df = df.copy(deep=True)
    cross = t_df.groupby([x, y]).size().unstack()
    if not t_df.loc[:, x].cat.ordered:
        cross['eventRate'] = cross[1]/np.nansum(cross, axis=1)
        cross.sort_values('eventRate', ascending=False, inplace=True)
        cross.drop(['eventRate'], inplace=True, axis=1)
    cut = {v: k for k, v in enumerate(list(cross.index))}
    cross.reset_index(inplace=True, drop=True)
    allsize = t_df.groupby([y]).size()
    na_cross = pd.DataFrame({0: np.nansum([allsize[0], -cross.sum()[0]]),
                             1: np.nansum([allsize[1], -cross.sum()[1]])},
                            index=[-1])
    cross.reset_index(inplace=True, drop=True)
    cross = cross.append(na_cross)
    cross.fillna(0, inplace=True)
    return cross, cut


def gen_cut_cross(df, col_var, dep_var, n=10, mthd='eqqt', vtype=None):
    """
    根据切点个数和切分方法生成列联表.

    input
        df          原始数据，原值
        col_var     待切分变量
        dep_var     应变量
        n           切点个数
        mthd        切分方式，参考gen_cut方法
        prec        精度
    """
    # 生成原始切点
    t_df = df.copy(deep=True)
    if vtype != 'categ':
        cut = gen_cut(t_df[col_var], n=n, mthd=mthd)
        if type(cut).__name__ != 'list':
            return None, cut
        # 切分后返回bin[0, 1, ...]
        t_df[col_var] = pd.cut(t_df[col_var], cut, labels=False,
                               duplicates='drop')
        cross = t_df.groupby([col_var, dep_var]).size().unstack()
        categ_cut = None
    else:
        cross = t_df.groupby([col_var, dep_var]).size().unstack()
        cross['eventRate'] = cross[1]/np.nansum(cross, axis=1)
        cross.sort_values('eventRate', ascending=False, inplace=True)
        cross.drop(['eventRate'], inplace=True, axis=1)
        categ_cut = list(cross.index)
        cross.reset_index(inplace=True, drop=True)
        cut = list(cross.index)
        cut.insert(0, -np.inf)
        cut.append(np.inf)
    # 调整切点生成下限
    t_cut = [cut[int(x+1)] for x in cross.index]
    t_cut.insert(0, -np.inf)
    # 生成缺失组
    allsize = t_df.groupby([dep_var]).size()
    na_cross = pd.DataFrame({0: np.nansum([allsize[0], -cross.sum()[0]]),
                             1: np.nansum([allsize[1], -cross.sum()[1]])},
                            index=[-1])
    cross.reset_index(inplace=True, drop=True)
    cross = cross.append(na_cross)
    cross.fillna(0, inplace=True)
    return cross, t_cut, categ_cut


def cal_WOE_IV(df, modify=True):
    """
    计算WOE、IV及分箱细节.

    input
        df          bin和[0, 1]的列联表
        modify      是否调整缺失组的WOE值
            调整逻辑：将缺失组的WOE限制在除缺失组以外的WOE上下限范围内，保证模型稳定
                     若缺失组的WOE最大，则调整为非缺失组的最大值
                     若缺失组的WOE最小，则调整为0
    """
    warnings.filterwarnings('ignore')
    cross = df.values
    col_margin = cross.sum(axis=0)
    row_margin = cross.sum(axis=1)
    event_rate = cross[:, 1]/row_margin
    event_prop = cross[:, 1]/col_margin[1]
    non_event_prop = cross[:, 0]/col_margin[0]
    # 将0替换为极小值，便于计算，计算后将rate为0的组赋值为其他组的最小值，
    # rate为1的组赋值为其他组的最大值
    WOE = np.log(np.where(event_prop == 0, 0.0005, event_prop)
                 / np.where(non_event_prop == 0, 0.0005, non_event_prop))
    WOE[event_rate == 0] = np.min(WOE[(event_rate != 0) & (df.index != -1)])
    WOE[event_rate == 1] = np.max(WOE[(event_rate != 1) & (df.index != -1)])
    # 调整缺失组的WOE
    if modify is True:
        if WOE[df.index == -1] == max(WOE):
            WOE[df.index == -1] = max(WOE[df.index != -1])
        elif WOE[df.index == -1] == min(WOE):
            WOE[df.index == -1] = 0

    IV = np.where(event_rate == 1, 0, (event_prop-non_event_prop)*WOE)
    warnings.filterwarnings('default')
    return pd.DataFrame({'All': row_margin, 'eventRate': event_rate,
                         'WOE': WOE.round(4), 'IV': IV}, index=df.index),\
        IV.sum()


def merge_bin(df, idxlist, cut):
    """
    合并分箱，返回合并后的列联表和切点，合并过程中不会改变缺失组，向下合并的方式.

    input
        df          bin和[0, 1]的列联表
        idxlist     需要合并的箱的索引，列表格式
        cut         原始切点
    """
    cross = df[df.index != -1].copy(deep=True).values
    cols = df.columns
    # 倒序循环需合并的列表，正序会导致表索引改变，合并出错
    for idx in idxlist[::-1]:
        cross[idx] = cross[idx-1: idx+1].sum(axis=0)
        cross = np.delete(cross, idx-1, axis=0)

    cross = pd.DataFrame(cross, columns=cols)\
        .append(df[df.index == -1])
    # 调整合并后的切点
    t_cut = [x for x in cut if cut.index(x) not in idxlist]
    return cross, t_cut


def merge_bin_by_idx(crs, idxlist):
    """
    合并分箱，返回合并后的列联表和切点，合并过程中不会改变缺失组，向下合并的方式.

    input
        df          bin和[0, 1]的列联表
        idxlist     需要合并的箱的索引，列表格式
    """
    cross = crs[crs.index != -1].copy(deep=True).values
    cols = crs.columns
    # 倒序循环需合并的列表，正序会导致表索引改变，合并出错
    for idx in idxlist[::-1]:
        cross[idx] = cross[idx-1: idx+1].sum(axis=0)
        cross = np.delete(cross, idx-1, axis=0)

    cross = pd.DataFrame(cross, columns=cols)\
        .append(crs[crs.index == -1])
    return cross


def merge_lowpct_zero(df, thrd_PCT=0.05, mthd='PCT'):
    """
    合并个数为0和占比过低的箱，不改变缺失组的结果.

    input
        df          bin和[0, 1]的列联表
        cut         原始切点
        thrd_PCT    占比阈值
        mthd        合并方法
            PCT     合并占比过低的箱
            zero    合并个数为0的箱
    """
    cross = df[df.index != -1].copy(deep=True)
    s = 1
    merge_idxs = []
    while s:
        row_margin = cross.sum(axis=1)
        total = row_margin.sum()
        min_num = row_margin.min()
        # 找到占比最低的组或个数为0的组
        if mthd.upper() == 'PCT':
            min_idx = row_margin.idxmin()

        else:
            zero_idxs = cross[(cross == 0).any(axis=1)].index
            if len(zero_idxs) >= 1:
                min_idx = zero_idxs[0]
                min_num = 0

            else:
                min_num = np.inf
        # 占比低于阈值则合并
        if min_num/total <= thrd_PCT:
            idxs = list(cross.index)
            # 最低占比的组的索引作为需要合并的组
            # sup_idx确定合并索引的上界，上界不超过箱数
            # inf_idx确定合并索引的下界，下界不低于0
            min_idx_row = idxs.index(min_idx)
            sup_idx = idxs[min(len(cross)-1, min_idx_row+1)]
            inf_idx = idxs[max(0, min_idx_row-1)]
            # 需合并组为第一组，向下合并
            if min_idx == idxs[0]:
                merge_idx = idxs[1]
            # 需合并组为最后一组，向上合并
            elif min_idx == idxs[-1]:
                merge_idx = min_idx

            elif sup_idx == inf_idx:
                merge_idx = inf_idx
            # 介于第一组和最后一组之间，找向上或向下最近的组合并

            else:
                merge_idx = slc_min_dist(cross.loc[inf_idx: sup_idx])

            cross = merge_bin_by_idx(cross, [merge_idx])
            merge_idxs.append(merge_idx)
        else:
            s = 0

    return cross.append(df[df.index == -1]), merge_idxs


def merge_PCT_zero(df, cut, thrd_PCT=0.05, mthd='PCT'):
    """
    合并个数为0和占比过低的箱，不改变缺失组的结果.

    input
        df          bin和[0, 1]的列联表
        cut         原始切点
        thrd_PCT    占比阈值
        mthd        合并方法
            PCT     合并占比过低的箱
            zero    合并个数为0的箱
    """
    cross = df[df.index != -1].copy(deep=True)
    s = 1
    while s:
        row_margin = cross.sum(axis=1)
        total = row_margin.sum()
        min_num = row_margin.min()
        # 找到占比最低的组或个数为0的组
        if mthd.upper() == 'PCT':
            min_idx = row_margin.idxmin()

        else:
            zero_idxs = cross[(cross == 0).any(axis=1)].index
            if len(zero_idxs) >= 1:
                min_idx = zero_idxs[0]
                min_num = 0

            else:
                min_num = np.inf
        # 占比低于阈值则合并
        if min_num/total <= thrd_PCT:
            idxs = list(cross.index)
            # 最低占比的组的索引作为需要合并的组
            # sup_idx确定合并索引的上界，上界不超过箱数
            # inf_idx确定合并索引的下界，下界不低于0
            min_idx_row = idxs.index(min_idx)
            sup_idx = idxs[min(len(cross)-1, min_idx_row+1)]
            inf_idx = idxs[max(0, min_idx_row-1)]
            # 需合并组为第一组，向下合并
            if min_idx == idxs[0]:
                min_idx = idxs[1]
                cross, cut = merge_bin(cross, [min_idx], cut)
            # 需合并组为最后一组，向上合并
            elif min_idx == idxs[-1]:
                cross, cut = merge_bin(cross, [min_idx], cut)

            elif sup_idx == inf_idx:
                cross, cut = merge_bin(cross, [inf_idx], cut)
            # 介于第一组和最后一组之间，找向上或向下最近的组合并
            else:
                min_dist_idx = slc_min_dist(cross.loc[inf_idx: sup_idx])
                cross, cut = merge_bin(cross, [min_dist_idx], cut)

        else:
            s = 0

    return cross.append(df[df.index == -1]), cut
