# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:22:35 2019

@author: linjn
"""

import pandas as pd
import numpy as np
import itertools as its
import scipy.stats as sps
import os
from matplotlib import pyplot as plt
import utils as utl


class Discretization:
    """
    卡方分箱，可用全局最优和局部最优方法.

    若变量是有序分类变量，则I_min最小为2，U_min最小为3。
    返回一个类，类属性有categories，表示分箱方法的种数
    应用select_global_best方法后增加类属性，best、bestU、bestI、bestD，
    分别对应最优分类的可能形状，最优的U形分类可能的箱数，最优递增分类可能的箱数，
    最优递减分类可能的箱数。
    var.bestU[n]有detail，score，cut，IV这些属性。
    input
        df          原始数据集，原值
        col_var     待分箱变量
        dep_var     应变量
        n_cut       初始切分数
        I_min       单调最小组数
        U_min       U形最小组数
        cut_mthd    切分方法
            eqqt    等频
            eqdist  等距
            categ   有序分类变量
        suffix      组变量前缀
        var_type    变量类型，count/sum/percent/category
        thrd_PCT    分箱过程中最小占比的阈值
        max_bins    分箱最大箱数
        prior_shape 变量形状
        prec        精度
    """
    def __init__(self, df, col_var, dep_var='FLAG', n_cut=50, I_min=3,
                 U_min=4, cut_mthd='eqqt', suffix='raw_bins_', var_type=None,
                 thrd_PCT=0.03, max_bins=6, prior_shape=np.nan):
        self.df = df.loc[:, [col_var, dep_var]]
        self.dep_var = dep_var
        self.var_type = var_type
        if var_type in ['count', 'category']:
            self.I_min = 2
            self.U_min = 3
            self.cut_mthd = 'categ'
        else:
            self.I_min = I_min
            self.U_min = U_min
            self.cut_mthd = cut_mthd
        self.n_cut = n_cut
        self.col_var = col_var
        self.suffix = suffix
        self.thrd_PCT = thrd_PCT
        self.max_bins = max_bins
        self.prior_shape = prior_shape

    def gen_cross(self):
        """
        生成列联表，并对列联表做最低占比阈值和个数为0的合并
        """
        df = self.df.copy(deep=True)
        cross, cut, categ_cut = utl.gen_cut_cross(
                df, self.col_var, self.dep_var, n=self.n_cut,
                mthd=self.cut_mthd, vtype=self.var_type)
        if type(cut).__name__ == 'list':
            cross, cut = utl.merge_PCT_zero(cross, cut)
            cross, cut = utl.merge_PCT_zero(
                    cross, cut, thrd_PCT=self.thrd_PCT, mthd='zero')
        self.cross = cross
        self.orig_cut = cut
        self.categ_cut = categ_cut

    class sub:
        def __init__(self, obj):
            self.obj = obj
            self.col_var = self.obj.col_var

        def genPlot(self):
            WOE = self.score
            col_var = self.col_var
            IV = self.IV
            P = self.p
            fig, ax = plt.subplots(figsize=(10, 8))
            xlabels = [i for i in WOE.keys() if i != -1]
            y = [WOE[xi] for xi in xlabels]
            x = range(len(xlabels))
            ax.plot(x, y)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels)
            plt.xticks(rotation=-45)
            ax.set_title('%s\n ID=%d\n IV=%.5f\n P=%.5f'
                         % (col_var, self.id, IV, P))
            fig.show()

    def gen_comb(self):
        """
        生成全部切点的排列组合，计算所有的WOE、IV、p、detail等信息
        """
        cross = self.cross.copy(deep=True)
        I_min = self.I_min
        U_min = self.U_min
        max_bins = self.max_bins
        sub = self.sub
        orig_cut = self.orig_cut
        prior_shape = list(self.prior_shape)
        bin_list = [x for x in cross.index[:] if x != -1]
        bin_num = len(bin_list)
        # 限定分组数的上下限
        max_bin_num = bin_num - min(I_min, U_min)
        min_bin_num = max(bin_num - max_bins, 1)
        s = 0
        for i in range(min_bin_num, max_bin_num+1):
            comb = its.combinations(bin_list[1:], i)
            for j in comb:
                # 根据选取的切点合并列联表
                merged, cut = utl.merge_bin(cross, list(j), orig_cut)
                shape = utl.bad_rate_shape(merged, I_min, U_min)
                # badrate的形状符合先验形状的分箱方式保留下来
                if not pd.isna(shape) and shape in prior_shape:
                    subName = 'categ'+str(s)
                    setattr(self, subName, sub(obj=self))
                    setattr(getattr(self, subName), 'id', s)
                    setattr(getattr(self, subName), 'shape', shape)
                    setattr(getattr(self, subName), 'cross', merged)
                    setattr(getattr(self, subName), 'cut', cut)
                    # 计算p值、散度和IV
                    detail, IV = utl.cal_WOE_IV(merged)
                    detail.loc[detail.index != -1, 'infimum'] = cut[:-1]
                    detail.loc[detail.index != -1, 'supremum'] = cut[1:]
                    WOEDict = detail['WOE'].to_dict()
                    chi, p, dof, expFreq =\
                        sps.chi2_contingency(
                                merged.loc[~merged.index.isin([-1]), :].values,
                                correction=False)
                    var_entropy = sps.entropy(
                            detail.loc[detail.index != -1, 'All'])
                    setattr(getattr(self, subName), 'detail', detail)
                    setattr(getattr(self, subName), 'IV', IV)
                    setattr(getattr(self, subName), 'score', WOEDict)
                    setattr(getattr(self, subName), 'p', p)
                    setattr(getattr(self, subName), 'NumBin', len(merged)-1)
                    setattr(getattr(self, subName), 'entropy', var_entropy)
                    s += 1
        self.categories = s

    def select_var(self, subName, Dict, mthd, delother):
        """
        选择分箱方式
        input
            subName         子类的名称
            Dict            不同形状的最优分类字典
            mthd            选择方法
                p           选择p值最小的
                IV          选择IV最大的
                entropy     选择entropy最大的
            delother        是否删除其他子类
        """
        NumBin = getattr(getattr(self, subName), 'NumBin')
        if NumBin not in Dict.keys():
            Dict[NumBin] = getattr(self, subName)

        elif mthd == 'p':
            if Dict[NumBin].p > getattr(getattr(self, subName), 'p'):
                Dict[NumBin] = getattr(self, subName)

        elif mthd == 'IV':
            if Dict[NumBin].IV < getattr(getattr(self, subName), 'IV'):
                Dict[NumBin] = getattr(self, subName)

        else:
            if Dict[NumBin].entropy < getattr(
                    getattr(self, subName), 'entropy'):
                Dict[NumBin] = getattr(self, subName)

        if delother:
            delattr(self, subName)

    def select_global_best(self, select_dthd='p', del_other=True):
        """
        通过全局所有可能的切点合并组合，找到相同箱数中最优的合并方式
        input
            select_dthd         选择方式
                p               最小p值
                IV              最大IV
                entropy         最大entropy
            del_other           是否删除其他类
        """
        self.gen_comb()
        categories = self.categories
        selectPIV = self.select_var
        IDict = {}
        DDict = {}
        UDict = {}

        for i in range(categories):
            subName = 'categ'+str(i)
            shape = getattr(getattr(self, subName), 'shape')
            if shape == 'I':
                selectPIV(subName, IDict, select_dthd, del_other)

            elif shape == 'D':
                selectPIV(subName, DDict, select_dthd, del_other)

            elif shape == 'U':
                selectPIV(subName, UDict, select_dthd, del_other)

            self.bestI = IDict
            self.bestD = DDict
            self.bestU = UDict
        rev = False if select_dthd == 'p' else True
        self.best = {}
        for s in ['I', 'D', 'U']:
            try:
                self.best[s] = sorted(locals()[s+'Dict'].items(),
                                      key=lambda x: getattr(x[1], select_dthd),
                                      reverse=rev)[0][1]
            except IndexError:
                pass

    def select_local_best(self):
        """
        通过每次取局部最优达到最后的分箱，每次取局部最后取每减少一箱中最优的合并。
        """
        cross = self.cross.copy(deep=True)
        nonna = cross.loc[cross.index != -1, :]
        nacross = cross.loc[cross.index == -1, :]
        I_min = self.I_min
        U_min = self.U_min
        max_bins = self.max_bins
        sub = self.sub
        prior_shape = list(self.prior_shape)
        bin_list = [x for x in cross.index[:] if x != -1]
        bin_num = len(bin_list)
        max_bin_num = bin_num - min(I_min, U_min)
        min_bin_num = max(bin_num - max_bins, 1)
        s = 0
        min_p = np.inf
        min_sub = np.nan
        while True:
            min_dist_idx = utl.slc_min_dist(nonna)
            nonna = utl.merge_bin(nonna, [min_dist_idx])
            if len(nonna) <= min_bin_num:
                break
            if len(nonna) <= max_bin_num:
                shape = utl.bad_rate_shape(nonna, I_min, U_min)
                if not pd.isna(shape) and shape in prior_shape:
                    merged = nonna.append(nacross)
                    subName = 'local_best_'+str(s)
                    values = [x.right for x in nonna.index if x != -1]
                    values.insert(0, -np.inf)
                    setattr(self, subName, sub(obj=self))
                    setattr(getattr(self, subName), 'shape', shape)
                    setattr(getattr(self, subName), 'cross', merged)
                    setattr(getattr(self, subName), 'values', values)
                    detail, IV = utl.cal_WOE_IV(merged)
                    WOEDict = detail['WOE'].to_dict()
                    chi, p, dof, expFreq =\
                        sps.chi2_contingency(merged.values,
                                             correction=False)
                    setattr(getattr(self, subName), 'detail', detail)
                    setattr(getattr(self, subName), 'IV', IV)
                    setattr(getattr(self, subName), 'score', WOEDict)
                    setattr(getattr(self, subName), 'p', p)
                    setattr(getattr(self, subName), 'NumBin', len(merged)-1)
                    if np.log(p) <= min_p:
                        min_p = np.log(p)
                        min_sub = subName
                    s += 1
        if not pd.isna(min_sub):
            self.bestC = getattr(self, min_sub)


def vars_discrete(df, dep_var='FLAG', n_cut=50, I_min=3, U_min=4,
                  cut_mthd='eqqt', suffix='raw_bins_', thrd_PCT=0.03,
                  max_bins=6, prior_shape=None, prec=5, select_dthd='p'):
    """
    将所有变量进行分箱操作。
    input
        df          原始数据，原值
        dep_var     应变量
        n_cut       原始分箱数
        I_min       单调最小箱数
        U_min       U形最小箱数
        cut_mthd    分箱方式
        suffix      分箱中间前缀
        thrd_PCT    最小占比阈值
        max_bins    粗分箱最大箱数
        prior_shape 先验形状，传入dataframe，需要包含VAR、SHAPE、TYPE字段
        prec        精度
        select_dthd 选择最优分箱的标准
    """
    # 只对有先验形状的变量进行分箱
    if prior_shape is not None:
        names = list(set(df.columns)-set(dep_var) & set(prior_shape['VAR']))
    else:
        names = list(set(df.columns)-set(dep_var))
    i = 0
    n = len(names)
    ret_names = names[:]
    for name in names:
        i += 1
        if prior_shape is not None:
            prior = prior_shape.query('VAR=="'+name+'"')['SHAPE'].iloc[0]
            v_type = prior_shape.query('VAR=="'+name+'"')['TYPE'].iloc[0]
        else:
            prior = np.nan
            v_type = None
        print('%d/%d %s %s' % (i, n, name, prior))
        globals()[name] = Discretization(
                df, col_var=name, dep_var=dep_var, n_cut=n_cut, I_min=I_min,
                U_min=U_min, cut_mthd=cut_mthd, suffix=suffix, var_type=v_type,
                thrd_PCT=thrd_PCT, max_bins=max_bins, prior_shape=prior,
                prec=prec)
        globals()[name].gen_cross()
        # 切点错误，不分箱
        if type(globals()[name].orig_cut).__name__ == 'list':
            globals()[name].select_global_best(select_dthd=select_dthd)
        # 无分箱可能，删除变量
        if hasattr(globals()[name], 'categories'):
            if globals()[name].categories <= 0:
                ret_names.remove(name)
                del globals()[name]
        else:
            ret_names.remove(name)
            del globals()[name]
    return ret_names


def select_fea_IV(names, IV_thrd=0.1):
    """
    根据IV筛选变量，返回超过阈值的变量和全部变量的IV
    input
        names           变量名集合
        IV_thrd         IV下限
    """
    ret_names = []
    IV_df = pd.DataFrame()
    for name in names:
        for s in getattr(globals()[name], 'best'):
            for i in getattr(globals()[name], 'best'+s):
                IV = getattr(globals()[name], 'best'+s)[i].IV
                IV_df = IV_df.append(pd.DataFrame(
                        {'VAR': name+'_'+s+'_'+str(i), 'IV': IV},
                        index=[name+'_'+s+'_'+str(i)]))

                if IV >= IV_thrd:
                    ret_names.append(name+'_'+s+'_'+str(i))
    return ret_names, IV_df


def discrete_apply(df, names):
    """
    分箱结果应用到数据集
    input
        df          原始数据，原值
        names       需要应用的变量名集合
    """
    t_df = df.copy(deep=True)
    for name in names:
        for s in getattr(globals()[name], 'best'):
            for i in getattr(globals()[name], 'best'+s):
                cut = getattr(globals()[name], 'best'+s)[i].cut
                dic = getattr(globals()[name], 'best'+s)[i].score

                t_df[name+'_'+s+'_'+str(i)] = pd.cut(
                        t_df[name], cut, labels=False)
                # 缺失组改为-1
                t_df[name+'_'+s+'_'+str(i)].replace(np.nan, -1, inplace=True)
                t_df[name+'_'+s+'_'+str(i)] = t_df[
                        name+'_'+s+'_'+str(i)].map(dic)

    t_df.drop(names, axis=1, inplace=True)
    return t_df
