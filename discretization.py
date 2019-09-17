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
from imp import reload

os.chdir(r'E:\微众税银\C01 代码\W01_vzcm')
import utils as ut
reload(ut)


class Discretization:
    def __init__(self, df, col_var, dep_var='FLAG', n_cut=50, I_min=3,
                 U_min=4, cut_mthd='eqqt', suffix='raw_bins_',
                 thrd_PCT=0.03, max_bins=6, prior_shape=np.nan, prec=5):
        self.df = df
        self.dep_var = dep_var
        self.I_min = I_min
        self.U_min = U_min
        self.n_cut = n_cut
        self.cut_mthd = cut_mthd
        self.col_var = col_var
        self.suffix = suffix
        self.thrd_PCT = thrd_PCT
        self.max_bins = max_bins
        self.prior_shape = prior_shape
        self.prec = prec

    def gen_cross(self):
        df = self.df.copy(deep=True)
        cross = ut.gen_cut_cross(df, self.col_var, self.dep_var, n=self.n_cut,
                                 mthd=self.cut_mthd, prec=self.prec)
        cross = ut.merge_PCT_zero(cross)
        cross = ut.merge_PCT_zero(cross, thrd_PCT=self.thrd_PCT, mthd='zero')
        self.cross = cross

    class sub:
        def __init__(self, obj):
            self.obj = obj
            self.col_var = self.obj.col_var

        def genPlot(self):
            WOE = self.WOEDict
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
        cross = self.cross.copy(deep=True)
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
        for i in range(min_bin_num, max_bin_num+1):
            comb = its.combinations(bin_list[1:], i)
            for j in comb:
                merged = ut.merge_bin(cross, list(j))
                shape = ut.bad_rate_shape(merged, I_min, U_min)
                if not pd.isna(shape) and shape in prior_shape:
                    subName = 'categ'+str(s)
                    values = [x.right for x in merged.index if x != -1]
                    values.insert(0, -np.inf)
                    setattr(self, subName, sub(obj=self))
                    setattr(getattr(self, subName), 'id', s)
                    setattr(getattr(self, subName), 'shape', shape)
                    setattr(getattr(self, subName), 'cross', merged)
                    setattr(getattr(self, subName), 'values', values)
                    detail, IV = ut.cal_WOE_IV(merged)
                    WOEDict = detail['WOE'].to_dict()
                    chi, p, dof, expFreq =\
                        sps.chi2_contingency(merged.values,
                                             correction=False)
                    setattr(getattr(self, subName), 'detail', detail)
                    setattr(getattr(self, subName), 'IV', IV)
                    setattr(getattr(self, subName), 'WOEDict', WOEDict)
                    setattr(getattr(self, subName), 'p', p)
                    setattr(getattr(self, subName), 'NumBin', len(merged)-1)
                    s += 1
        self.categories = s

    def select_PIV(self, subName, Dict, mthd, delother):
        NumBin = getattr(getattr(self, subName), 'NumBin')
        if NumBin not in Dict.keys():
            Dict[NumBin] = getattr(self, subName)
        elif mthd == 'p':
            if Dict[NumBin].p > getattr(getattr(self, subName), 'p'):
                Dict[NumBin] = getattr(self, subName)
        else:
            if Dict[NumBin].IV < getattr(getattr(self, subName), 'IV'):
                Dict[NumBin] = getattr(self, subName)
        if delother:
            delattr(self, subName)

    def select_global_best(self, select_dthd='p', del_other=True):
        self.gen_comb()
        categories = self.categories
        selectPIV = self.select_PIV
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
            min_dist_idx = ut.slc_min_dist(nonna)
            nonna = ut.merge_bin(nonna, [min_dist_idx])
            if len(nonna) <= min_bin_num:
                break
            if len(nonna) <= max_bin_num:
                shape = ut.bad_rate_shape(nonna, I_min, U_min)
                if not pd.isna(shape) and shape in prior_shape:
                    merged = nonna.append(nacross)
                    subName = 'local_best_'+str(s)
                    values = [x.right for x in nonna.index if x != -1]
                    values.insert(0, -np.inf)
                    setattr(self, subName, sub(obj=self))
                    setattr(getattr(self, subName), 'shape', shape)
                    setattr(getattr(self, subName), 'cross', merged)
                    setattr(getattr(self, subName), 'values', values)
                    detail, IV = ut.cal_WOE_IV(merged)
                    WOEDict = detail['WOE'].to_dict()
                    chi, p, dof, expFreq =\
                        sps.chi2_contingency(merged.values,
                                             correction=False)
                    setattr(getattr(self, subName), 'detail', detail)
                    setattr(getattr(self, subName), 'IV', IV)
                    setattr(getattr(self, subName), 'WOEDict', WOEDict)
                    setattr(getattr(self, subName), 'p', p)
                    setattr(getattr(self, subName), 'NumBin', len(merged)-1)
                    if np.log(p) <= min_p:
                        min_p = np.log(p)
                        min_sub = subName
                    s += 1
        if not pd.isna(min_sub):
            self.bestC = getattr(self, min_sub)





















