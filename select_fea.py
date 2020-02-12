# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 11:16:35 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import os
import itertools as its
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

os.chdir(r'E:\微众税银\C01 代码\W01_vzcm')
import utils as ut


def df_to_ser(df):
    ret_ser = pd.Series()
    for x in df:
        t_ser = df.loc[:, x].copy(deep=True)
        t_ser.index = [tuple(sorted([x, ix])) for ix in t_ser.index.to_list()]
        ret_ser = ret_ser.append(t_ser)
    ret_ser = ret_ser[~ret_ser.index.duplicated(keep='first')]
    ret_ser = ret_ser[ret_ser != 1]
    return ret_ser


def step_select_fea(df, corr, thred=0.4):
    t_corr = corr.copy(deep=True)
    print('THRED|VAR_N|BEST_AUC|FEA_N')
    for thr in np.linspace(thred, 0.9999, 10)[::-1]:
        t_fea = obtain_mul_mass(df, t_corr, thred=thr)
        t_corr = corr.loc[t_fea, t_fea]
    return t_fea


def obtain_mul_mass(df, corr, thred=0.4):
    t_corr = corr.copy(deep=True)
    s = 1
    retain_feas = []
    while s:
        most_corr, oth_corr = split_corr(t_corr, thred=thred)
        most_feas = filter_relation(df, most_corr, thred=thred)
        if most_feas is not None:
            retain_feas.extend(most_feas)
        if len(oth_corr) <= 0:
            s = 0
        t_corr = oth_corr
    return retain_feas


def split_corr(corr, thred=0.4):
    t_corr = corr.copy(deep=True)
    t_corr = t_corr[abs(t_corr) > thred]
    most_fea = t_corr.count().idxmax()
    fea_vars = list(t_corr[most_fea].dropna().index)
    oth_vars = list(set(t_corr.index)-set(fea_vars))
    if len(oth_vars) <= 0:
        return corr.loc[fea_vars, fea_vars], pd.DataFrame()
    return corr.loc[fea_vars, fea_vars], corr.loc[oth_vars, oth_vars]


def filter_relation(df, corr, thred=0.4):
    t_corr = corr.copy(deep=True)
    fea_lists = []
    brk = False
    for i in range(1, len(t_corr)+1)[::-1]:
        combs = its.combinations(list(t_corr.index), i)
        for comb in combs:
            print('%d|%d' % (len(comb), len(corr)))
            comb_corr = t_corr.loc[comb, comb]
            if (abs(comb_corr) > thred).sum().sum() <= len(comb_corr):
                fea_lists.append(comb)
                brk = True
        if brk:
            break
    if len(fea_lists) <= 0:
        return None
    auc_list = []
    for fea_l in fea_lists:
        auc = LR_score(df, fea_l)
        auc_list.append(auc)
    auc_idx = np.argmax(auc_list)
    print('%.3f|%5d|%.6f|%5d' % (thred, i, auc_list[auc_idx], len(corr)))
    return list(fea_lists[auc_idx])


def LR_score(df, feas):
    lr = LogisticRegression(penalty='l1', C=1e6, solver='saga', n_jobs=-1)
    lr.fit(df.loc[:, feas], df.loc[:, 'FLAG'])
    pred = lr.predict_proba(df.loc[:, feas])[:, 1]
    return roc_auc_score(df.loc[:, 'FLAG'], pred)






t2 = fea_corr.loc[fea_name[:50], fea_name[:50]]
tfea = obtain_mul_mass(short_train_yfp, t2, thred=0.7)
tfea = step_select_fea(short_train_yfp, t2, thred=0.7)
t3 = t2.loc[tfea, tfea]
tfea = obtain_mul_mass(short_train_yfp, t3, thred=0.8)
t4 = t2.loc[tfea, tfea]
tfea = obtain_mul_mass(short_train_yfp, t4, thred=0.7)
tfea = filter_relation(short_train_yfp, t2, thred=0.1)

fea_lr = LogisticRegression(penalty='l1', C=1e6,
                            solver='liblinear')
fea_lr.fit(short_train_yfp.loc[:, tfea], short_train_yfp.loc[:, 'FLAG'])
fea_pred = fea_lr.predict_proba(short_train_yfp.loc[:, tfea])
mp.plotROCKS(short_train_yfp.loc[:, 'FLAG'], fea_pred[:, 1], ks_label='TRAIN')
fea_lr.coef_
auc = roc_auc_score(short_train_yfp.loc[:, 'FLAG'], fea_pred)

lr = sm.Logit(short_train_yfp.loc[:, 'FLAG'],
              sm.add_constant(short_train_yfp.loc[:, tfea]))
lr_rst = lr.fit()
# lr_coef = lr_rst.summary2().tables[1]
lr_pred = lr_rst.predict(sm.add_constant(short_train_yfp.loc[:, tfea]))
mp.plotROCKS(short_train_yfp.loc[:, 'FLAG'], lr_pred, ks_label='TRAIN')

t2 = fea_corr.loc[fea_name[:10], fea_name[:10]]

t3 = t2.loc[tfea, tfea]
t4 = (abs(t3)>0.5).sum().any().any()
t13 = random.choices(t8.index, k=10)
t14 = t7.loc[t13, t13]
print(t14.values)
t5 = short_train_yfp.loc[:, 'FLAG']


def fuc_1(x):
    t_x = x.values
    x_col = x.columns
    x_index = x.index
    return pd.DataFrame(np.where(t_x>0.4, t_x, np.nan), columns=x_col, index=x_index)

def fuc_2(x):
    return x[x>0.4]

%timeit fuc_1(t)
%timeit fuc_2(t)



def df_to_ser(df):
    ret_ser = pd.Series()
    for x in df:
        t_ser = df.loc[:, x].copy(deep=True)
        t_ser.index = [{x, ix} for ix in t_ser.index.to_list()]


df_to_ser(fea_corr)





{1, 0} == {0, 1}












