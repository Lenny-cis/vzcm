# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:46:06 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import os
from sklearn.utils import resample
os.chdir(r'D:\微众税银\C01 代码\W01_vzcm')
from utils import gen_cut
from model_performance import calPSI


def gen_bootstrap_sample(sample, dep_var='FLAG', random=0):
    bootstrap = resample(sample, n_samples=len(sample),
                         random_state=random)
    oob = sample.reindex(sample.index.difference(bootstrap.index))
    return bootstrap, oob


def ser_group(ser, cut, prec=5):
    bin_ser = pd.cut(ser, cut, precision=prec, labels=False)
    group = bin_ser.value_counts()
    g_name = group.name
    group = group.append(pd.Series(ser.isna().sum(), index=[-1]))
    group.name = g_name
    return group


def var_PSI(base_ser, new_ser, n=10, mthd='eqqt', prec=5):
    var = base_ser.name
    cut = gen_cut(base_ser, n=n, mthd=mthd, prec=prec)
    if cut in ['MTHD ERROR', 'N_CUT ERROR']:
        return var
    base_group = ser_group(base_ser, cut, prec=prec)
    new_group = ser_group(new_ser, cut, prec=prec)
    return pd.Series(calPSI(base_group, new_group), index=[var])


def sample_PSI(base_df, new_df, dep_var='FLAG', n=10, mthd='eqqt', prec=5):
    var_names = list(set(base_df.columns)-set(list(dep_var)))
    vars_PSI = pd.Series()
    for var in var_names:
        PSI = var_PSI(base_df[var], new_df[var], n=n, mthd=mthd, prec=prec)
        if type(PSI).__name__ != 'str':
            vars_PSI = vars_PSI.append(PSI)
    return vars_PSI


def exclude_one_value(df):
    drop_vars = []
    for v in df.columns:
        n_value = len(df[v].dropna().unique())
        if n_value == 1:
            drop_vars.append(v)
    return df[list(set(df.columns)-set(drop_vars))], drop_vars


def bootstrapping_byPSI(sample, dep_var='FLAG', n=10,
                        prec=5, mthd='eqqt', thrd=0.01):
    s = 1
    print('min_psi|mean_psi|max_psi')
    while s:
        i = np.random.randint(10000)
        boot, oob = gen_bootstrap_sample(sample, dep_var=dep_var, random=i)
        PSI = sample_PSI(boot, oob, dep_var=dep_var,
                         n=n, mthd=mthd, prec=prec)
        print(' %.4f|  %.4f| %.4f' % (PSI.min(), PSI.mean(), PSI.max()))
        if (PSI < thrd).all():
            s = 0
            break
    return boot, oob, i
