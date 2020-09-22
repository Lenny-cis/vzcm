# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:16:35 2020

@author: linjianing
"""


import pandas as pd
import numpy as np
from binning.globalchi import Binning


test_raw = pd.DataFrame()
for i, (e, noe) in enumerate(zip([40, 41, 80, 199, 200], [360, 359, 320, 201, 200])):
    t_df = pd.DataFrame({'x': i + np.random.random(e + noe) / 100, 'flag': [1] * e + [0] * noe})
    test_raw = test_raw.append(t_df, ignore_index=True)
test_raw.index = test_raw.index + 10000

test_raw.groupby(['x', 'flag']).size().unstack()
var_dic = {'x': {'prior_shape': 'I', 'slc_mthd': 'entropy'}}

test_bin = Binning(var_dic, cut_cnt=5)
test_bin.fit(test_raw.loc[:, ['x']], test_raw.loc[:, 'flag'])
test_x = test_bin.transform(test_raw.loc[:, ['x']])
print(test_bin.x.best_dic)
test = pd.concat([test_raw.loc[:, 'flag'], test_x], axis=1)
g = test.groupby(['x_I_3', 'flag']).size().unstack()
g.loc[:, 'p'] = g.loc[:, 1]/g.sum(axis=1)
print(g)


test_raw = pd.DataFrame()
for i, (e, noe) in enumerate(zip([40, 41, 80, 202, 200], [360, 359, 320, 198, 200])):
    t_df = pd.DataFrame({'x': [str(i+10)] * (e + noe), 'flag': [1] * e + [0] * noe})
    test_raw = test_raw.append(t_df, ignore_index=True)
    t_df.loc[:, 'x'] = np.nan
    test_raw = test_raw.append(t_df, ignore_index=True)
test_raw.index = test_raw.index + 10000
test_raw.loc[:, 'x'] = test_raw.loc[:, 'x'].astype(pd.CategoricalDtype(['10', '11', '12', '13', '14'], ordered=False))

test_raw.groupby(['x', 'flag']).size().unstack()
var_dic = {'x': {'prior_shape': 'I', 'slc_mthd': 'entropy'}}

test_bin = Binning(var_dic, cut_cnt=5)
test_bin.fit(test_raw.loc[:, ['x']], test_raw.loc[:, 'flag'])
test_x = test_bin.transform(test_raw.loc[:, ['x']])
print(test_bin.x.best_dic)
test = pd.concat([test_raw.loc[:, 'flag'], test_x], axis=1)
g = test.groupby(['x_I_3', 'flag']).size().unstack()
g.loc[:, 'p'] = g.loc[:, 1]/g.sum(axis=1)
print(g)
