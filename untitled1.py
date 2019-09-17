# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:53:08 2019

@author: linjn
"""
import pandas as pd
import numpy as np

test = pd.DataFrame({1: [np.nan, 60, 100, 30, 200],
                     0: [200, 121, 300, 300, 500]},
                    index=[pd.Interval(1, 2), pd.Interval(2, 3),
                           pd.Interval(3, 4), pd.Interval(4, 5), -1])
np.nansum(test.values,axis=1)/test.sum().sum()
R_margin = test.sum(axis=1)
C_margin = test.sum(axis=0)
n = test.sum().sum()

A = test.div(R_margin, axis=0)
R = R_margin/n
C = C_margin/n
dist = (A-A.shift()).dropna().applymap(np.square)\
    .div(C, axis=1).sum(axis=1)*(R*R.shift()/(R+R.shift())).dropna()

slc_min_dist(test[1:4])


def a1():
    cross = test.values
    colMargin = cross.sum(axis=0)
    rowMargin = cross.sum(axis=1)
    eventRate = cross[:, 1]/rowMargin
    eventProp = cross[:, 1]/colMargin[1]
    nonEventProp = cross[:, 0]/colMargin[0]
    WOE = np.log(np.where(eventProp == 0, 0.0005, eventProp)
                 / np.where(nonEventProp == 0, 0.0005, nonEventProp))
    WOE[eventRate == 0] = np.min(WOE[(eventRate != 0)&(test.index != -1)])
    WOE[eventRate == 1] = np.max(WOE[(eventRate != 1)&(test.index != -1)])
    WOE[test.index == -1] = max(0, WOE[test.index == -1])
    IV = np.where(eventRate == 1, 0, (eventProp-nonEventProp)*WOE)
    cross = pd.DataFrame({
            'All': rowMargin,
            'eventRate': eventRate,
            'WOE': WOE.round(4),
            'IV': IV}, index=test.index)
    return cross, IV.sum()


def a2():
    cross = np.mat(test.values)
    colMargin = cross.sum(axis=0)
    rowMargin = cross.sum(axis=1)
    eventRate = cross[:, 1]/rowMargin
    eventProp = cross[:, 1]/colMargin[:, 1]
    nonEventProp = cross[:, 0]/colMargin[:, 0]
    WOE = np.log(np.where(eventProp == 0, 0.0005, eventProp)
                 / np.where(nonEventProp == 0, 0.0005, nonEventProp))
    WOE[eventRate == 0] = np.min(WOE[(eventRate != 0)&np.mat(test.index != -1).T])
    WOE[eventRate == 1] = np.max(WOE[(eventRate != 1)&np.mat(test.index != -1).T])
    WOE[test.index == -1] = max(0, WOE[test.index == -1])
    IV = (eventProp-nonEventProp).T*np.where(eventRate == 1, 0, WOE)
    cross = pd.DataFrame({
            'All': np.array(rowMargin.T)[0],
            'eventRate': np.array(eventRate.T)[0],
            'WOE': WOE.round(4).T[0]}, index=test.index)
    return cross, IV


%timeit a1()
%timeit a2()


merge_bin(test, [pd.Interval(2, 3), pd.Interval(3, 4)])

merge_PCT_zero(test, 0.2, mthd='PCT')
sps.chi2_contingency([[100, 200], [60, 421], [30, 300]], correction=False)
sps.chi2_contingency([[100, 200], [60, 121], [30, 600]], correction=False)
sps.chi2_contingency(test.values, correction=False)



c = Discretization(tx, 'tx', n_cut=20, prior_shape='D')
c.gen_cross()
c.cross
c.gen_comb()
c.categories
c.selectBest(select_dthd='p')
c.select_local_best()
c.bestC
lis = [pd.Interval(2, 3)]
for l in lis:
    print(l)



os.chdir(r'H:\work\微众税银\T02 临时工作\20190711 模型代码验证')
test_data = pd.read_csv(r'企金贷前_20190711.csv')
cv = Discretization(test_data, 'CV_12', prior_shape='I')
cv.gen_cross()
cv.select_global_best()
cv.bestI[6].detail
t = cv.cross
t[1]/t.sum(axis=1)






