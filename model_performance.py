# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:28:17 2019

@author: 86182

model performance
"""

import numpy as np
import scipy.stats as sps
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.stats as sms
import pandas as pd
from sklearn.metrics import roc_curve, auc


def gen_KS(y, pred):
    fpr, tpr, thrd = roc_curve(y, pred)
    eventNum = np.sum(y)
    allNum = len(y)
    nonEventNum = allNum - eventNum
    ksTile = (eventNum*tpr + nonEventNum*fpr)/allNum
    return np.max(tpr - fpr), tpr - fpr, ksTile


def calMcNemar(mat):
    return sps.chi2.sf(np.square(np.abs(mat[1, 0]-mat[0, 1])-1)
                       / (mat[1, 0]+mat[0, 1]), 1)


def calIV(mat):
    magc = mat.sum(axis=0)
    return ((mat[:, 1]/magc[1] - mat[:, 0]/magc[0])*np.log(
            (mat[:, 1]/magc[1]) / (mat[:, 0]/magc[0]))).sum()


def calPSI(mat):
    return calIV(mat)


def genVIFSet(X):
    X_names = X.columns
    X_val = X.values
    ncol = len(X_names)
    vifl = []
    for i in range(ncol):
        vif = sms.outliers_influence.variance_inflation_factor(X_val, i)
        vifl.append(vif)
    return pd.DataFrame({'Variable': X_names, 'VIF': vifl})


def plotROCKS(y, pred):
    xmajorLocator = matplotlib.ticker.MaxNLocator(6)
    xminorLocator = matplotlib.ticker.MaxNLocator(11)
    fpr, tpr, thrd = roc_curve(y, pred)
    ks_stp, w, ksTile = gen_KS(y, pred)
    auc_stp = auc(fpr, tpr)
    ks_x = fpr[w.argmax()]
    ks_y = tpr[w.argmax()]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
    ax[0].plot(fpr, tpr, 'r-', label='AUC=%.5f' % auc_stp, linewidth=0.5)
    ax[0].plot([0, 1], [0, 1], '-', color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax[0].plot([ks_x, ks_x], [ks_x, ks_y], 'r--', linewidth=0.5)
    ax[0].text(ks_x, (ks_x+ks_y)/2, '  KS=%.5f' % ks_stp)
    ax[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='FPR', ylabel='TPR',
              title='Receiver Operating Characteristic')
    ax[0].xaxis.set_major_locator(xmajorLocator)
    ax[0].xaxis.set_minor_locator(xminorLocator)
    ax[0].yaxis.set_minor_locator(xminorLocator)
    ax[0].fill_between(fpr, tpr, color='red', alpha=0.1)
    ax[0].legend()
    ax[0].grid(alpha=0.5, which='minor')
    ax[1].set_title('KS')
    allNum = len(y)
    eventNum = np.sum(y)
    nonEventNum = allNum - eventNum
    ks_p_x = (eventNum*ks_y + nonEventNum*ks_x)/allNum
    ax[1].plot(ksTile, w, 'r-', linewidth=0.5)
    ax[1].plot(ksTile, fpr, '-', color=(0.6, 0.6, 0.6),
               label='Good', linewidth=0.5)
    ax[1].text(ks_p_x, ks_y+0.05, 'Bad', color=(0.6, 0.6, 0.6))
    ax[1].plot(ksTile, tpr, '-', color=(0.6, 0.6, 0.6),
               label='Bad', linewidth=0.5)
    ax[1].text(ks_p_x, ks_x-0.05, 'Good', color=(0.6, 0.6, 0.6))
    ax[1].plot([ks_p_x, ks_p_x], [ks_stp, 0], 'r--', linewidth=0.5)
    ax[1].text(ks_p_x, ks_stp/2, '  KS=%.5f' % ks_stp)
    ax[1].set(xlim=(0, 1), ylim=(0, 1), xlabel='Prop', ylabel='TPR/FPR',
              title='KS')
    ax[1].xaxis.set_major_locator(xmajorLocator)
    ax[1].xaxis.set_minor_locator(xminorLocator)
    ax[1].yaxis.set_minor_locator(xminorLocator)
    ax[1].grid(alpha=0.5, which='minor')
    return fig


def gen_gaintable(df, pred, y, bins=20, output=False):
    t_df = df.loc[:, [pred, y]].query(y+' in([0, 1])')
    t_df = t_df.sort_values(by=[pred], ascending=True)
    t_df['range'] = range(len(t_df))
    t_df['cut'] = pd.cut(t_df['range'], bins)
    t_df['total'] = 1
    total_bad_num = t_df[y].sum()
    total_good_num = len(t_df)-total_bad_num
    score_df = t_df.groupby(['cut'])[pred]\
        .agg(min_score='min', max_score='max')
    score_df = score_df.applymap(lambda x: round(x, 4))
    score_df['score_range'] = score_df.apply(
            lambda x: pd.Interval(x['min_score'], x['max_score']), axis=1)
    num_df = t_df.groupby(['cut'])[y].agg(
            bad_num='sum', total='count')
    num_df['good_num'] = num_df['total']-num_df['bad_num']
    num_df['bad_rate'] = num_df['bad_num']/num_df['total']
    num_df['cum_bad'] = num_df['bad_num'].cumsum()
    num_df['cum_num'] = num_df['total'].cumsum()
    num_df['cum_good'] = num_df['cum_num'] - num_df['cum_bad']
    num_df['cumbad_rate'] = num_df['cum_bad']/num_df['cum_num']
    sample = score_df[['score_range']].merge(num_df, left_index=True,
                                             right_index=True, how='left')
    sample['gain'] = sample['cum_bad']/total_bad_num
    sample['lift'] = sample['gain']/((sample['cum_num'])/(len(t_df)))
    sample['ks'] = np.abs(
            sample['cum_good']/total_good_num
            - sample['cum_bad']/total_bad_num)
    sample.set_index('score_range', inplace=True)
    sample[['bad_rate', 'cumbad_rate', 'gain', 'lift', 'ks']] = sample[
            ['bad_rate', 'cumbad_rate', 'gain', 'lift', 'ks']].applymap(
            lambda x: round(x, 4))
    if output:
        sample.to_csv('gaintable.csv', index=None)
    return sample
