# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:41:56 2020

@author: linjianing
"""


import numpy as np
import math
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from model.stepwise import stepwise_selection


class LRModel:
    """逻辑回归模型."""

    def __init__(self, X_df, y_ser):
        self.X_df = X_df
        self.y_ser = y_ser
        self.init_vars = X_df.columns.to_list()
        self.res_vars = self.init_vars

    def lass_fit(self):
        """lasso逻辑回归."""
        X_df = self.X_df.loc[:, self.res_vars].copy(deep=True)
        y_ser = self.y_ser.copy(deep=True)
        X_names = X_df.columns.to_list()
        params = {'C': 1/np.logspace(np.log(1e-6), np.log(1), 50, base=math.e)}
        lass_lr = LogisticRegression(penalty='l1', solver='liblinear')
        while True:
            gscv = GridSearchCV(lass_lr, params)
            gscv.fit(X_df, y_ser)
            if sum(lass_lr.coef_ < 0) <= 0:
                break
            X_names = [k for k, v in dict(zip(X_names, lass_lr.coef_.to_list())).items() if v > 0]
            X_df = X_df.loc[:, X_names]

        best_params = lass_lr.get_params()
        lr = LogisticRegression(penalty='l1', **best_params, solver='liblinear')
        lr.fit(X_df, y_ser)
        coef_dict = dict(zip(X_names, lr.coef_.to_list()))
        self.lasso_vars = [k for k, v in coef_dict.items() if v > 0]
        self.res_vars = self.lasso_vars
        return self

    def stepwise_fit(self, threshold_in=0.01, threshold_out=0.05, verbose=True):
        """逐步回归."""
        X_df = self.X_df.loc[:, self.res_vars].copy(deep=True)
        step_out = stepwise_selection(X_df, self.y_ser, threshold_in=0.01, threshold_out=0.05, verbose=True)
        self.stepwise_vars = step_out
        self.res_vars = self.stepwise_vars
        return self

    def final_fit(self):
        """最终回归."""
        X_df = self.X_df.loc[:, self.res_vars].copy(deep=True)
        y_ser = self.y_ser.copy(deep=True)
        lr = sm.Logit(y_ser, sm.add_constant(X_df)).fit()
        self.model_ = lr
        return self
