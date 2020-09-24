# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:20:45 2020

@author: linjianing
"""


import pandas as pd
import numpy as np
from varclushi import VarClusHi


class Collinear:
    """
    处理多重共线性问题.

    使用varclus的方式进行变量聚类
    """

    def __init__(self, maxeigval2=1, maxclus=None, n_rs=0):
        self.maxeigval2 = maxeigval2
        self.maxclus = maxclus
        self.n_rs = n_rs

    def fit(self, df, feat_list=None, speedup=False):
        """训练出非共线性变量."""
        vc = VarClusHi(df, feat_list, self.maxeigval2, self.maxclus, self.n_rs)
        vc.varclus(speedup)
        vc_rs = vc.rsquare
        cls_fst_var = vc_rs.sort_values(by=['RS_Ratio']).groupby(['Cluster']).head(1).loc[:, 'Variable']
        self.rsquare = vc_rs
        self.info = vc.info
        self.nocoll_var = cls_fst_var
        return self
