# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:20:45 2020

@author: linjianing
"""


import pandas as pd
import numpy as np
from varclushi import VarClusHi


def collinear(df, maxeigval2=1, maxclus=None, n_rs=0, feat_list=None, speedup=False):
    """
    处理多重共线性问题.

    使用varclus的方式进行变量聚类
    """
    vc = VarClusHi(df, feat_list, maxeigval2, maxclus, n_rs)
    vc.varclus(speedup)
    vc_rs = vc.rsquare
    cls_fst_var = vc_rs.sort_values(by=['RS_Ratio']).groupby(['Cluster']).head(1).loc[:, 'Variable']
    return cls_fst_var, vc.info, vc_rs
