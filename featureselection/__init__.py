# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:53:40 2020

@author: linjianing
"""


import pandas as pd
from collections import OrderedDict
from .description import df_mrcr
from performance.var_perf import df_perf
from utils import update_dict


slc_dic = OrderedDict({'nonmissing_ratio': 0.5,
                       'nonconcentration_ratio': 0.05,
                       'iv': 0.05})


class FeatureSelection:
    """特征筛选."""

    def __init__(self, entityset):
        self.entityset = entityset

    def fit(self, entity_id, thrd_dict=slc_dic):
        """筛选."""
        entity = self.entityset.get_entity(entity_id)
        df = entity.df
        y = entity.target
        thrd = OrderedDict(thrd_dict)
        mrcr = df_mrcr(df, y)
        perf = df_perf(df, y)
        perf_dict = mrcr
        perf_dict = update_dict(perf_dict, perf)
        orient_dict = perf_dict.copy()
        perf_df = pd.DataFrame.from_dict(perf_dict, orient='index')
        handle_dict = {}
        for th in thrd:
            pre_n = perf_df.shape[0]
            df_slc = perf_df.loc[perf_df.loc[:, th] >= thrd[th], :]
            left_n = df_slc.shape[0]
            handle_dict.update({th: {'pre': pre_n, 'handle': pre_n - left_n}})

        perf_dict = df_slc.to_dict(orient='index')
        self.perf_dict = perf_dict
        self.handle_dict = handle_dict
        self.orient_dict = orient_dict
        return self
