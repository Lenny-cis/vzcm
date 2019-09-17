# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:48:32 2019

@author: 86182

model validation
"""

import pandas as pd


def twoScoresCrossMat(df, sc1, sc2, flag, bins=10):
    t_df = df.copy(deep=True)
    t_df[sc1+'_bs'] = pd.cut(t_df[sc1], bins)
    t_df[sc2+'_bs'] = pd.cut(t_df[sc2], bins)
    group = t_df.groupby([sc1+'_bs', sc2+'_bs'])[flag].agg(['size', 'sum'])
    group['br'] = group.apply(lambda x: x['sum']/x['size'], axis=1)
    t_g = group['br'].unstack()
    return t_g
