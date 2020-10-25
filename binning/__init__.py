# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:52:14 2020

@author: linjianing
"""


from .globalchi import Binning


class EntitySetBinning:
    """实体集合分箱."""

    def __init__(self, entityset):
        self.entityset = entityset

    def fit(self, entity_id):
        """分箱."""
        entity = self.entityset.get_entity(entity_id)
        df = entity.df
        y = df.loc[:, entity.target]
        df_X = df.loc[:, df.columns.difference(entity.target)]
        ops = entity.variable_options
        binn = Binning(ops)
        binn.fit(df_X, y)
        self.bins = binn
        return self

    def transform(self, entity_id):
        """分箱应用."""
        entity = self.entityset.get_entity(entity_id)
        df = entity.df
        df_X = df.loc[:, df.columns.difference(entity.target)]
        X_tran = self.bins.transform(df_X)
        return X_tran
