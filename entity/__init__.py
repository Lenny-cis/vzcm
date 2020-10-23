# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 19:08:02 2020

@author: linjianing
"""


import numpy as np
import pandas as pd


class EntitySet:
    """实体集合."""

    def __init__(self, id, entities=None):
        """创建实体集合.

        Example_:
            entities = {'train_sample': (train_df, {'target': 'flag, 'variable_options':{}})}.
        """
        self.id = id
        self.entity_dict = {}
        entities = entities or {}
        for entity in entities:
            df = entities[entity][0]
            kw = {}
            if len(entities[entity]) == 2:
                kw = entities[entity][1]
            self.entity_from_dataframe(entity_id=entity,
                                       dataframe=df,
                                       **kw)

    def entity_from_dataframe(self, entity_id, dataframe, target=None, variable_options=None):
        """从dataframe生成实体."""
        variable_options = variable_options or {}
        entity = Entity(entity_id,
                        dataframe,
                        target,
                        variable_options)
        self.entity_dict[entity.id] = entity
        return self

    @property
    def entities(self):
        """获取实体集合."""
        return list(self.entity_dict.values())

    def get_entity(self, entity_id):
        """获取实体."""
        return self.entity_dict[entity_id]


class Entity:
    """实体."""

    def __init__(self, id, df, target=None, variable_options=None):
        self.id = id
        self.df = df
        self.target = target
        self._create_variables(variable_options)

    def _create_variables(self, variable_options):
        variable_options = variable_options or {}
        ops = {'variable_type': pd.CategoricalDtype(), 'variable_shape': 'IDU'}
        _vars = self.df.columns.difference([self.target])
        _var_types = self.df.loc[:, _vars].dtypes
        _ops = {k: {'variable_type': ops.get('variable_type') if not issubclass(v.type, np.number) else v.type,
                    'variable_shape': 'D' if not issubclass(v.type, np.number) else ops.get('variable_shape')}
                for k, v in _var_types.items()}
        for k, v in _ops.items():
            v.update(variable_options.get(k, {}))
        self.variable_options = _ops
        variable_types = {k: v.get('variable_type') for k, v in self.variable_options.items()}
        self.df = convert_all_variable_data(self.df, variable_types)

    @property
    def df(self):
        """Dataframe providing the data for the entity."""
        return self._df.copy(deep=True)

    @df.setter
    def df(self, t_df):
        self._df = t_df

def convert_all_variable_data(df, vartypes):
    """变量转换."""
    df = df.copy(deep=True)
    for k, v in vartypes.items():
        df.loc[:, k] = df.loc[:, k].astype(v)
    return df
