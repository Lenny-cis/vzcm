# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:10:10 2020

@author: linjianing
"""


import pandas as pd
import numpy as np
from entity import EntitySet


a = pd.DataFrame({'A':[1, 2, 3], 'B': ['A', 'B', 'C']})
es = EntitySet('test')
es = es.entity_from_dataframe('a', a)
ta = es.get_entity('a').df
ta.loc[:, 'B'] = 1
ta
es.get_entity('a').df
