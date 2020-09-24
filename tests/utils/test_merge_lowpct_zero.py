# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:23:10 2020

@author: linjianing
"""


import pandas as pd
import numpy as np
import copy
from binning.utils import merge_lowpct_zero

test_raw = pd.DataFrame({1: [40, 41, 80, 0, 200], 0: [360, 359, 320, 198, 2000]})

test_raw, cut_idx = merge_lowpct_zero(test_raw, mthd='zero')
merge_lowpct_zero(test_raw, mthd='PCT')

def _cut_adj(cut, bin_idxs):
    if isinstance(cut, list):
        return [x for i, x in enumerate(cut) if i not in bin_idxs]
    elif isinstance(cut, dict):
        t_cut = copy.deepcopy(cut)
        for idx in bin_idxs[::-1]:
            t_d = {k: v-1 for k, v in t_cut.items() if v >= idx}
            t_cut.update(t_d)
        return t_cut

_cut_adj({'A':0, 'B': 1, 'C': 2, 'D': 3}, [])
