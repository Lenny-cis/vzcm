# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""

import pandas as pd
import numpy as np
import itertools as its
import scipy.stats as sps
import copy
from binning.utils import (gen_cross_numeric, gen_cross_category, merge_bin_by_idx,
                           bad_rate_shape, cal_WOE_IV, make_tqdm_iterator, merge_lowpct_zero)

PBAR_FORMAT = "Var: {desc} | Possible: {total} | Elapsed: {elapsed} | Progress: {l_bar}{bar}"


class Binning:
    """特征全集分箱."""

    def __init__(self, dic={}, cut_cnt=20, thrd_PCT=0.05, max_bin_cnt=6):
        self.cut_cnt = cut_cnt
        self.thrd_PCT = thrd_PCT
        self.max_bin_cnt = max_bin_cnt
        self.dic = dic

    def fit(self, X, y):
        """训练."""
        for x_name in X.columns:
            x_binning = varBinning(self.dic.get(x_name, {}),
                                   cut_cnt=self.cut_cnt,
                                   thrd_PCT=self.thrd_PCT,
                                   max_bin_cnt=self.max_bin_cnt)
            x_binning.fit(X.loc[:, x_name], y)
            setattr(self, x_name, x_binning)
            del x_binning
        return self

    def transform(self, X):
        """应用."""
        X_trns = X.copy(deep=True)
        for x_name in X.columns:
            x_binning = getattr(self, x_name)
            x_trns = x_binning.transform(X.loc[:, x_name])
            X_trns = pd.concat([X_trns, x_trns], axis=1)
        return X_trns


class varBinning(Binning):
    """单个变量分箱."""

    def __init__(self, dic, cut_cnt=20, thrd_PCT=0.05, max_bin_cnt=6):
        super().__init__(dic, cut_cnt, thrd_PCT, max_bin_cnt)
        self.I_min = dic.get('I_min', 3)
        self.U_min = dic.get('U_min', 4)
        self.cut_mthd = dic.get('cut_mthd', 'eqqt')
        self.prior_shape = list(dic.get('prior_shape', None))
        self.slc_mthd = dic.get('slc_mthd', 'entropy')
        self.dic['I_min'] = self.I_min
        self.dic['U_min'] = self.U_min
        self.dic['cut_mthd'] = self.cut_mthd
        self.dic['prior_shape'] = self.prior_shape
        self.dic['slc_mthd'] = self.slc_mthd

    def _validate_input(self):
        allowed_mthd = ['p', 'IV', 'entropy']
        if self.slc_mthd not in allowed_mthd:
            raise ValueError("仅支持slc_mthd {0}"
                             "但使用了slc_mthd={1}".format(
                                 allowed_mthd, self.slc_mthd))

        allowed_mthd = ['eqqt', 'eqdist']
        if self.cut_mthd not in allowed_mthd:
            raise ValueError("仅支持cut_mthd {0}"
                             "但使用了cut_mthd={1}".format(
                                 allowed_mthd, self.cut_mthd))

        allowed_shape = ['I', 'D', 'U', 'A']
        if not np.isin(self.prior_shape, allowed_shape).all():
            raise ValueError("仅支持prior_shape {0}"
                             "但使用了prior_shape={1}".format(
                                 allowed_shape, self.prior_shape))

    @staticmethod
    def _cut_adj(cut, bin_idxs):
        if isinstance(cut, list):
            return [x for i, x in enumerate(cut) if i not in bin_idxs]
        elif isinstance(cut, dict):
            t_cut = copy.deepcopy(cut)
            for idx in bin_idxs[::-1]:
                t_d = {k: v-1 for k, v in t_cut.items() if v >= idx}
                t_cut.update(t_d)
            return t_cut

    def _numeric_cross(self, df_set, x):
        cross, cut = gen_cross_numeric(
            df_set, x, self.dep, self.cut_cnt, self.cut_mthd)
        return cross, cut

    def _category_cross(self, df_set, x):
        cross, cut = gen_cross_category(df_set, x, self.dep)
        return cross, cut

    def _lowpct_zero_merge(self, crs, cut):
        cross = crs.copy(deep=True)
        cross, pct_idxs = merge_lowpct_zero(cross, thrd_PCT=self.thrd_PCT, mthd='PCT')
        pct_cut = varBinning._cut_adj(cut, pct_idxs)
        cross, zero_idxs = merge_lowpct_zero(cross, thrd_PCT=self.thrd_PCT, mthd='zero')
        t_cut = varBinning._cut_adj(pct_cut, zero_idxs)
        return cross, t_cut



    def _gen_comb_bins(self, crs, cut):
        cross = crs.copy(deep=True)
        min_I = self.I_min - 1
        min_U = self.U_min - 1
        max_cut_cnt = self.max_bin_cnt - 1
        cut_point_list = [x for x in cross.index[:] if x != -1][1:]
        cut_point_cnt = len(cut_point_list)
        # 限定分组数的上下限
        max_cut_loops_cnt = cut_point_cnt - min(min_I, min_U) + 1
        min_cut_loops_cnt = max(cut_point_cnt - max_cut_cnt, 0)
        var_bin_dic = {}
        s = 0
        loops_ = range(min_cut_loops_cnt, max_cut_loops_cnt)
        bcs = [bi for loop in loops_ for bi in its.combinations(cut_point_list, loop)]
        tqdm_options = {'bar_format': PBAR_FORMAT,
                        'total': len(bcs),
                        'desc': self.indep}
        with make_tqdm_iterator(**tqdm_options) as progress_bar:
            for bin_idxs in bcs:
                # 根据选取的切点合并列联表
                merged = merge_bin_by_idx(cross, bin_idxs)
                shape = bad_rate_shape(merged, self.I_min, self.U_min)
                # badrate的形状符合先验形状的分箱方式保留下来
                if not pd.isna(shape) and shape in self.prior_shape:
                    var_bin_dic[s] = {}
                    detail, IV = cal_WOE_IV(merged)
                    chi, p, dof, expFreq =\
                        sps.chi2_contingency(
                                merged.loc[~merged.index.isin([-1]), :].values,
                                correction=False)
                    var_entropy = sps.entropy(
                            detail.loc[detail.index != -1, 'All'])
                    var_bin_dic[s]['detail'] = detail
                    var_bin_dic[s]['IV'] = IV
                    var_bin_dic[s]['flogp'] = -np.log(p)
                    var_bin_dic[s]['entropy'] = var_entropy
                    var_bin_dic[s]['shape'] = shape
                    var_bin_dic[s]['bin_cnt'] = len(merged)-1
                    var_bin_dic[s]['bin_idxs'] = bin_idxs
                    var_bin_dic[s]['cut'] = varBinning._cut_adj(cut, bin_idxs)
                    var_bin_dic[s]['WOE'] = detail['WOE'].to_dict()
                    s += 1
                progress_bar.update()
        return var_bin_dic

    def _select_best(self, dic):
        self._validate_input()
        if len(dic) == 0:
            return {}
        dd = pd.DataFrame.from_dict(dic, orient='index')
        if self.slc_mthd == 'p':
            sort_keys = ['flogp', 'entropy', 'IV']
        elif self.slc_mthd == 'IV':
            sort_keys = ['IV', 'entropy', 'flogp']
        else:
            sort_keys = ['entropy', 'IV', 'flogp']

        best_dd = dd.sort_values(by=sort_keys, ascending=False).groupby(['shape', 'bin_cnt']).head(1)
        return best_dd.to_dict(orient='index')

    def fit(self, x, y):
        """单变量训练."""
        df_set = pd.DataFrame({x.name: x, y.name: y})
        self.dep = y.name
        self.indep = x.name
        if x.values.dtype.kind in ['f', 'i', 'u']:
            cross, cut = self._numeric_cross(df_set, x.name)
            cross, cut = self._lowpct_zero_merge(cross, cut)
            numeric_dic = self._gen_comb_bins(cross, cut)
            best_dic = self._select_best(numeric_dic)

        else:
            if not x.dtypes.ordered:
                self.prior_shape = 'D'
            cross, cut = self._category_cross(df_set, x.name)
            cross, cut = self._lowpct_zero_merge(cross, cut)
            category_dic = self._gen_comb_bins(cross, cut)
            best_dic = self._select_best(category_dic)

        self.best_dic = best_dic
        return self

    def transform(self, x):
        """单变量应用."""
        trns = pd.DataFrame()
        for i in self.best_dic:
            shape = self.best_dic[i]['shape']
            bin_cnt = self.best_dic[i]['bin_cnt']
            name = '_'.join([str(x.name), str(shape), str(bin_cnt)])
            if x.values.dtype.kind in ['f', 'i', 'u']:
                x_trns = pd.cut(x, self.best_dic[i]['cut'], labels=False)
            else:
                t_x = x.copy(deep=True)
                cl = list(t_x.dtypes.categories)
                cl.append(-1)
                co = t_x.dtypes.ordered
                t_x = t_x.astype(pd.CategoricalDtype(cl, ordered=co))
                x_trns = t_x.map(self.best_dic[i]['cut'])
            x_trns.fillna(-1, inplace=True)
            x_trns = x_trns.map(self.best_dic[i]['WOE'])
            x_trns.name = name
            trns = pd.concat([trns, x_trns], axis=1)
        return trns
