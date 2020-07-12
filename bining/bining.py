# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""

import pandas as pd
import numpy as np
import itertools as its
import scipy.stats as sps
import sys

sys.path.append('../')

import utils as utl


class Bining:

    def __init__(self, dic={}, cut_cnt=50, thrd_PCT=0.03, max_bin_cnt=6):
        self.cut_cnt = cut_cnt
        self.thrd_PCT = thrd_PCT
        self.max_bin_cnt = max_bin_cnt
        self.dic = dic

    def fit(self, X, y):
        for x_name in X.columns:
            x_bining = varBining(self.dic.get(x_name, {}), self.cut_cnt,
                                 self.thrd_PCT, self.max_bin_cnt)
            x_bining.fit(X.loc[:, x_name], y)
            setattr(self, x_name, x_bining)
            del x_bining
        return self

    def transform(self, X):
        X_trns = X.copy(deep=True)
        for x_name in X.columns:
            x_bining = getattr(self, x_name)
            X_trns.loc[:, x_name] = x_bining.transform(X.loc[:, x_name])
        return X_trns


class varBining(Bining):
    def __init__(self, dic, cut_cnt, thrd_PCT, max_bin_cnt):
        super().__init__(dic, cut_cnt, thrd_PCT, max_bin_cnt)
        self.I_min = dic.get('I_min', 3)
        self.U_min = dic.get('U_min', 4)
        self.cut_mthd = dic.get('cut_mthd', 'eqqt')
        self.prior_shape = list(dic.get('prior_shape', 'IDUA'))
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

    def _category_cut_adj(self, cut, idxs):
        dic = dict(enumerate(cut))
        for idx in idxs[::-1]:
            dic[idx-1] = list(dic[idx-1])
            dic[idx-1].extend(dic[idx])
            del dic[idx]
        return list(dic.values())

    def _numeric_cross(self, df_set, x):
        cross, cut = utl.gen_cross_numeric(
            df_set, x, self.dep, self.cut_cnt, self.cut_mthd)
        return cross, cut

    def _category_cross(self, df_set, x):
        cross, cut = utl.gen_cross_category(df_set, x, self.dep)
        return cross, cut

    def _merge_PCT_zero(self, crs, mthd='PCT'):
        """
        合并个数为0和占比过低的箱，不改变缺失组的结果.

        input
            df          bin和[0, 1]的列联表
            cut         原始切点
            thrd_PCT    占比阈值
            mthd        合并方法
                PCT     合并占比过低的箱
                zero    合并个数为0的箱
        """
        cross = crs[crs.index != -1].copy(deep=True)
        s = 1
        idx_lst = []
        while s:
            row_margin = cross.sum(axis=1)
            total = row_margin.sum()
            min_num = row_margin.min()
            # 找到占比最低的组或个数为0的组
            if mthd.upper() == 'PCT':
                min_idx = row_margin.idxmin()

            else:
                zero_idxs = cross[(cross == 0).any(axis=1)].index
                if len(zero_idxs) >= 1:
                    min_idx = zero_idxs[0]
                    min_num = 0

                else:
                    min_num = np.inf

            # 占比低于阈值则合并
            if min_num/total <= self.thrd_PCT:
                idxs = list(cross.index)
                # 最低占比的组的索引作为需要合并的组
                # sup_idx确定合并索引的上界，上界不超过箱数
                # inf_idx确定合并索引的下界，下界不低于0
                min_idx_row = idxs.index(min_idx)
                sup_idx = idxs[min(len(cross)-1, min_idx_row+1)]
                inf_idx = idxs[max(0, min_idx_row-1)]

                # 需合并组为第一组，向下合并
                if min_idx == idxs[0]:
                    min_idx = idxs[1]
                    cross = utl.merge_bin_by_idx(cross, [min_idx])
                    idx_lst.append(min_idx)

                # 需合并组为最后一组，向上合并
                elif min_idx == idxs[-1]:
                    cross = utl.merge_bin_by_idx(cross, [min_idx])
                    idx_lst.append(min_idx)

                elif sup_idx == inf_idx:
                    cross = utl.merge_bin_by_idx(cross, [inf_idx])
                    idx_lst.append(inf_idx)

                # 介于第一组和最后一组之间，找向上或向下最近的组合并
                else:
                    min_dist_idx = utl.slc_min_dist(
                        cross.loc[inf_idx: sup_idx])
                    cross = utl.merge_bin_by_idx(cross, [min_dist_idx])
                    idx_lst.append(min_dist_idx)

            else:
                s = 0

        return cross.append(crs[crs.index == -1]), idx_lst

    def _gen_comb_bins(self, crs):
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
        for loop in range(min_cut_loops_cnt, max_cut_loops_cnt):
            bin_comb = its.combinations(cut_point_list, loop)
            for bin_idxs in bin_comb:
                # print(bin_idxs)
                # 根据选取的切点合并列联表
                merged = utl.merge_bin_by_idx(cross, bin_idxs)
                shape = utl.bad_rate_shape(merged, self.I_min, self.U_min)
                # badrate的形状符合先验形状的分箱方式保留下来
                if not pd.isna(shape) and shape in self.prior_shape:
                    # print(shape)
                    var_bin_dic[s] = {}
                    detail, IV = utl.cal_WOE_IV(merged)
                    chi, p, dof, expFreq =\
                        sps.chi2_contingency(
                                merged.loc[~merged.index.isin([-1]), :].values,
                                correction=False)
                    var_entropy = sps.entropy(
                            detail.loc[detail.index != -1, 'All'])
                    var_bin_dic[s]['detail'] = detail
                    var_bin_dic[s]['IV'] = IV
                    var_bin_dic[s]['p_value'] = p
                    var_bin_dic[s]['entropy'] = var_entropy
                    var_bin_dic[s]['shape'] = shape
                    var_bin_dic[s]['bin_cnt'] = len(merged)-1
                    var_bin_dic[s]['bin_idxs'] = list(bin_idxs)
                    s += 1
        return var_bin_dic

    def _select_best(self, dic):
        self._validate_input()
        if self.slc_mthd == 'p':
            comp = pd.DataFrame(dic).T.sort_values(['shape', 'p'],
                                                   ascending=True)

        elif self.slc_mthd in ['IV', 'entropy']:
            comp = pd.DataFrame(dic).T.sort_values(['shape', self.slc_mthd],
                                                   ascending=False)
        comp.drop_duplicates(['shape'], keep='first', inplace=True)
        best_idx = list(comp.index)
        return {idx: dic[idx] for idx in best_idx}

    def fit(self, x, y):
        df_set = pd.DataFrame({x.name: x, y.name: y})
        self.dep = y.name
        if x.values.dtype.kind in ['f', 'i', 'u']:
            cross, cut = self._numeric_cross(df_set, x.name)
            cross, pct_idxs = self._merge_PCT_zero(cross, mthd='PCT')
            cross, zero_idxs = self._merge_PCT_zero(cross, mthd='zero')
            numeric_dic = self._gen_comb_bins(cross)
            best_dic = self._select_best(numeric_dic)
            idxs = list(set(pct_idxs+zero_idxs+best_dic['bin_idxs']))
            best_dic['cut'] = [x for x in cut if x not in best_dic['bin_idxs']]
            best_dic['WOE'] = best_dic['detail']['WOE'].to_dict()

        else:
            cross, cut = self._category_cross(df_set, x.name)
            cross, pct_idxs = self._merge_PCT_zero(cross, mthd='PCT')
            cross, zero_idxs = self._merge_PCT_zero(cross, mthd='zero')
            category_dic = self._gen_comb_bins(cross)
            best_dic = self._select_best(category_dic)
            idxs = list(set(pct_idxs + zero_idxs + best_dic['bin_idxs']))
            best_dic['cut'] = self._category_cut_adj(cut, idxs)
            best_dic['WOE'] = best_dic['detail']['WOE'].to_dict()

        self.best_dic = best_dic
        return self

    def transform(self, x):
        if x.values.dtype.kind in ['f', 'i', 'u']:
            x_trns = pd.cut(x, self.best_dic['cut'], labels=False)
            x_trns.fillna(-1, inplace=True)
            x_trns = x_trns.map(self.best_dic['WOE'])
        return x_trns
