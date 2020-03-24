# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import os
import pickle as pkl


class VZ_model:
    def __init__(self):
        pass

    class sub:
        def __init__(self, obj):
            self.obj = obj

    def read_file(self):
        path = self.inpath
        file = self.infile
        os.chdir(path)
        pj = pd.read_excel(file, sheet_name='模型分数划分', header=4)
        scorecard = pd.read_excel(file, sheet_name='评分卡')

        scorecard.iloc[:, 0] = scorecard.iloc[:, 0].str.upper()
        scorecard.iloc[:, 2].replace('low', -np.inf, inplace=True)
        scorecard.iloc[:, 2].replace('missing', np.nan, inplace=True)
        scorecard.iloc[:, 3].replace('high', np.inf, inplace=True)
        scorecard.iloc[:, 3].replace('missing', np.nan, inplace=True)

        return scorecard, pj

    def vars_scores(self, scorecard):
        sub = self.sub

        names = list(set(scorecard.iloc[:, 0].str.upper()))

        for name in names:
            var_df = scorecard.loc[scorecard.iloc[:, 0] == name, :]
            non_na = var_df.loc[var_df.iloc[:, 2].notna()]
            non_na.reset_index(drop=True, inplace=True)
            nan_df = var_df.loc[var_df.iloc[:, 2].isna()]

            score_dic = non_na.iloc[:, 4].round(4).to_dict()
            score_dic[-1] = nan_df.iloc[0, 4].round(4)
            cut = list(non_na.iloc[:, 2].round(9))
            cut.append(np.inf)

            setattr(self, name, sub(obj=self))
            setattr(getattr(self, name), 'score', score_dic)
            setattr(getattr(self, name), 'cut', cut)

        self.names = names

    def model_pj(self, pj):
        sub = self.sub
        col_name = list(pj.columns)[1]
        t_pj = pj.sort_values(by=[col_name])

        cut = list(t_pj.iloc[:, 2])
        cut.insert(0, -np.inf)
        pj_dic = t_pj.iloc[:, 0].to_dict()

        setattr(self, 'pj', sub(obj=self))
        setattr(getattr(self, 'pj'), 'cut', cut)
        setattr(getattr(self, 'pj'), 'score', pj_dic)

    def internal(self, obj):
        setattr(self, 'pj', obj.pj)
        setattr(self, 'names', obj.names)
        names = obj.names
        for name in names:
            setattr(self, name, getattr(obj, name))

    def read_in(self, inpath, infile):
        self.inpath = inpath
        self.infile = infile
        read_file = self.read_file
        vars_score = self.vars_scores
        model_pj = self.model_pj
        internal = self.internal

        if self.infile[-3:] == 'pkl':
            with open(infile, 'rb') as f:
                t_pkl = pkl.load(f)
                internal(t_pkl)
            f.close()
        else:
            scorecard, pj = read_file()
            vars_score(scorecard)
            model_pj(pj)

    def save_out(self, outpath, outfile):
        os.chdir(outpath)
        with open(outfile, 'wb') as f:
            pkl.dump(self, f)
        f.close()
