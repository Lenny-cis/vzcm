# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:10:36 2020

@author: linjianing
"""


import numpy as np
import pandas as pd


class Variable:
    """变量基类."""

    def __init__(self, ser=None):
        self._ser = ser

    @property
    def series(self):
        """序列值."""
        return self._ser

    @series.setter
    def series(self, ser):
        self._ser = ser


class Numeric(Variable):
    """数值型."""

    def __init__(self, ser=None):
        super().__init__(ser)


class Discrete(Variable):
    """离散型."""

    def __init__(self, categories, ordered, ser=None):
        super().__init__(ser)
        self.cates = categories
        self._ordered = ordered

    @property
    def categories(self):
        """类值."""
        return self.cates

    @property
    def ordered(self):
        """有序."""
        return self._ordered

    @property
    def vtype(self):
        """类型值."""
        return pd.CategoricalDtype(categories=self.categories, ordered=self.ordered)


class Count(Numeric):
    """计数型."""

    def __init__(self, ser=None):
        super().__init__(ser)

    @property
    def vtype(self):
        """类型值."""
        return np.integer


class Summ(Numeric):
    """汇总型."""

    def __init__(self, ser=None):
        super().__init__(ser)

    @property
    def vtype(self):
        """类型值."""
        return np.floating


class Category(Discrete):
    """分类型."""

    def __init__(self, categories):
        super().__init__(categories, False)


class Ordinal(Discrete):
    """有序型."""

    def __init__(self, categories):
        super().__init__(categories, True)
