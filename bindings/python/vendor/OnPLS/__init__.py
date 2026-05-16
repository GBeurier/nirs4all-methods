# -*- coding: utf-8 -*-
"""
The :mod:`OnPLS` module includes several different machine learning methods
for linear projection-based latent variable methods.

Created on Sun Aug 21 12:42:11 2016

Copyright (c) 2016, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
from OnPLS import consts
from OnPLS import estimators
from OnPLS import resampling
from OnPLS import utils

__version__ = "0.0.1"

__all__ = ["consts", "estimators", "resampling", "utils"]
