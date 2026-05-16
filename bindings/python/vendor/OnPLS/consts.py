# -*- coding: utf-8 -*-
"""
The :mod:`OnPLS.consts` module contains constants used in the package.

Created on Fri Jul 22 21:55:44 2016

Copyright (c) 2016, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import numpy as np

__all__ = ["TOLERANCE", "MAX_ITER", "FLOAT_EPSILON", "FLOAT_INF",
           "LIMIT_R2"]

# Settings
TOLERANCE = 5e-8

# TODO: MAX_ITER is heavily algorithm-dependent, so we have to think about if
# we should include a package-wide maximum at all.
MAX_ITER = 1000

FLOAT_EPSILON = np.finfo(float).eps

FLOAT_INF = np.inf

LIMIT_R2 = 0.01
