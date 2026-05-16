# -*- coding: utf-8 -*-
"""
The :mod:`OnPLS.utils` module contains useful functions and functionality used
throughout the package.

Created on Fri Jul 29 22:10:29 2016

Copyright (c) 2016, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import itertools

import numpy as np

try:
    from . import consts  # When imported as a package.
except ValueError:
    import OnPLS.consts as consts  # When run as a program.

__all__ = ["leastNonZero", "normaliseColumns", "sumOfSquares",
           "sumOfCovariances", "cov", "project", "list_product"]


def leastNonZero(A):

    A = np.array(A)
    return np.min(A[A > 0.0])


def normaliseColumns(A):

    try:
        norms = np.linalg.norm(A, axis=0)
    except TypeError:
        norms = np.sqrt(np.sum(A**2.0, axis=0))
    # Avoid division by zero: Define 0 / 0 = 0.
    norms[np.abs(norms) < consts.TOLERANCE] = 1.0
    A = A / norms

    return A


def sumOfSquares(A):

    return np.sum(A**2.0)


def sumOfCovariances(T, predComp, comp=None):

    n = len(T)
    sumCov = [0] * n

    if comp is None:
        comps = range(T[0].shape[1])
    else:
        comps = comp

    for c in comps:
        for i in range(n):
            Ti = T[i]
            for j in range(n):
                Tj = T[j]
                if predComp[i][j] > 0:
                    sumCov[i] = sumCov[i] + cov(Ti[:, c], Tj[:, c])

    return sumCov


def cov(x, y):
    n = np.max(x.shape)
    x_ = x - np.mean(x)
    y_ = y - np.mean(y)

    coef = np.dot(x_.T, y_) / n

    return coef


def project(v, u):
    """ Project v onto u.
    """
    return (np.dot(v.T, u) / np.dot(u.T, u)) * u


def list_product(start, stop, step=1):
    """Return a list containing all combinations of lists with elements
    starting from start and stopping one before stop. The cartesian product of
    the values of each elements in the list ranging from start to stop.

    Example
    -------
    >>> import OnPLS
    >>> start = [0, 0]
    >>> stop = [2, 3]
    >>> OnPLS.utils.list_product(start, stop)
    [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    """
    start = np.asarray(start)
    orig_shape = start.shape
    start = start.ravel().tolist()
    stop = np.asarray(stop).ravel().tolist()

    dimlists = [0] * len(start)
    for i in range(len(start)):
        dimlists[i] = list(range(start[i], stop[i], step))

    prod = [np.asarray(list(i)).reshape(orig_shape).tolist()
            for i in list(itertools.product(*dimlists))]

    return prod


if __name__ == "__main__":
    import doctest
    doctest.testmod()
