# -*- coding: utf-8 -*-
"""
The :mod:`OnPLS.resampling` module contains resampling methods that can be used
to determine statistics from an estimator.

Created on Wed Nov 23 21:02:03 2016

Copyright (c) 2016-2024, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import copy
import itertools

import numpy as np

import OnPLS.estimators as estimators

__all__ = ["cross_validation", "grid_search"]


def cross_validation(estimator, X, cv_rounds=7, random_state=None):
    """Performs k-fold cross-validation for a given estimator and data set.

    Parameters
    ----------
    estimator : OnPLS.estimators.BaseEstimator
        The estimator to use in the cross-validation.

    X : numpy.ndarray or list of numpy.ndarray
        The data to perform cross-validation over.

    cv_rounds : int
        The number of cross-validation folds. Will be set to N, the number of
        rows of the blocks in X, if ``cv_rounds > N``.

    random_state : numpy.random.RandomState, optional
        A random number generator state to use for random numbers. Used e.g.
        when generating start vectors. Default is None, do not use a random
        state.

    Example
    -------
    >>> import numpy as np
    >>> import OnPLS
    >>>
    >>> np.random.seed(42)
    >>>
    >>> n, p_1, p_2, p_3 = 4, 3, 4, 5
    >>> t = np.sort(np.random.randn(n, 1), axis=0)
    >>> p1 = np.sort(np.random.randn(p_1, 1), axis=0)
    >>> p2 = np.sort(np.random.randn(p_2, 1), axis=0)
    >>> p3 = np.sort(np.random.randn(p_3, 1), axis=0)
    >>> X1 = np.dot(t, p1.T) + 0.1 * np.random.randn(n, p_1)
    >>> X2 = np.dot(t, p2.T) + 0.1 * np.random.randn(n, p_2)
    >>> X3 = np.dot(t, p3.T) + 0.1 * np.random.randn(n, p_3)
    >>>
    >>> predComp = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    >>> orthComp = [1, 1, 1]
    >>> onpls = OnPLS.estimators.OnPLS(predComp, orthComp, verbose=0)
    >>>
    >>> OnPLS.resampling.cross_validation(onpls, [X1, X2, X3],
    ...     cv_rounds=4)  # doctest: +ELLIPSIS
    [0.5493..., 0.9941..., 0.9904..., 0.9859...]
    """
    if isinstance(estimator, estimators.BaseUniblock):
        if isinstance(X, np.ndarray):
            X = [X]

    if random_state is None:
        random_state = np.random.RandomState()

    n = len(X)

    N = X[0].shape[0]

    cv_rounds = min(max(1, N), int(cv_rounds))

    scores = []
    for k in range(cv_rounds):

        test_samples = list(range(k, N, cv_rounds))
        train_samples = list(set(range(0, N)).difference(test_samples))

        Xtest = [0] * n
        Xtrain = [0] * n
        for i in range(n):
            Xi = X[i]
            Xtest[i] = Xi[test_samples, :]
            Xtrain[i] = Xi[train_samples, :]

        estimator.fit(Xtrain)

        score = estimator.score(Xtest)
        scores.append(score)

    return scores


def grid_search(estimator, X, params_grid, random_state=None):
    """Exhaustive search over a parameter grid for parameters of an estimator.

    Parameters
    ----------
    estimator : OnPLS.estimators.BaseEstimator
        The estimator to use in the cross-validation.

    X : numpy.ndarray or list of numpy.ndarray
        The data to perform grid search over.

    params_grid : dict or list of dict
        Dictionary with parameter names (key) and lists (values) of parameter
        settings to try. Or a list of such dictionaries. The values will be
        set, one at the time, to the attribute of the estimator with the given
        name.

    Attributes
    ----------
    best_estimator_ : OnPLS.estimators.BaseEstimator
        The estimator that was deemed the best one (highest score) by the
        search.

    best_score_ : float
        The cross-validation score of the best estimator.

    best_params_ : dict
        A dictionary with the parameter settings that gave the highest score.

    Example
    -------
    >>> import numpy as np
    >>> import OnPLS
    >>>
    >>> np.random.seed(42)
    >>>
    >>> n, p_1, p_2, p_3 = 4, 3, 4, 5
    >>> t = np.sort(np.random.randn(n, 1), axis=0)
    >>> p1 = np.sort(np.random.randn(p_1, 1), axis=0)
    >>> p2 = np.sort(np.random.randn(p_2, 1), axis=0)
    >>> p3 = np.sort(np.random.randn(p_3, 1), axis=0)
    >>> X1 = np.dot(t, p1.T) + 0.1 * np.random.randn(n, p_1)
    >>> X2 = np.dot(t, p2.T) + 0.1 * np.random.randn(n, p_2)
    >>> X3 = np.dot(t, p3.T) + 0.1 * np.random.randn(n, p_3)
    >>> X = [X1, X2, X3]
    >>>
    >>> pred_comp = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    >>> orth_comp = [1, 1, 1]
    >>> onpls = OnPLS.estimators.OnPLS(pred_comp, orth_comp, verbose=0)
    >>>
    >>> params_grid = OnPLS.utils.list_product([0, 0, 0], [3, 3, 3])
    >>> OnPLS.resampling.grid_search(onpls, X,
    ...     {"orth_comp": params_grid})  # doctest: +ELLIPSIS
    (<OnPLS.estimators.OnPLS ...>, 0.8841..., {'orth_comp': [2, 2, 1]})
    >>>
    >>> np.random.seed(42)
    >>>
    >>> pred_comp = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    >>> orth_comp = [1, 1, 1]
    >>> onpls = OnPLS.estimators.OnPLS(pred_comp, orth_comp, verbose=0)
    >>>
    >>> params_grid_pred_ = OnPLS.utils.list_product([[0, 0, 0],
    ...                                               [0, 0, 0],
    ...                                               [0, 0, 0]],
    ...                                              [[1, 2, 2],
    ...                                               [2, 1, 2],
    ...                                               [2, 2, 1]])
    >>> to_remove = []
    >>> for i in range(len(params_grid_pred_)):
    ...     M = np.array(params_grid_pred_[i])
    ...     if np.sum(M[0, :]) == 0:
    ...         to_remove.append(i)
    ...     if np.sum(M[1, :]) == 0:
    ...         to_remove.append(i)
    ...     if np.sum(M[2, :]) == 0:
    ...         to_remove.append(i)
    ...     if np.sum(M[:, 0]) == 0:
    ...         to_remove.append(i)
    ...     if np.sum(M[:, 1]) == 0:
    ...         to_remove.append(i)
    ...     if np.sum(M[:, 2]) == 0:
    ...         to_remove.append(i)
    >>> params_grid_pred = []
    >>> for i in range(len(params_grid_pred_)):
    ...     if i not in to_remove:
    ...         params_grid_pred.append(params_grid_pred_[i])
    >>>
    >>> params_grid_orth = OnPLS.utils.list_product([0, 0, 0], [3, 3, 3])
    >>>
    >>> res = OnPLS.resampling.grid_search(onpls, X,
    ...     {"orth_comp": params_grid_orth,
    ...      "pred_comp": params_grid_pred})
    >>> res[0]  # doctest: +ELLIPSIS
    <OnPLS.estimators.OnPLS ...>
    >>> res[1]  # doctest: +ELLIPSIS
    0.9027...
    >>> res[2]["orth_comp"]
    [2, 2, 1]
    >>> res[2]["pred_comp"]
    [[0, 1, 0], [1, 0, 1], [1, 0, 0]]
    """
    if isinstance(params_grid, dict):
        params_grid = [params_grid]

    if random_state is None:
        random_state = np.random.RandomState()

    best_estimator_ = None
    best_score_ = -np.inf
    best_params_ = None

    for pg in params_grid:
        names = list(pg.keys())
        values = []
        for name in names:
            values.append(pg[name])

        # Store current values from the estimator.
        old_params = dict()
        for i in range(len(names)):
            old_params[name] = getattr(estimator, name)

        for value in itertools.product(*values):

            params = dict()
            for i in range(len(names)):
                name = names[i]
                val = value[i]

                # Set new values
                setattr(estimator, name, val)
                params[name] = val

            # Fit model using updated estimator.
            score = np.mean(cross_validation(estimator, X, cv_rounds=7,
                                             random_state=random_state))

            # Save if better than previous tries.
            if score > best_score_:
                best_estimator_ = copy.deepcopy(estimator)
                best_score_ = score
                best_params_ = params
                # print "Better found!", best_params_, best_score_

        # Set the values back to what they were in the estimator
        for i in range(len(names)):
            setattr(estimator, name, old_params[name])
        estimator.reset()  # The estimator is no longer valid.

    return best_estimator_, best_score_, best_params_


if __name__ == "__main__":
    import doctest
    doctest.testmod()
