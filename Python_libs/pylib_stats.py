#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:18:19 2022

@author: glavrent
"""

import numpy as np
from scipy import linalg

def lse(A, b, B, d):
    """
    Equality-contrained least squares.
    The following algorithm minimizes ||Ax - b|| subject to the
    constrain Bx = d.
    (https://gist.github.com/fabianp/915461)
    
    Parameters
    ----------
    A : array-like, shape=[m, n]
    B : array-like, shape=[p, n]
    b : array-like, shape=[m]
    d : array-like, shape=[p]
    
    Reference
    ---------
    Matrix Computations, Golub & van Loan, algorithm 12.1.2
    
    Examples
    --------
    >>> A = np.array([[0, 1], [2, 3], [3, 4.5]])
    >>> b = np.array([1, 1])
    >>> # equality constrain: ||x|| = 1.
    >>> B = np.ones((1, 3))
    >>> d = np.ones(1)
    >>> lse(A.T, b, B, d)
    array([-0.5,  3.5, -2. ])
    """
    
    if not hasattr(linalg, 'solve_triangular'):
        # compatibility for old scipy
        solve_triangular = linalg.solve
    else:
        solve_triangular = linalg.solve_triangular
    A, b, B, d = map(np.asanyarray, (A, b, B, d))
    p = B.shape[0]
    Q, R = linalg.qr(B.T)
    y = solve_triangular(R[:p, :p].T, d)
    A = np.dot(A, Q)
    z = linalg.lstsq(A[:, p:], b - np.dot(A[:, :p], y))[0].ravel()
    return np.dot(Q[:, :p], y) + np.dot(Q[:, p:], z)