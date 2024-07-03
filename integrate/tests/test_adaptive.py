import numpy as np

from integrate.adaptive import quad_vec
from integrate.base_quadratures import gauss_kronrod_21


def test_quad_vec_simple():
    n = np.arange(10)

    def f(x):
        return x ** n

    for tol in [0.1, 1e-3, 1e-6]:
        exact = 2**(n+1)/(n + 1)

        res, _ = quad_vec(f, 0, 2, tol, gauss_kronrod_21, np.linalg.norm)
        assert np.linalg.norm(res - exact) < tol
