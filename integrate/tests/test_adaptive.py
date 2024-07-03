import pytest

import numpy as np
from numpy.testing import assert_allclose

from integrate.adaptive import ndquad_vec, quad_vec
from integrate.base_quadratures import gauss_kronrod_15, gauss_kronrod_21, trapezoid


quadrature_params = pytest.mark.parametrize(
    'quadrature',
    [gauss_kronrod_15, gauss_kronrod_21, trapezoid]
)


def _max_norm(x):
    return np.amax(abs(x))


@quadrature_params
def test_quad_vec_simple(quadrature):
    n = np.arange(10)

    def f(x):
        return x ** n

    for tol in [0.1, 1e-3, 1e-6]:
        exact = 2**(n+1)/(n + 1)

        res, _ = quad_vec(f, 0, 2, tol, quadrature, _max_norm)
        assert_allclose(res, exact, rtol=0, atol=tol, verbose=True)

        res, _ = quad_vec(f, 0, 2, tol, quadrature, np.linalg.norm)
        assert np.linalg.norm(res - exact) < tol


@quadrature_params
def test_nquad_vec(quadrature):
    n = np.arange(10)

    def f(x, y, z):
        return (x + y + z)**n

    for tol in [0.1, 1e-3, 1e-6]:
        # According to Wolfram Alpha
        exact = (-3 * 2**(n+3) + 3**(n+3) + 3)/(n**3 + 6*n**2 + 11*n + 6)

        res, _ = ndquad_vec(
            f,
            [
                lambda: (0, 1),
                lambda z: (0, 1),
                lambda y, z: (0, 1)
            ],
            1e-10, gauss_kronrod_21, _max_norm
        )

        assert_allclose(res, exact, rtol=0, atol=tol, verbose=True)
