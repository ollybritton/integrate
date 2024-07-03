import numpy as np

from integrate.base_quadratures import gauss_kronrod_21
from integrate.adaptive import ndquad_vec


n = np.arange(10)


def f(x, y, z):
    return (1 + x**2 + y**2 + z**2)**(-n)


est, err = ndquad_vec(
    f,
    [
        lambda: (0, 1),
        lambda z: (0, 1),
        lambda y, z: (0, 1)
    ],
    1e-5, gauss_kronrod_21, np.linalg.norm
)

print(est, err)
