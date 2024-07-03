import numpy as np

from integrate.base_quadratures import gauss_kronrod_21
from integrate.adaptive import ndquad_vec

n = np.arange(12)


def f(x, y, z):
    return np.power(float(x + y + z), n)


est, err = ndquad_vec(
    f,
    [
        lambda: (0, 1),
        lambda z: (0, 1),
        lambda y, z: (0, 1)
    ],
    1e-10, gauss_kronrod_21, np.linalg.norm
)

exact = (-3 * 2**(n+3) + 3**(n+3) + 3)/(n**3 + 6*n**2 + 11*n + 6)

print("Estimate:", est)
print("Error:", err)
print("Exact:", exact)
print("Difference:", est - exact)

# Outputs:
# Estimate: [1.00000000e+00 1.50000000e+00 2.50000000e+00 4.50000000e+00
#  8.60000000e+00 1.72500000e+01 3.60119048e+01 7.77500000e+01
#  1.72733333e+02 3.93300000e+02 9.14772727e+02 2.16750000e+03]
# Error: 4.348737873398455e-11
# Exact: [1.00000000e+00 1.50000000e+00 2.50000000e+00 4.50000000e+00
#  8.60000000e+00 1.72500000e+01 3.60119048e+01 7.77500000e+01
#  1.72733333e+02 3.93300000e+02 9.14772727e+02 2.16750000e+03]
# Difference: [ 2.22044605e-16 -4.44089210e-16  4.44089210e-16 -8.88178420e-16
#  -1.77635684e-15  0.00000000e+00  0.00000000e+00  2.84217094e-14
#   2.84217094e-14  0.00000000e+00  1.13686838e-13 -4.54747351e-13]
