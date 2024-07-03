import sys
import functools


def _generic_gauss_kronrod(f, a, b, x, w, v, norm_func):
    """Quadrature of vector-valued using a Gauss-Kronrod rule.

    Parameters
    ----------
    f : callable
        Vector-valued function f(x) to integrate
    a : float
        Initial point.
    b : float
        Final point.
    x : list of float
        Nodes to evaluate the function.
    w : list of float
        Weights for the full higher-order rule.
    v : list of float
        Weights for the lower-order (embedded) rule.
    norm_func: callable
        Vector norm to use for error estimation.

    Returns
    -------
    res : {float, array-like}
        Estimate for the final result.
    err : float
        Error estimate for the result using the given norm.
    """

    # Initially the quadrature rule is only for [-1, 1], we need to map the nodes and
    # weights so they work for the interval [a, b].
    #
    # The SciPy implementation does this for each call to f, I am calculating the nodes
    # at the start for convinience.
    x_scaled = [(x_i + 1) * (b - a) / 2 + a for x_i in x]
    w_scaled = [w_i * (b - a) / 2 for w_i in w]
    v_scaled = [v_i * (b - a) / 2 for v_i in v]

    # Cache calls to f, SciPy does this by storing all the evaluations in a list
    f = functools.cache(f)

    # Let "gk_" be estimates obtained via the full Gauss-Kronrod quadrature
    # Let "g_" be estimates obtained via Gauss quadrature
    gk_estimate = sum([v_scaled[i] * f(x_scaled[i]) for i in range(len(x))])
    g_estimate = sum([w_scaled[i] * f(x_scaled[2 * i + 1]) for i in range(len(w))])

    avg_of_f = gk_estimate / (b - a)

    avg_deviation_of_f = norm_func(sum(
        [w_scaled[i] * abs(f(x_scaled[i]) - avg_of_f) for i in range(len(w))]
    ))

    eps = sys.float_info.epsilon

    # This is the error estimate as used in SciPy's implementation, which itself comes
    # from `dqk21.f` in QUADPACK.

    gk_err_estimate = norm_func(gk_estimate - g_estimate)

    if avg_deviation_of_f != 0 and gk_err_estimate != 0:
        err_estimate = avg_deviation_of_f * min(
            1,
            (200 * gk_err_estimate / avg_deviation_of_f)**1.5,
        )
    else:
        err_estimate = gk_err_estimate

    # TODO: Understand why this is a good estimate of the rounding error, or what even
    # "rounding error" precisely means in the first place
    round_err = float(norm_func(50 * eps * avg_deviation_of_f))

    # TODO: Also don't understand why this is necessary, surely the round_error will
    # always be zero or greater than or equal to the minimum float?
    if round_err > sys.float_info.min:
        err_estimate = max(err_estimate, round_err)

    return gk_estimate, err_estimate


def gauss_kronrod_21(f, a, b, norm_func):
    # Gauss-Kronrod points

    # I'm using lists here rather than tuples so that it's easier to map over each
    # element, but SciPy uses tuples which is maybe more efficient?
    #
    # TODO: where do these actually come from? Above, to calculate the g_estimate, we
    # take every other node but still the same weights. Why?
    x = [
        0.995657163025808080735527280689003,
        0.973906528517171720077964012084452,
        0.930157491355708226001207180059508,
        0.865063366688984510732096688423493,
        0.780817726586416897063717578345042,
        0.679409568299024406234327365114874,
        0.562757134668604683339000099272694,
        0.433395394129247190799265943165784,
        0.294392862701460198131126603103866,
        0.148874338981631210884826001129720,
        0,
        -0.148874338981631210884826001129720,
        -0.294392862701460198131126603103866,
        -0.433395394129247190799265943165784,
        -0.562757134668604683339000099272694,
        -0.679409568299024406234327365114874,
        -0.780817726586416897063717578345042,
        -0.865063366688984510732096688423493,
        -0.930157491355708226001207180059508,
        -0.973906528517171720077964012084452,
        -0.995657163025808080735527280689003,
    ]

    # 10-point weights
    #
    # TODO: These are the roots of the degree 10 Legendre polynomial. Adding to the
    # above, why are these the weights? Shouldn't the nodes be the roots?
    w = [
        0.066671344308688137593568809893332,
        0.149451349150580593145776339657697,
        0.219086362515982043995534934228163,
        0.269266719309996355091226921569469,
        0.295524224714752870173892994651338,
        0.295524224714752870173892994651338,
        0.269266719309996355091226921569469,
        0.219086362515982043995534934228163,
        0.149451349150580593145776339657697,
        0.066671344308688137593568809893332,
    ]

    # 21-point weights
    v = [
        0.011694638867371874278064396062192,
        0.032558162307964727478818972459390,
        0.054755896574351996031381300244580,
        0.075039674810919952767043140916190,
        0.093125454583697605535065465083366,
        0.109387158802297641899210590325805,
        0.123491976262065851077958109831074,
        0.134709217311473325928054001771707,
        0.142775938577060080797094273138717,
        0.147739104901338491374841515972068,
        0.149445554002916905664936468389821,
        0.147739104901338491374841515972068,
        0.142775938577060080797094273138717,
        0.134709217311473325928054001771707,
        0.123491976262065851077958109831074,
        0.109387158802297641899210590325805,
        0.093125454583697605535065465083366,
        0.075039674810919952767043140916190,
        0.054755896574351996031381300244580,
        0.032558162307964727478818972459390,
        0.011694638867371874278064396062192,
    ]

    return _generic_gauss_kronrod(f, a, b, x, w, v, norm_func)
