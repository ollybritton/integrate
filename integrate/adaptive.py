import heapq


def quad_vec(f, a, b, tol, quadrature, norm_func):
    """
    Adaptive quadrature of vector-valued function.

    Parameters
    ----------
    f : callable
        Vector-valued function f(x) to integrate
    a : float
        Initial point.
    b : float
        Final point.
    tol : float
        Absolute tolerance.
    quadrature : callable
        Underlying quadrature rule to use. Should be a function like
        quadrature(f, a, b, norm_func) which returns an estimate of the integral and
        and an estimate of the error.
    norm_func: callable
        Vector norm to use for error estimation.

    Returns
    -------
    res : {float, array-like}
        Estimate for the final result.
    err : float
        Error estimate for the result using the given norm.
    """
    global_est, global_err = quadrature(f, a, b, norm_func)

    # Since tuples are compared based on their first element, making the first entry
    # -err means that the top of the heap is always the interval with the largest error
    intervals = [(-global_err, a, b, global_est)]

    while global_err > tol:
        interval = heapq.heappop(intervals)
        neg_err_k, a_k, b_k, est_k = interval
        err_k = -neg_err_k

        m = (a_k + b_k) / 2

        est_left, err_left = quadrature(f, a_k, m, norm_func)
        est_right, err_right = quadrature(f, m, b_k, norm_func)

        global_est = global_est - est_k + est_left + est_right
        global_err = global_err - err_k + err_left + err_right

        heapq.heappush(intervals, (-err_left, a_k, m, est_left))
        heapq.heappush(intervals, (-err_right, m, b_k, est_right))

    return global_est, global_err


def ndquad_vec(f, ranges, tol, quadrature, norm_func):
    """
    Adaptive quadrature of an n-dimensional vector-valued function.

    Wraps `quad_vec` to enable integration over multiple variables, it is not doing true
    cubature.

    Parameters
    ----------

    f : callable
        Vector-valued function f(x) to integrate
    ranges : list of callable
        Ranges in which to integrate over, specified as functions from variables in
        outer integrals to a tuple (lower_limit, upper_limit).
    tol : float
        Absolute tolerance.
    quadrature : callable
        Underlying quadrature rule to use. Should be a function like
        quadrature(f, a, b, norm_func) which returns an estimate of the integral and
        and an estimate of the error.
    norm_func: callable
        Vector norm to use for error estimation.
    """
    depth = len(ranges)

    # Doesn't depend on a variable, these are the limits of the outer intergral
    a, b = ranges[0]()

    if depth == 1:
        return quad_vec(f, a, b, tol, quadrature, norm_func)

    ranges = ranges[1:]

    # The inner integral is a function of the variable we are currently integrating over
    # Originally, f is a function of e.g. x, y, z
    #  e.g. ∫ ∫ ∫ xyz dx dy dz
    # Now we need to consider partial_f, which is a function of just x and y, and treat
    # z as a constant
    def inner_integral(z):
        def partial_f(*args):
            return f(*args, z)

        new_ranges = []

        for rng in ranges:
            new_ranges.append(_bind_last_argument(rng, z))

        est, _ = ndquad_vec(partial_f, new_ranges, tol, quadrature, norm_func)

        return est

    est, err = quadrature(inner_integral, a, b, norm_func)

    return est, err


def _bind_last_argument(f, z):
    """Fixes the last argument of f as z"""
    def new_func(*args):
        return f(*args, z)

    return new_func
