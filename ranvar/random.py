import cython as pyx

if pyx.compiled:
    from cython.cimports.libc.math import ceil as c_ceil
    from cython.cimports.libc.math import floor as c_floor
    from cython.cimports.libc.math import log as c_log
    from cython.cimports.libc.math import sqrt as c_sqrt
    from cython.cimports.libc.math import exp as c_exp
    from cython.cimports.libc.time import time as c_time
    from cython.cimports.libc.stdlib import rand as c_rand
    from cython.cimports.libc.stdlib import srand as c_srand
    from cython.cimports.libc.stdlib import RAND_MAX as c_RAND_MAX
else:
    pass

c_srand(c_time(pyx.NULL))


@pyx.cfunc
@pyx.cdivision(True)
def rand() -> pyx.double:
    out: pyx.double = pyx.cast(pyx.double, c_rand()) / pyx.cast(pyx.double, c_RAND_MAX)
    return out


@pyx.cfunc
def randint(l: pyx.double, h: pyx.double) -> pyx.double:
    l2: pyx.double = l - 1
    out: pyx.double = c_ceil((h - l2) * rand() + l2)
    if out < l:
        out = l
    return out


@pyx.cfunc
def randnorm(mu: pyx.double, stdev: pyx.double) -> pyx.double:
    """
    https://en.wikipedia.org/wiki/Marsaglia_polar_method
    """
    while True:
        x: pyx.double = 2*rand() - 1
        y: pyx.double = 2*rand() - 1
        s: pyx.double = x**2 + y**2
        if s < 1:
            break
    z: pyx.double = x * c_sqrt(-2*c_log(s) / s)
    return z * stdev + mu


@pyx.cfunc
@pyx.cdivision(True)
def randexp(rate: pyx.double) -> pyx.double:
    """
    Sample from exponential distribution with given rate parameter.
    Uses inverse transform sampling: X = -ln(U) / rate
    """
    u: pyx.double = rand()
    while u == 0:
        u = rand()
    return -c_log(u) / rate


@pyx.cfunc
@pyx.cdivision(True)
def randgamma(shape: pyx.double, scale: pyx.double) -> pyx.double:
    """
    Sample from gamma distribution with given shape (k) and scale (theta) parameters.
    Uses Marsaglia and Tsang's method for shape >= 1, and Ahrens-Dieter method for shape < 1.
    https://en.wikipedia.org/wiki/Gamma_distribution#Generating_gamma-distributed_random_variables
    """
    d: pyx.double
    c: pyx.double
    x: pyx.double
    v: pyx.double
    u: pyx.double
    shape_use: pyx.double = shape

    # For shape < 1, use the transformation: if X ~ Gamma(shape+1), then X * U^(1/shape) ~ Gamma(shape)
    if shape < 1:
        shape_use = shape + 1

    # Marsaglia and Tsang's method for shape >= 1
    d = shape_use - 1.0 / 3.0
    c = 1.0 / c_sqrt(9.0 * d)

    while True:
        while True:
            x = randnorm(0, 1)
            v = 1.0 + c * x
            if v > 0:
                break

        v = v * v * v
        u = rand()

        if u < 1.0 - 0.0331 * (x * x) * (x * x):
            break

        if c_log(u) < 0.5 * x * x + d * (1.0 - v + c_log(v)):
            break

    result: pyx.double = d * v * scale

    # Transform back if original shape < 1.
    # U must be strictly in (0, 1) for the Ahrens-Dieter method.
    # U == 0 causes log(0) = -inf, and U == 1 causes U^(1/shape) = 1
    # which skips the scaling entirely, returning Gamma(shape+1) instead of Gamma(shape).
    if shape < 1:
        u = rand()
        while u <= 0 or u >= 1:
            u = rand()
        result = result * c_exp(c_log(u) / shape)

    return result


@pyx.cfunc
@pyx.cdivision(True)
def randpoisson(lam: pyx.double) -> pyx.double:
    """
    Sample from Poisson distribution with given rate (lambda) parameter.
    Uses Knuth's algorithm for small lambda, normal approximation for large lambda.
    """
    k: pyx.int
    p: pyx.double
    L: pyx.double
    u: pyx.double
    result: pyx.double

    if lam <= 0:
        return 0.0

    if lam < 30:
        # Knuth's algorithm for small lambda
        L = c_exp(-lam)
        k = 0
        p = 1.0

        while True:
            k = k + 1
            u = rand()
            p = p * u
            if p <= L:
                break

        result = pyx.cast(pyx.double, k - 1)
    else:
        # Normal approximation for large lambda
        result = randnorm(lam, c_sqrt(lam))
        if result < 0:
            result = 0
        result = c_floor(result + 0.5)

    return result


@pyx.cfunc
@pyx.cdivision(True)
def randpert(low: pyx.double, mode: pyx.double, high: pyx.double) -> pyx.double:
    """
    Sample from a PERT distribution parameterised by minimum, most-likely, and maximum.
    Uses a Beta(α, β) variate via the Gamma-ratio method where
    α = 1 + 4*(mode-low)/(high-low) and β = 1 + 4*(high-mode)/(high-low).
    https://en.wikipedia.org/wiki/PERT_distribution
    """
    span: pyx.double = high - low
    alpha: pyx.double = 1.0 + 4.0 * (mode - low) / span
    beta: pyx.double  = 1.0 + 4.0 * (high - mode) / span
    g1: pyx.double = randgamma(alpha, 1.0)
    g2: pyx.double = randgamma(beta,  1.0)
    return low + span * g1 / (g1 + g2)


@pyx.cfunc
@pyx.cdivision(True)
def randnegbinom(n: pyx.double, p: pyx.double) -> pyx.double:
    """
    Sample from negative binomial distribution with parameters n and p.
    Uses the Gamma-Poisson mixture representation:
    If Y ~ Gamma(n, (1-p)/p) and X | Y ~ Poisson(Y), then X ~ NegativeBinomial(n, p)
    """
    scale: pyx.double = (1.0 - p) / p
    gamma_sample: pyx.double = randgamma(n, scale)
    return randpoisson(gamma_sample)
