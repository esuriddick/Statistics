#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import warnings
import math
import random
from probability_distributions.continuous.special_functions import beta, incbeta

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def beta_pdf(x, a, b):
    r"""
    beta_pdf
    ===========
    Computes the probability density function (PDF) of the Beta distribution at point :math:`x` with shape parameters :math:`a` and :math:`b`.

    Mathematical Definition
    ----------
    .. math::
        \text{PDF}_{\beta}(x; a, b) = \frac{x^{a - 1}(1 - x)^{b - 1}}{B(a, b)}

    where :math:`B(a, b)` is the Beta function.

    Parameters
    ----------
    x : float or int
        The point at which the PDF is evaluated. Must be in the interval :math:`[0, 1]`.
    
    a : float or int
        First shape parameter of the Beta distribution. Must be greater than 0.
    
    b : float or int
        Second shape parameter of the Beta distribution. Must be greater than 0.
    
    Returns
    ----------
    float
        The value of the Beta distribution PDF at point :math:`x` for the given shape parameters :math:`a` and :math:`b`.
    
    Examples
    ----------
    >>> beta_pdf(0.2, 1, 20)
    0.2882303761517118

    >>> beta_pdf(0, 175, 200)
    0.0

    >>> beta_pdf(0.75, 5, 3)
    2.0764160156250036

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta_distribution
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != float and type(x) != int:
        raise TypeError("Parameter 'x' is not a float or integer.")
    elif x < 0 or x > 1:
        raise ValueError("Parameter 'x' must be within the interval [0; 1]")

    if type(a) != float and type(a) != int:
        raise TypeError("Parameter 'a' is not a float or integer.")
    elif a <= 0:
        raise ValueError("Parameter 'a' must be greater than 0.")

    if type(b) != float and type(b) != int:
        raise TypeError("Parameter 'b' is not a float or integer.")
    elif b <= 0:
        raise ValueError("Parameter 'b' must be greater than 0.")

    # Engine
    #-------------------------------------------------------------------------#
    # Numerator
    if x == 0 or x == 1:
        return 0
    else:
        num = (a - 1) * math.log(x) + (b - 1) * math.log(1 - x)

    # Denominator
    den = beta(a, b, False)

    # Calculate the pdf
    pdf = num - den

    # Output
    #-------------------------------------------------------------------------#                
    return math.exp(pdf)

def beta_cdf(x, a, b):
    r"""
    beta_cdf
    ===========
    Computes the cumulative distribution function (CDF) of the Beta distribution at point :math:`x` with shape parameters :math:`a` and :math:`b`.

    Mathematical Definition
    ----------
    .. math::
        \mathrm{CDF}_{\beta}(x; a, b) = I_x(a, b)
    
    where :math:`I_x(a, b)` is the regularized incomplete beta function.

    Parameters
    ----------
    x : float or int
        The point at which the CDF is evaluated. Must be in the interval :math:`[0, 1]`.

    a : float or int
        First shape parameter (also known as alpha). Controls the behavior of the distribution near 0.
    
    b : float or int
        Second shape parameter (also known as beta). Controls the behavior of the distribution near 1.

    Returns
    ----------
    float
        The value of the CDF of the Beta distribution evaluated at :math:`x` with shape parameters :math:`a` and :math:`b`.


    Examples
    ----------
    >>> beta_cdf(0.2, 1, 20)
    0.9884707849539315

    >>> beta_cdf(0, 175, 200)
    0.0

    >>> beta_cdf(0.75, 5, 3)
    0.7564086914062493

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta_distribution
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != float and type(x) != int:
        raise TypeError("Parameter 'x' is not a float or integer.")
    elif x < 0 or x > 1:
        raise ValueError("Parameter 'x' must be within the interval [0; 1]")

    if type(a) != float and type(a) != int:
        raise TypeError("Parameter 'a' is not a float or integer.")
    elif a <= 0:
        raise ValueError("Parameter 'a' must be greater than 0.")

    if type(b) != float and type(b) != int:
        raise TypeError("Parameter 'b' is not a float or integer.")
    elif b <= 0:
        raise ValueError("Parameter 'b' must be greater than 0.")

    # Engine
    #-------------------------------------------------------------------------#
    if x == 0:
        cdf = 0.0
    elif x == 1:
        cdf = 1.0
    else:
        cdf = incbeta(x, a, b)

    # Output
    #-------------------------------------------------------------------------#                
    return cdf

def beta_invcdf(p, a, b, tol = 1e-10, max_iter = 100, warn = True):
    r"""
    beta_invcdf
    ===========
    Computes the inverse of the cumulative distribution function (CDF) of the Beta distribution.

    The inverse of the Beta CDF cannot be expressed in closed-form, so a numerical method is required to
    approximate it. This function uses a hybrid approach combining the Newton-Raphson method and the
    bisection method for inversion, leveraging the fact that the derivative of the CDF is equal to the
    probability density function (PDF), which improves convergence and efficiency.

    Mathematical Definition
    ----------
    The inverse CDF (quantile function) for a Beta distribution is mathematically defined as:

    .. math::
        CDF^{-1}_{\beta}(p, a, b) = I_p^{-1}(a, b)
    
    where :math:`p` is the cumulative probability, and :math:`a`, :math:`b` are the shape parameters of the Beta distribution.

    Parameters
    ----------
    p : float or int
        Cumulative probability value for which the inverse CDF is computed. Must be in the interval :math:`]0, 1[`.

    a : float or int
        The first shape parameter of the Beta distribution, influencing the behavior of the distribution near 0.

    b : float or int
        The second shape parameter of the Beta distribution, influencing the behavior of the distribution near 1.

    tol : float or int, optional, default=1e-10
        The acceptable error margin for convergence. The method stops when the difference between successive iterations is smaller than `tol`. A smaller `tol` increases accuracy but requires more iterations.

    max_iter : int, optional, default=100
        The maximum number of iterations the Newton-Raphson method will attempt before stopping. This parameter helps prevent infinite loops in cases where the method fails to converge.

    warn : bool, optional, default=True
        If True, a warning is raised if the method does not converge within the specified `max_iter`.

    Returns
    ----------
    float
        The value :math:`x` such that the CDF of the Beta distribution with parameters :math:`a` and :math:`b` evaluated at :math:`x` equals the input probability :math:`p`.

    Examples
    ----------
    >>> beta_invcdf(0.75, 1, 20)
    0.0669670084631634

    >>> beta_invcdf(0.99, 175, 200)
    0.526683967779479

    >>> beta_invcdf(0.01, 5, 3)
    0.23632356383745984

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta_distribution
    .. [2] https://en.wikipedia.org/wiki/Newton%27s_method
    .. [3] https://en.wikipedia.org/wiki/Bisection_method
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(p) != float and type(p) != int:
        raise TypeError("Parameter 'p' is not a float or integer.")
    elif p <= 0 or p >= 1:
        raise ValueError("Parameter 'p' must be within the interval ]0; 1[.")

    if type(a) != float and type(a) != int:
        raise TypeError("Parameter 'a' is not a float or integer.")
    elif a <= 0:
        raise ValueError("Parameter 'a' must be greater than 0.")

    if type(b) != float and type(b) != int:
        raise TypeError("Parameter 'b' is not a float or integer.")
    elif b <= 0:
        raise ValueError("Parameter 'b' must be greater than 0.")

    if type(tol) != float and type(tol) != int:
        raise TypeError("Parameter 'tol' is not a float or integer.")
    elif tol <= 0:
        raise ValueError("Parameter 'tol' must be greater than 0.")

    if type(max_iter) != int:
        raise TypeError("Parameter 'max_iter' is not an integer.")
    elif max_iter <= 0:
        raise ValueError("Parameter 'max_iter' must be greater than 0.")

    if type(warn) != bool:
        raise TypeError("Parameter 'warn' is not a boolean.")

    # Engine
    #-------------------------------------------------------------------------#
    # Initial guess
    x = a / (a + b)
    
    # Fallback settings - Bisection method
    low, high = 0, 1

    # Iterations
    conv = False
    for _ in range(max_iter):
        # Calculate the function value and its derivative
        f_x = beta_cdf(x, a, b) - p
        f_prime_x = beta_pdf(x, a, b)

        # Check for convergence
        if abs(f_x) < tol:
            invcdf = x
            conv = True
            break

        # Update the guess using Newton-Raphson formula
        if f_prime_x > tol:
            x_new = x - f_x / f_prime_x

        # Fallback to Bisection method if derivative is too small
        else:
            x_new = (low + high) / 2
            
        # Keep x within bounds from Bisection method
        if x_new <= low or x_new >= high:
            x_new = (low + high) / 2
            
        # Update bounds from Bisection method
        if beta_cdf(x_new, a, b) < p:
            low = x_new
        else:
            high = x_new

        # Update value
        x = x_new

    # Convergence failed
    if conv == False:
        if warn == True:
            warnings.warn("Convergence was not achieved. Either increase the value of the parameter 'tol' or the value of the parameter 'max_iter'.", UserWarning)
        invcdf = x

    # Output
    #-------------------------------------------------------------------------#
    return invcdf

def beta_rvs(a, b, size = 1, seed = None):
    r"""
    beta_rvs
    ===========
    Generate random variates based on the Beta distribution.

    The Beta distribution is a continuous probability distribution with two shape parameters, :math:`a` and :math:`b`,
    which control the distribution's shape. The function generates random numbers that follow the Beta
    distribution with the specified parameters.

    Mathematical Definition
    ----------
    Given shape parameters :math:`a` and :math:`b`, the random variates are drawn from the Beta distribution:
    
    .. math::
        RVS_{\beta}(a, b) \sim \beta(a, b)

    Parameters
    ----------
    a : float or int
        The first shape parameter of the Beta distribution. It controls the distribution's behavior near 0.

    b : float or int
        The second shape parameter of the Beta distribution. It controls the distribution's behavior near 1.

    size : int, optional, default=1
        The number of random variates to generate. If `size` is 1, a single float is returned; otherwise, a list of floats is returned.

    seed : int, optional, default=None
        The seed value for the random number generator. Setting a specific seed ensures reproducibility; the same seed will produce the same output each time.

    Returns
    ----------
    float or list of floats
        A single random variate (if `size` is 1), or a list of random variates (if `size` is greater than 1).

    Examples
    ----------
    >>> beta_rvs(20, 12345, 1, 1234)
    0.0007275026736979269

    >>> beta_rvs(20, 12345, 5, 1234)
    [0.0007275026736979269,
     0.0017997533248415066,
     0.0009355644590647561,
     0.0017608371255472904,
     0.0016562916374822491]

    >>> beta_rvs(5, 3, 1, 6789)
    0.6353096405830087

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta_distribution
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(a) != float and type(a) != int:
        raise TypeError("Parameter 'a' is not a float or integer.")
    elif a <= 0:
        raise ValueError("Parameter 'a' must be greater than 0.")

    if type(b) != float and type(b) != int:
        raise TypeError("Parameter 'b' is not a float or integer.")
    elif b <= 0:
        raise ValueError("Parameter 'b' must be greater than 0.")

    if type(size) != int:
        raise TypeError("Parameter 'size' is not an integer.")
    elif size <= 0:
        raise ValueError("Parameter 'size' must be greater than 0.")

    if type(seed) != int and seed != None:
        raise TypeError("Parameter 'seed' is not an integer.")

    # Engine
    #-------------------------------------------------------------------------#
    # Set and store the seed
    if seed != None:
        random.seed(seed)

    # Generate the random variates (rvs)
    if size == 1:
        rvs = random.betavariate(alpha = a, beta = b)
    else:
        rvs = [random.betavariate(alpha = a, beta = b) for _ in range(size)]

    # Output
    #-------------------------------------------------------------------------#
    return rvs
