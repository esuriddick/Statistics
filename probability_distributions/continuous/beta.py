#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import warnings
import random
from special_functions import beta, incbeta

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def beta_pdf(x, a, b):
    r"""
    Retrieves the probability density function (PDF) of the Beta distribution.

    Function
    ===========
    .. math::
        PDF_{\beta}(x, a, b) = \frac{x^{a - 1}(1 - x)^{b - 1}}{B(a, b)}

    Parameters
    ===========
    x : integer or float

    Value at which the probability density function is evaluated.

    a : float or integer

    Shape parameter of the beta distribution. It influences the behaviour of the function for small values of t.

    b : float or integer

    Shape parameter of the beta distribution. It influences the behaviour of the function for values of t close to 1.

    Examples
    ===========
    >>> beta_pdf(0.2, 1, 20)
    0.2882303761517119

    >>> beta_pdf(0, 175, 200)
    0.0

    >>> beta_pdf(0.75, 5, 3)
    2.076416015625003

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Beta_distribution)
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
    num = (x ** (a - 1)) * ((1 - x) ** (b - 1))

    # Denominator
    den = beta(a, b)

    # Calculate the pdf
    pdf = num / den

    # Output
    #-------------------------------------------------------------------------#                
    return pdf

def beta_cdf(x, a, b):
    r"""
    Retrieves the cumulative distribution function (CDF) of the Beta distribution.

    Function
    ===========
    .. math::
        CDF_{\beta}(x, a, b) = I_x(a, b)

    Parameters
    ===========
    x : integer or float

    Value up to which the cumulative probability is calculated.

    a : float or integer

    Shape parameter of the beta distribution. It influences the behaviour of the function for small values of t.

    b : float or integer

    Shape parameter of the beta distribution. It influences the behaviour of the function for values of t close to 1.

    Examples
    ===========
    >>> beta_cdf(0.2, 1, 20)
    0.9884707849539315

    >>> beta_cdf(0, 175, 200)
    0.0

    >>> beta_cdf(0.75, 5, 3)
    0.7564086914062493

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Beta_distribution)
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

def beta_invcdf(p, a, b, tol = 1e-10, max_iter = 100, warn = 1):
    r"""
    Retrieves the inverse of the cumulative distribution function (CDF) of the Beta distribution.

    Since there is no closed-form expression (i.e., algebraic expression) for the inverse of the CDF of the Beta distribution, a root-finding method (e.g., bisection) must be used to invert the CDF.

    Due to its better performance and the fact that the derivate is equal to the PDF, the Newton-Raphson Method is used to invert the CDF.

    Function
    ===========
    .. math::
        CDF^{-1}_{\beta}(p, dof) = I_p^{-1}(a, b)

    Parameters
    ===========
    p : integer or float

    Cumulative probability value of a Beta distribution with parameters a and b.

    a : float or integer

    Shape parameter of the beta distribution. It influences the behaviour of the function for small values of t.

    b : float or integer

    Shape parameter of the beta distribution. It influences the behaviour of the function for values of t close to 1.

    tol : integer or float

    Acceptable error margin for the solution. It determines how close the function value must be to zero for the method to consider the solution as converged. A smaller tol value means higher accuracy but may require more iterations. Default value is 1e-10.

    max_iter : integer

    Maximum number of iterations the Newton-Raphson method will perform before stopping. It acts as a safeguard to prevent infinite loops in case the method does not converge. Default value is 100.

    warn : integer

    Whether a warning message should be sent when convergence is not reached. Default value is 1 (i.e., warning will be sent).

    Examples
    ===========
    >>> beta_invcdf(0.75, 1, 20)
    0.06696700846331306

    >>> beta_invcdf(0.99, 175, 200)
    0.5266839677807172

    >>> beta_invcdf(0.01, 5, 3)
    0.23632356383714626

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Beta_distribution)
    .. [2] Wikipedia (https://en.wikipedia.org/wiki/Newton%27s_method)
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

    if type(warn) != int:
        raise TypeError("Parameter 'warn' is not an integer.")
    elif warn not in [0, 1]:
        raise ValueError("Parameter 'warn' must be equal to 0 or 1.")

    # Engine
    #-------------------------------------------------------------------------#
    # Initial guess
    x = a / (a + b)

    # Iterations
    conv = False
    for _ in range(max_iter):
        # Calculate the function value and its derivative
        f_x = beta_cdf(x, a, b) - p
        f_prime_x = beta_pdf(x, a, b)

        # Update the guess using Newton-Raphson formula
        x_new = x - f_x / f_prime_x

        # Check for convergence
        if abs(x_new - x) < tol:
            invcdf = x_new
            conv = True
            break

        # Update value
        x = min(max(x_new, tol), 1 - tol)

    # Convergence failed
    if conv == False:
        if warn == 1:
            warnings.warn("Convergence was not achieved. Either decrease the value of the parameter 'tol' or increase the value of the parameter 'max_iter'.", UserWarning)
        invcdf = x

    # Output
    #-------------------------------------------------------------------------#
    return invcdf

def beta_rvs(a, b, size = 1, seed = None):
    r"""
    Generate a random number or a list of random numbers based on the Beta distribution.

    Function
    ===========
    .. math::
        RVS_{\beta}(dof) \sim \beta(a, b)

    Parameters
    ===========
    a : float or integer

    Shape parameter of the beta distribution. It influences the behaviour of the function for small values of t.

    b : float or integer

    Shape parameter of the beta distribution. It influences the behaviour of the function for values of t close to 1.

    size : integer

    Number of random variates. If size = 1, the output is a float; otherwise, it is a list.

    seed : integer

    The seed determines the sequence of random numbers generated (i.e., the same seed will generate the exact same random number or list of random numbers).

    Examples
    ===========
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
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Beta_distribution)
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
