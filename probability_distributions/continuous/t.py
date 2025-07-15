#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import warnings
import math
import random
from norm import norm_invcdf, norm_rvs
from special_functions import incbeta

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def t_pdf(x, dof):
    r"""
    Retrieves the probability density function (PDF) of the Student’s t distribution.

    Function
    ===========

    .. math::
        PDF_{T}(x, dof) = \frac{\Gamma \left(\frac{dof + 1}{2} \right)}{\sqrt{{\pi} \times dof} \times \Gamma \left(\frac{dof}{2} \right)} \left( 1 + \frac{x^2}{dof} \right)^{-\left( \frac{dof + 1}{2} \right)}

    Parameters
    ===========
    x : integer or float

    Value at which the probability density function is evaluated.

    dof : integer

    Degrees of freedom (dof) of the Student's t distribution, and it determines the shape of the distribution.

    Examples
    ===========
    >>> t_pdf(0, 5)
    0.3796066898224941

    >>> t_pdf(-1.2345, 6789)
    0.18618821724154982

    >>> t_pdf(0.5, 20)
    0.3458086123837425

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Student's_t-distribution)
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != float and type(x) != int:
        raise TypeError("Parameter 'x' is not a float or integer.")

    if type(dof) != int:
        raise TypeError("Parameter 'dof' is not an integer.")
    elif dof <= 0:
        raise ValueError("Parameter 'dof' must be greater than 0.")

    # Engine
    #-------------------------------------------------------------------------#
    # Numerator
    num = math.lgamma((dof + 1) / 2)

    # Denominator
    den =  math.log(math.sqrt(dof * math.pi)) + math.lgamma(dof / 2)

    # Multiplier
    mult = math.log((1 + (x ** 2) / dof) ** (- (dof + 1) / 2))

    # Calculate the pdf
    pdf = math.exp(((num - den) + mult))

    # Output
    #-------------------------------------------------------------------------#                
    return pdf

def t_cdf(x, dof):
    r"""
    Retrieves the cumulative distribution function (CDF) of the Student’s t distribution.

    Function
    ===========
    .. math::
        CDF_{T}(x, dof) = \int_{-\infty}^x PDF_{t}(t, \;dof) \;dt = 1 - \frac{1}{2} I_x \left( \frac{dof}{2}, \frac{1}{2} \right), \;where \;x \gt 0

    Due to the symmetry of the Student's t distribution, when x ≤ 0:

    .. math::
        CDF_{T}(x, dof) = 1 - \left[ 1 - \frac{1}{2} I_x \left( \frac{dof}{2}, \frac{1}{2} \right) \right] = \frac{1}{2} I_x \left( \frac{dof}{2}, \frac{1}{2} \right)

    Parameters
    ===========
    x : integer or float

    Upper limit of the integration.

    dof : integer

    Degrees of freedom (dof) of the Student's t distribution, and it determines the shape of the distribution.

    Examples
    ===========
    >>> t_cdf(0, 5)
    0.5

    >>> t_cdf(-1.2345, 6789)
    0.10852968701609644

    >>> t_cdf(0.5, 20)
    0.6887340788597041

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Student's_t-distribution)
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != float and type(x) != int:
        raise TypeError("Parameter 'x' is not a float or integer.")

    if type(dof) != int:
        raise TypeError("Parameter 'dof' is not an integer.")
    elif dof <= 0:
        raise ValueError("Parameter 'dof' must be greater than 0.")

    # Engine
    #-------------------------------------------------------------------------#
    if x > 0:
        cdf = 1 - 0.5 * incbeta(dof/(x**2 + dof), dof / 2, 0.5)
    else:
        cdf = 0.5 * incbeta(dof/(x**2 + dof), dof / 2, 0.5)

    # Output
    #-------------------------------------------------------------------------#
    return cdf

def t_invcdf(p, dof, tol = 1e-10, max_iter = 100, warn = 1):
    r"""
    Retrieves the inverse of the cumulative distribution function (CDF) of the Student’s t distribution.

    Since there is no closed-form expression (i.e., algebraic expression) for the inverse of the CDF of the Student's t distribution, a root-finding method (e.g., bisection) must be used to invert the CDF.

    Due to its better performance and the fact that the derivate is equal to the PDF, the Newton-Raphson Method is used to invert the CDF.

    Function
    ===========
    .. math::
        CDF^{-1}_{T}(p, dof) = x

    Parameters
    ===========
    p : integer or float

    Cumulative probability value of a Student’s t distribution with degrees of freedom = dof.

    dof : integer

    Degrees of freedom (dof) of the Student's t distribution, and it determines the shape of the distribution.

    tol : integer or float

    Acceptable error margin for the solution. It determines how close the function value must be to zero for the method to consider the solution as converged. A smaller tol value means higher accuracy but may require more iterations. Default value is 1e-10.

    max_iter : integer

    Maximum number of iterations the Newton-Raphson method will perform before stopping. It acts as a safeguard to prevent infinite loops in case the method does not converge. Default value is 100.

    warn : integer

    Whether a warning message should be sent when convergence is not reached. Default value is 1 (i.e., warning will be sent).

    Examples
    ===========
    >>> t_invcdf(0.75, 5)
    0.7266868438007232

    >>> t_invcdf(0.99, 30)
    2.4572615423814135

    >>> t_invcdf(0.01, 15)
    -2.6024802950022194

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Student's_t-distribution)
    .. [2] Wikipedia (https://en.wikipedia.org/wiki/Newton%27s_method)
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(p) != float and type(p) != int:
        raise TypeError("Parameter 'p' is not a float or integer.")
    elif p <= 0 or p >= 1:
        raise ValueError("Parameter 'p' must be within the interval ]0; 1[.")

    if type(dof) != int:
        raise TypeError("Parameter 'dof' is not an integer.")
    elif dof <= 0:
        raise ValueError("Parameter 'dof' must be greater than 0.")

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
    x = norm_invcdf(p)

    # Iterations
    conv = False
    for _ in range(max_iter):
        # Calculate the function value and its derivative
        f_x = t_cdf(x, dof) - p
        f_prime_x = t_pdf(x, dof)

        # Update the guess using Newton-Raphson formula
        x_new = x - f_x / f_prime_x

        # Check for convergence
        if abs(x_new - x) < tol:
            invcdf = x_new
            conv = True
            break

        # Update value
        x = x_new

    # Convergence failed
    if conv == False:
        if warn == 1:
            warnings.warn("Convergence was not achieved. Either decrease the value of the parameter 'tol' or increase the value of the parameter 'max_iter'.", UserWarning)
        invcdf = x

    # Output
    #-------------------------------------------------------------------------#
    return invcdf

def t_rvs(dof, size = 1, seed = None):
    r"""
    Generate a random number or a list of random numbers based on the Student's t distribution.

    Function
    ===========
    .. math::
        RVS_{T}(dof) \sim T(dof)

    Parameters
    ===========
    dof : integer

    Degrees of freedom (dof) of the Student's t distribution, and it determines the shape of the distribution.

    size : integer

    Number of random variates. If size = 1, the output is a float; otherwise, it is a list.

    seed : integer

    The seed determines the sequence of random numbers generated (i.e., the same seed will generate the exact same random number or list of random numbers).

    Examples
    ===========
    >>> t_rvs(5, 1, 12345)
    1.4232328961343679

    >>> t_rvs(5, 5, 12345)
    [1.4232328961343679,
     -0.8949899845560745,
     0.8899937515402874,
     -0.10687772402946792,
     -1.9011444446183]

    >>> t_rvs(55, 1, 6789)
    1.404154177073949

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Student's_t-distribution)
    .. [2] Wikipedia [https://en.wikipedia.org/wiki/Chi-squared_distribution]
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(dof) != int:
        raise TypeError("Parameter 'dof' is not an integer.")
    elif dof <= 0:
        raise ValueError("Parameter 'dof' must be greater than 0.")

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

    # Generator
    def rvs_gen(dof):
        z = norm_rvs()
        w = sum(norm_rvs() ** 2 for _ in range(dof))
        return z / math.sqrt(w / dof)

    # Generate the random variates (rvs)
    if size == 1:
        rvs = rvs_gen(dof)
    else:
        rvs = [rvs_gen(dof) for _ in range(size)]

    # Output
    #-------------------------------------------------------------------------#
    return rvs
