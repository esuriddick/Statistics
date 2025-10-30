#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import warnings
import math
import random
from probability_distributions.continuous.norm import norm_invcdf, norm_rvs
from probability_distributions.continuous.special_functions import incbeta

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def t_pdf(x, dof):
    r"""
    t_pdf
    ===========
    Computes the probability density function (PDF) of the Student's t-distribution.

    Mathematical Definition
    ----------
    .. math::
        PDF_{T}(x, dof) = \frac{\Gamma \left(\frac{dof + 1}{2} \right)}{\sqrt{\pi \times dof}
            \times \Gamma \left(\frac{dof}{2} \right)} \left( 1 + \frac{x^2}{dof}
             \right)^{-\left( \frac{dof + 1}{2} \right)}

    Where:
        
        - :math:`\Gamma` is the Gamma function,
        - :math:`x` is the value at which the PDF is evaluated,
        - :math:`dof` is the degrees of freedom.
        
    Parameters
    ----------
    x : float or int
        The value at which the probability density function is evaluated.

    dof : int
        The degrees of freedom of the Student's t-distribution. It affects the shape and scale
        of the distribution.

    Returns
    ----------
    float
        The probability density of the Student’s t-distribution at the specified value `x`
        for the given `dof`.
    
    Examples
    ----------
    >>> t_pdf(0, 5)
    0.3796066898224941

    >>> t_pdf(-1.2345, 6789)
    0.18618821724154982

    >>> t_pdf(0.5, 20)
    0.3458086123837425

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Student's_t-distribution
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
    t_cdf
    ===========
    Computes the cumulative distribution function (CDF) of the Student’s t-distribution.

    Mathematical Definition
    ----------
    The CDF is defined as the integral of the probability density function (PDF) up to a given value \(x\). 
    For positive values of \(x\), it is calculated as:

    .. math::
        CDF_{T}(x, dof) = \int_{-\infty}^x PDF_{t}(t, dof) \, dt = 1 - \frac{1}{2} I_x \left( \frac{dof}{2}, \frac{1}{2} \right), \quad \text{for } x > 0

    Due to the symmetry of the Student’s t-distribution, for negative values of \(x\), the CDF is given by:

    .. math::
        CDF_{T}(x, dof) = \frac{1}{2} I_x \left( \frac{dof}{2}, \frac{1}{2} \right), \quad \text{for } x \leq 0

    Parameters
    ----------
    x : float or int
        The value at which to evaluate the CDF. This is the upper limit of the integral.

    dof : int
        The degrees of freedom of the Student's t-distribution. The dof parameter determines
        the shape of the distribution and must be a positive integer.
    
    Returns
    ----------
    float
        The cumulative probability up to the given value :math:`x`, corresponding to the CDF of the
        Student’s t-distribution with the specified degrees of freedom.
    
    Examples
    ----------
    >>> t_cdf(0, 5)
    0.5

    >>> t_cdf(-1.2345, 6789)
    0.10852968701609644

    >>> t_cdf(0.5, 20)
    0.6887340788597041

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Student's_t-distribution
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

def t_invcdf(p, dof, tol = 1e-10, max_iter = 100, warn = True):
    r"""
    t_invcdf
    ===========
    Computes the inverse of the cumulative distribution function (CDF) for the Student's t-distribution.
    
    Mathematical Definition
    ----------
    .. math::
        CDF^{-1}_{T}(p, dof) = x

    Parameters
    ----------
    p : float or int
        The cumulative probability for which the inverse CDF is to be computed. It should lie in the interval (0, 1).
        
    dof : int
        The degrees of freedom of the Student’s t-distribution, which affects the shape of the distribution.
    
    tol : float or int, optional, default=1e-10
        The acceptable error tolerance for convergence. It determines how close the function value must be
        to zero for the method to consider the solution as converged. A smaller value results in higher
        accuracy but may require more iterations.
    
    max_iter : int, optional, default=100
        The maximum number of iterations the Newton-Raphson method will perform. This acts as a safeguard
        to avoid infinite loops in case of non-convergence.
    
    warn : bool, optional, default=True
        Whether to issue a warning if convergence is not reached within the maximum number of iterations.
        If `True`, a warning will be shown.

    Returns
    ----------
    float
        The value :math:`x` that corresponds to the inverse of the cumulative distribution function for
        the Student's t-distribution with the given degrees of freedom.
    
    Examples
    ----------
    >>> t_invcdf(0.75, 5)
    0.7266868438006572

    >>> t_invcdf(0.99, 30)
    2.4572615423814086

    >>> t_invcdf(0.01, 15)
    -2.6024802948993853

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Student's_t-distribution
    .. [2] https://en.wikipedia.org/wiki/Newton%27s_method
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

    if type(warn) != bool:
        raise TypeError("Parameter 'warn' is not a boolean.")

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
        
        # Check for convergence
        if abs(f_x) < tol:
            invcdf = x
            conv = True
            break

        # Update the guess using Newton-Raphson formula
        x_new = x - f_x / f_prime_x

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

def t_rvs(dof, size = 1, seed = None):
    r"""
    t_rvs
    ===========
    Generate a random number or a list of random numbers from the Student's t-distribution.

    Mathematical Definition
    ----------
    .. math::
        RVS_{T}(dof) \sim T(dof)

    Parameters
    ----------
    dof : int
        The degrees of freedom (dof) of the Student's t-distribution. This parameter determines the shape of the distribution.

    size : int, optional, default=1
        The number of random variates to generate. If `size = 1`, the function returns a single float.
        If `size > 1`, the function returns a list of random numbers.

    seed : int, optional, default=None
        The seed for the random number generator. Setting the seed ensures that the sequence of random
        numbers is reproducible. If not provided, the random number generator is not seeded.

    Returns
    ----------
    float or list of float
        A single random number if `size = 1`, or a list of random numbers if `size > 1`, drawn from the
        Student's t-distribution with the specified degrees of freedom :math:`(dof)`.
    
    Examples
    ----------
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
    ----------
    .. [1] https://en.wikipedia.org/wiki/Student's_t-distribution
    .. [2] https://en.wikipedia.org/wiki/Chi-squared_distribution
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
