#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import warnings
import math
import random
from probability_distributions.continuous.special_functions import beta, incbeta
from probability_distributions.continuous.chi2 import chi2_rvs

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def f_pdf(x, dof1, dof2):
    r"""
    f_pdf
    ===========
    Computes the probability density function (PDF) of the F-distribution 
    for a given value `x` and degrees of freedom `dof1` and `dof2`.

    Mathematical Definition
    ----------
    .. math::
        PDF_F(x, dof_1, dof_2) = \frac{\sqrt{(dof_1 \times x)^{dof_1} \times dof_2^{dof_2}}}
        {x \times B \left(\frac{dof_1}{2}, \frac{dof_2}{2} \right)}

    Where:
        
        - :math:`B` is the beta function,
        - :math:`x` is the value at which the PDF is evaluated,
        - :math:`dof_1` and :math:`dof_2` are the degrees of freedom.
        
    Parameters
    ----------
    x : float or int
        The value at which the probability density function is evaluated.

    dof1 : int
        Degrees of freedom for the numerator.
        
    dof2 : int
        Degrees of freedom for the denominator.

    Returns
    ----------
    float
        The probability density of the F-distribution at the specified value `x`
        for the given `dof1` and `dof2`.
    
    Examples
    ----------
    >>> f_pdf(1, 2, 3)
    0.27885480092693427
    
    >>> f_pdf(4, 5, 6)
    0.03139105459757474
    
    >>> f_pdf(0.1, 999, 1000)
    0.0

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/F-distribution
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != float and type(x) != int:
        raise TypeError("Parameter 'x' is not a float or integer.")
    elif x <= 0:
        raise ValueError("Parameter 'x' must be greater than 0.")

    if type(dof1) != int:
        raise TypeError("Parameter 'dof1' is not an integer.")
    elif dof1 <= 0:
        raise ValueError("Parameter 'dof1' must be greater than 0.")
        
    if type(dof2) != int:
        raise TypeError("Parameter 'dof2' is not an integer.")
    elif dof2 <= 0:
        raise ValueError("Parameter 'dof2' must be greater than 0.")

    # Engine
    #-------------------------------------------------------------------------#
    # Numerator 01
    num_01 = dof1 * math.log(dof1 * x) + dof2 * math.log(dof2)

    # Denominator 01
    den_01 = (dof1 + dof2) * math.log(dof1 * x + dof2)
    
    # Numerator 02
    num_02 = math.sqrt(math.exp(num_01 - den_01))
    
    # Denominator 02
    den_02 = x * beta(dof1 / 2, dof2 / 2)

    # Calculate the pdf
    pdf = num_02 / den_02

    # Output
    #-------------------------------------------------------------------------#                
    return pdf

def f_cdf(x, dof1, dof2):
    r"""
    f_cdf
    ===========
    Computes the cumulative distribution function (CDF) of the F-distribution.

    Mathematical Definition
    ----------
    The cumulative distribution function of the F-distribution with degrees of freedom 
    :math:`dof_1` and :math:`dof_2` at a value :math:`x \geq 0` is given by:

    .. math::
        CDF_{F}(x, dof_1, dof_2) = I_{\frac{dof_1 \times x}{dof_1 \times x + dof_2}}
        \left( \frac{dof_1}{2}, \frac{dof_2}{2} \right)
        
    Where:
        
        - :math:`I_z(a, b)` is the regularized incomplete beta function.

    Parameters
    ----------
    x : float or int
        The value at which to evaluate the CDF.

    dof1 : int
        Degrees of freedom for the numerator.
        
    dof2 : int
        Degrees of freedom for the denominator.
    
    Returns
    ----------
    float
        The cumulative probability up to the given value :math:`x`, corresponding to the 
        CDF of the F-distribution with the specified degrees of freedom.
    
    Examples
    ----------
    >>> f_cdf(1, 2, 3)
    0.535241998461282
    
    >>> f_cdf(4, 5, 6)
    0.9392882660117253
    
    >>> f_cdf(0.1, 999, 1000)
    1.0420447741994137e-242

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/F-distribution
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != float and type(x) != int:
        raise TypeError("Parameter 'x' is not a float or integer.")
    elif x < 0:
        raise ValueError("Parameter 'x' must be greater or equal to 0.")

    if type(dof1) != int:
        raise TypeError("Parameter 'dof1' is not an integer.")
    elif dof1 <= 0:
        raise ValueError("Parameter 'dof1' must be greater than 0.")
        
    if type(dof2) != int:
        raise TypeError("Parameter 'dof2' is not an integer.")
    elif dof2 <= 0:
        raise ValueError("Parameter 'dof2' must be greater than 0.")

    # Engine
    #-------------------------------------------------------------------------#
    if x == 0:
        cdf = 0
    else:
        cdf = incbeta((dof1 * x) / (dof1 * x + dof2), dof1 / 2, dof2 / 2)

    # Output
    #-------------------------------------------------------------------------#
    return cdf

def f_invcdf(p, dof1, dof2, tol = 1e-10, max_iter = 100, warn = True):
    r"""
    f_invcdf
    ===========
    Computes the inverse cumulative distribution function (CDF) for the F-distribution.

    This function uses the bisection method to compute the value `x` such that the 
    cumulative distribution function (CDF) of the F-distribution with degrees of freedom 
    `dof1` and `dof2` equals the given probability `p`.
    
    Mathematical Definition
    ----------
    .. math::
        CDF^{-1}_{F}(p, dof_1, dof_2) = x

    Parameters
    ----------
    p : float or int
        The cumulative probability for which the inverse CDF is to be computed.
        It should lie in the interval (0, 1).
        
    dof1 : int
        The degrees of freedom for the numerator of the F-distribution.
        
    dof2 : int
        The degrees of freedom for the denominator of the F-distribution.
    
    tol : float or int, optional, default=1e-10
        The tolerance for convergence. The function will stop iterating once the difference 
        between the current estimate and the desired probability `p` is less than `tol`.
    
    max_iter : int, optional, default=100
        The maximum number of iterations for the bisection method. If convergence is not achieved 
        within `max_iter` iterations, the function will issue a warning (if `warn` is True).
    
    warn : bool, optional, default=True
        Whether to issue a warning if convergence is not reached within the maximum number of iterations.
        If `True`, a warning will be shown.

    Returns
    ----------
    float
        The value `x` such that the F-distribution CDF with degrees of freedom `dof1` and `dof2`
        is equal to the given probability `p`.
    
    Examples
    ----------
    >>> f_invcdf(0.535241998461282, 2, 3)
    0.9999999999999989
    
    >>> f_invcdf(0.9392882660117253, 5, 6)
    4.0
    
    >>> f_invcdf(1.0420447741994137e-242, 999, 1000)
    0.1

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/F-distribution
    .. [2] https://en.wikipedia.org/wiki/Bisection_method
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(p) != float and type(p) != int:
        raise TypeError("Parameter 'p' is not a float or integer.")
    elif p <= 0 or p >= 1:
        raise ValueError("Parameter 'p' must be within the interval ]0; 1[.")

    if type(dof1) != int:
        raise TypeError("Parameter 'dof1' is not an integer.")
    elif dof1 <= 0:
        raise ValueError("Parameter 'dof1' must be greater than 0.")
        
    if type(dof2) != int:
        raise TypeError("Parameter 'dof2' is not an integer.")
    elif dof2 <= 0:
        raise ValueError("Parameter 'dof2' must be greater than 0.")

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
    lower_bound = 0
    upper_bound = 1000 # A large enough upper bound

    # Iterations
    conv = False
    for _ in range(max_iter):
        # Calculate the initial guess and the function value
        x = (lower_bound + upper_bound) / 2
        f_x = f_cdf(x, dof1, dof2)

        # Check for convergence
        if abs(f_x - p) < tol:
            conv = True
            invcdf = x

        # Update the guess using the Bisection method
        if f_x < p:
            lower_bound = x
        else:
            upper_bound = x

    # Convergence failed
    if conv == False:
        if warn == True:
            warnings.warn("Convergence was not achieved. Either increase the value of the parameter 'tol' or the value of the parameter 'max_iter'.", UserWarning)
        invcdf = x

    # Output
    #-------------------------------------------------------------------------#
    return invcdf

def f_rvs(dof1, dof2, size = 1, seed = None):
    r"""
    f_rvs
    ===========
    Generate a random number or a list of random numbers from the F-distribution
    with specified degrees of freedom.

    Mathematical Definition
    ----------
    .. math::
        RVS_{F}(dof_1, dof_2) \sim F(dof_1, dof_2)

    Parameters
    ----------
    dof1 : int
        The degrees of freedom for the numerator of the F-distribution.
        
    dof2 : int
        The degrees of freedom for the denominator of the F-distribution.

    size : int, optional, default=1
        The number of random variates to generate. If `size = 1`, the function returns a single float.
        If `size > 1`, the function returns a list of random numbers.

    seed : int, optional, default=None
        The seed for the random number generator. Setting the seed ensures that the sequence of random
        numbers is reproducible. If not provided, the random number generator is not seeded.

    Returns
    ----------
    float or list of float
        A single random number if `size = 1`, or a list of random numbers if `size > 1`,
        drawn from the F-distribution with the specified degrees of freedom
        :math:`(dof_1, dof_2)`
    
    Examples
    ----------
    >>> f_rvs(1, 2, 1, 12345)
    12.695102894643545

    >>> f_rvs(1, 2, 5, 12345)
    [12.695102894643545,
     0.17012702089062703,
     9.679951795238694,
     0.44608859707676235,
     0.42226212161057347]

    >>> f_rvs(7, 10, 1, 6789)
    3.551840770283067

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/F-distribution
    .. [2] https://en.wikipedia.org/wiki/Chi-squared_distribution
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(dof1) != int:
        raise TypeError("Parameter 'dof1' is not an integer.")
    elif dof1 <= 0:
        raise ValueError("Parameter 'dof1' must be greater than 0.")
        
    if type(dof2) != int:
        raise TypeError("Parameter 'dof2' is not an integer.")
    elif dof2 <= 0:
        raise ValueError("Parameter 'dof2' must be greater than 0.")

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
    def rvs_gen(dof1, dof2):
        u1 = chi2_rvs(dof1)
        u2 = chi2_rvs(dof2)
        return (u1 / dof1) / (u2 / dof2)

    # Generate the random variates (rvs)
    if size == 1:
        rvs = rvs_gen(dof1, dof2)
    else:
        rvs = [rvs_gen(dof1, dof2) for _ in range(size)]

    # Output
    #-------------------------------------------------------------------------#
    return rvs
