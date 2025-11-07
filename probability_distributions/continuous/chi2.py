#-----------------------------------------------------------------------------#
# ---- NATIVE MODULES
#-----------------------------------------------------------------------------#
import warnings
import math
import random

#-----------------------------------------------------------------------------#
# ---- CUSTOM MODULES
#-----------------------------------------------------------------------------#
import sys
import os
# Dynamically adjust the path to include the parent directory,
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from probability_distributions.continuous.norm import norm_rvs
from probability_distributions.continuous.special_functions import incgamma

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def chi2_pdf(x, dof):
    r"""
    chi2_pdf
    ===========
    Computes the probability density function (PDF) of the Chi-squared distribution.

    The Chi-squared distribution is a special case of the Gamma distribution with
     shape parameter :math:`\frac{dof}{2}` and scale parameter :math:`2`, where
     `dof` is the degrees of freedom.

    Mathematical Definition
    ----------
    .. math::
        PDF_{\chi^2}(x, dof) = \frac{x^{dof / 2 - 1} e^{-x / 2}}{2^{dof / 2}\Gamma(dof / 2)}

    Parameters
    ----------
    x : float or int
        The value at which the probability density function is evaluated.
        It must be non-negative :math:`(x \geq 0)`.

    dof : int
        The degrees of freedom (dof) of the Chi-squared distribution.
        It must be a positive integer :math:`(dof > 0)` and determines the shape
        of the distribution.

    Returns
    ----------
    float
        The value of the probability density function at the given value of
        :math:`x` and degrees of freedom :math:`dof`.


    Examples
    ----------
    >>> chi2_pdf(10, 5)
    0.02833455534173445

    >>> chi2_pdf(2, 3)
    0.2075537487102974

    >>> chi2_pdf(5, 5)
    0.12204152134938734

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Chi-squared_distribution
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != float and type(x) != int:
        raise TypeError("Parameter 'x' is not a float or integer.")
    elif x < 0:
        raise ValueError("Parameter 'x' must be greater or equal to 0.")

    if type(dof) != int:
        raise TypeError("Parameter 'dof' is not an integer.")
    elif dof <= 0:
        raise ValueError("Parameter 'dof' must be greater than 0.")

    # Engine
    #-------------------------------------------------------------------------#
    try:
        # Numerator
        num = math.log(x ** (dof / 2 - 1)) + math.log(math.e ** (-x / 2))
   
        # Denominator
        den = math.log(2 ** (dof / 2)) + math.lgamma(dof / 2)
   
        # Calculate the pdf
        pdf = math.exp(num - den)
       
    # For very large degrees of freedom
    except:
        pdf = 0

    # Output
    #-------------------------------------------------------------------------#                
    return pdf

def chi2_cdf(x, dof):
    r"""
    chi2_cdf
    ===========
    Calculates the cumulative distribution function (CDF) for the Chi-squared
    distribution.

    The CDF represents the probability that a random variable following a
    Chi-squared distribution with a given number of degrees of freedom
    (:math:`dof`) will take a value less than or equal to :math:`x`.

    Mathematical Definition
    ----------
    .. math::
        CDF_{\chi^2}(x, dof) = \frac{\gamma \left( \frac{dof}{2}, \frac{x}{2} \right)}{\Gamma \left( \frac{dof}{2} \right)}

    Where:
        
    - :math:`\gamma` is the lower incomplete gamma function
    - :math:`\Gamma` is the Gamma function
    
    Parameters
    ----------
    x : float or int
        The value up to which the cumulative probability is calculated.
        This can be any non-negative real number :math:`(x \geq 0)`.
    
    dof : int
        The degrees of freedom of the Chi-squared distribution. It must be a
        positive integer :math:`(dof > 0)`, and it determines the shape of the
        distribution.

    Returns
    ----------
    float
        The cumulative probability (CDF) for the Chi-squared distribution with the
        given degrees of freedom :math:`(dof)` and up to the value :math:`x`.

    Examples
    ----------
    >>> chi2_cdf(10, 5)
    0.9247647538511172

    >>> chi2_cdf(2, 3)
    0.4275932955279461

    >>> chi2_cdf(5, 5)
    0.5841198130032876

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Chi-squared_distribution
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != float and type(x) != int:
        raise TypeError("Parameter 'x' is not a float or integer.")
    elif x < 0:
        raise ValueError("Parameter 'x' must be greater or equal to 0.")

    if type(dof) != int:
        raise TypeError("Parameter 'dof' is not an integer.")
    elif dof <= 0:
        raise ValueError("Parameter 'dof' must be greater than 0.")

    # Engine
    #-------------------------------------------------------------------------#
    try:
        # Numerator
        num = math.log(incgamma(x / 2, dof / 2))
   
        # Denominator
        den = math.lgamma(dof / 2)
   
        # Calculate the pdf
        cdf = math.exp(num - den)

    # For very large degrees of freedom
    except:
        cdf = 0

    # Output
    #-------------------------------------------------------------------------#                
    return cdf

def chi2_invcdf(p, dof, tol = 1e-10, max_iter = 100, warn = True):
    r"""
    chi2_invcdf
    ===========
    Computes the inverse of the cumulative distribution function (CDF) for the
    Chi-squared distribution.

    Since there is no closed-form solution (i.e., no algebraic expression) for the
    inverse CDF of the Chi-squared distribution, numerical methods are used to
    approximate it. Specifically, the Newton-Raphson method is employed due to its
    efficiency, and because the derivative of the CDF is equivalent to the
    probability density function (PDF) of the Chi-squared distribution.

    Mathematical Definition
    ----------
    .. math::
        CDF^{-1}_{\chi^2}(p, dof) = x

    Where:

    - :math:`p` is the cumulative probability
    - :math:`dof` is the degrees of freedom

    Parameters
    ----------
    p : float or int
        The cumulative probability value (between 0 and 1) for which to compute
        the inverse CDF.
        This represents the probability that a random variable from the Chi-squared
        distribution will be less than or equal to a value :math:`x`.
    
    dof : int
        The degrees of freedom of the Chi-squared distribution. It influences
        the shape of the distribution and must be a positive integer.
    
    tol : float, optional, default = 1e-10
        The tolerance for convergence, which defines the acceptable error margin
        for the solution.
        The iteration stops when the difference between successive estimates is
        smaller than `tol`. A smaller `tol` value increases the accuracy of the
        result but may require more iterations.
        
    max_iter : int, optional, default=100
        The maximum number of iterations allowed for the Newton-Raphson method
        to converge.
        If convergence is not achieved within this limit, an error will be raised.
    
    warn : bool, optional, default=True
        If set to True, a warning will be issued when the method does not converge
        within the specified number of iterations.

    Returns
    ----------
    float
        The value :math:`x` such that the cumulative probability of the
        Chi-squared distribution with :math:`dof` degrees of freedom equals the
        given probability :math:`p`. This is the solution to the equation
        :math:`P(X \leq x) = p`, where :math:`X \sim \chi^2(dof)`.

    Examples
    ----------
    >>> chi2_invcdf(0.9247647538511172, 5)
    9.99999999999997

    >>> chi2_invcdf(0.4275932955279461, 3)
    1.999999999999726

    >>> chi2_invcdf(0.5841198130032876, 5)
    5.0

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Chi-squared_distribution
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(p) != float and type(p) != int:
        raise TypeError("Parameter 'p' is not a float or integer.")
    elif p < 0 or p > 1:
        raise ValueError("Parameter 'p' must be within the interval [0; 1].")

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
    if p == 0:
        invcdf = 0
    else:   
        # Initial guess
        x = dof
   
        # Iterations
        conv = False
        for _ in range(max_iter):
            # Calculate the function value and its derivative
            f_x = chi2_cdf(x, dof) - p
            f_prime_x = chi2_pdf(x, dof)
            
            # Check for convergence
            if abs(f_x) < tol:
                invcdf = x
                conv = True
                break
   
            # Update the guess using Newton-Raphson formula
            x_new = x - f_x / f_prime_x
   
            # Update value
            x = max(x_new, 0)
   
        # Convergence failed
        if conv == False:
            if warn == True:
                warnings.warn("Convergence was not achieved. Either increase the value of the parameter 'tol' or the value of the parameter 'max_iter'.", UserWarning)
            invcdf = x
   
    # Output
    #-------------------------------------------------------------------------#
    return invcdf

def chi2_rvs(dof, size = 1, seed = None):
    r"""
    chi2_rvs
    ===========
    Generate random numbers from a Chi-squared distribution with specified degrees
    of freedom :math:`(dof)`.

    Mathematical Definition
    ----------
    .. math::
        RVS_{\chi^2}(dof) \sim \chi^2(dof)

    Parameters
    ----------
    dof : int
        The degrees of freedom (dof) of the Chi-squared distribution, which
        determines the shape of the distribution. 
        Must be a positive integer.

    size : int, optional, default=1
        The number of random variates to generate. If `size` is 1, the output is
        a single float value; otherwise, it is a list of random variates.

    seed : int, optional, default=None
        A seed for the random number generator. If provided, it ensures
        reproducibility by generating the same sequence of random numbers for the
        same seed. 
        Default is `None`, which means no specific seed is used.
        
    Returns
    ----------
    float or list of float
        A random number (float) if `size` is 1, or a list of floats containing
        `size` random variates drawn from the Chi-squared distribution.

    Examples
    ----------
    >>> chi2_rvs(10, 1,12345)
    7.073580693820499

    >>> chi2_rvs(10, 5, 12345)
    [7.073580693820499,
     20.200019282318873,
     6.330957191501982,
     9.21042501387158,
     19.885553366269868]

    >>> chi2_rvs(7, 1, 6789)
    8.356440647750137

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Chi-squared_distribution
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
        w = sum(norm_rvs() ** 2 for _ in range(dof))
        return w

    # Generate the random variates (rvs)
    if size == 1:
        rvs = rvs_gen(dof)
    else:
        rvs = [rvs_gen(dof) for _ in range(size)]

    # Output
    #-------------------------------------------------------------------------#
    return rvs
