#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import warnings
import math
import random
from norm import norm_rvs
from special_functions import incgamma

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def chi2_pdf(x, dof):
    r"""
    Retrieves the probability density function (PDF) of the Chi-squared distribution.

    Function
    ===========
    .. math::
        PDF_{\chi^2}(x, dof) = \frac{x^{dof / 2 - 1} e^{-x / 2}}{2^{dof / 2}\Gamma(k / 2)}

    Parameters
    ===========
    x : integer or float

    Value at which the probability density function is evaluated.

    dof : integer

    Degrees of freedom (dof) of the Chi-squared distribution, and it determines the shape of the distribution.

    Examples
    ===========
    >>> chi2_pdf(10, 5)
    0.02833455534173445

    >>> chi2_pdf(2, 3)
    0.2075537487102974

    >>> chi2_pdf(5, 5)
    0.12204152134938734

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Chi-squared_distribution)
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
    Retrieves the cumulative distribution function (CDF) of the Chi-squared distribution.

    Function
    ===========
    .. math::
        CDF_{\chi^2}(x, dof) = \frac{\gamma \left( \frac{dof}{2}, \frac{x}{2} \right)}{\Gamma \left( \frac{dof}{2} \right)}

    Parameters
    ===========
    x : integer or float

    Value up to which the cumulative probability is calculated.

    dof : integer

    Degrees of freedom (dof) of the Chi-squared distribution, and it determines the shape of the distribution.

    Examples
    ===========
    >>> chi2_cdf(10, 5)
    0.9247647538511172

    >>> chi2_cdf(2, 3)
    0.4275932955279461

    >>> chi2_cdf(5, 5)
    0.5841198130032876

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Chi-squared_distribution)
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

def chi2_invcdf(p, dof, tol = 1e-10, max_iter = 100, warn = 1):
    r"""
    Retrieves the inverse of the cumulative distribution function (CDF) of the Chi-squared distribution.

    Since there is no closed-form expression (i.e., algebraic expression) for the inverse of the CDF of the Chi-squared distribution, a root-finding method (e.g., bisection) must be used to invert the CDF.

    Due to its better performance and the fact that the derivate is equal to the PDF, the Newton-Raphson Method is used to invert the CDF.

    Function
    ===========
    .. math::
        CDF^{-1}_{\chi^2}(p, dof) = x

    Parameters
    ===========
    p : integer or float

    Cumulative probability value of a Chi-squared distribution with parameter dof.

    dof : integer

    Degrees of freedom (dof) of the Chi-squared distribution, and it determines the shape of the distribution.

    tol : integer or float

    Acceptable error margin for the solution. It determines how close the function value must be to zero for the method to consider the solution as converged. A smaller tol value means higher accuracy but may require more iterations. Default value is 1e-10.

    max_iter : integer

    Maximum number of iterations the Newton-Raphson method will perform before stopping. It acts as a safeguard to prevent infinite loops in case the method does not converge. Default value is 100.

    warn : integer

    Whether a warning message should be sent when convergence is not reached. Default value is 1 (i.e., warning will be sent).

    Examples
    ===========
    >>> chi2_invcdf(0.9247647538511172, 5)
    9.999999999999977

    >>> chi2_invcdf(0.4275932955279461, 3)
    2.0000000000000004

    >>> chi2_invcdf(0.5841198130032876, 5)
    5.0

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Chi-squared_distribution)
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

    if type(warn) != int:
        raise TypeError("Parameter 'warn' is not an integer.")
    elif warn not in [0, 1]:
        raise ValueError("Parameter 'warn' must be equal to 0 or 1.")

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
   
            # Update the guess using Newton-Raphson formula
            x_new = x - f_x / f_prime_x
   
            # Check for convergence
            if abs(x_new - x) < tol:
                invcdf = x_new
                conv = True
                break
   
            # Update value
            x = max(x_new, 0)
   
        # Convergence failed
        if conv == False:
            if warn == 1:
                warnings.warn("Convergence was not achieved. Either decrease the value of the parameter 'tol' or increase the value of the parameter 'max_iter'.", UserWarning)
            invcdf = x
   
    # Output
    #-------------------------------------------------------------------------#
    return invcdf

def chi2_rvs(dof, size = 1, seed = None):
    r"""
    Generate a random number or a list of random numbers based on the Chi-squared distribution.

    Function
    ===========
    .. math::
        RVS_{\chi^2}(dof) \sim \chi^2(dof)

    Parameters
    ===========
    dof : integer

    Degrees of freedom (dof) of the Chi-squared distribution, and it determines the shape of the distribution.

    size : integer

    Number of random variates. If size = 1, the output is a float; otherwise, it is a list.

    seed : integer

    The seed determines the sequence of random numbers generated (i.e., the same seed will generate the exact same random number or list of random numbers).

    Examples
    ===========
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
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Chi-squared_distribution)
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
