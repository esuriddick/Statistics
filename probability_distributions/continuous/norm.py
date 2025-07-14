#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import math
import random

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def norm_pdf(x, mu = 0, sigma = 1):
    r"""
    Retrieves the probability density function (PDF) of the Normal distribution.
    
    Function
    ===========
    .. math::
        PDF_{norm}(x, \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} \times e^{-\frac{(x - \mu)^2}{2 \sigma^2}}
    
    Parameters
    ===========
    x : integer or float
    
    Value at which the probability density function is evaluated.
    
    mu : integer or float
    
    Central location of the distribution. It is the average or expected value of the distribution. Default value is 0.
    
    sigma : integer or float
    
    Spread or dispersion of the distribution. It indicates how much the values of the distribution deviate from the mean. Default value is 1.
    
    Examples
    ===========
    >>> norm_pdf(1, 0, 1)
    0.24197072451914337
    
    >>> norm_pdf(12345, -20, 6789)
    1.1188585644690106e-05
    
    >>> norm_pdf(-123, 456, 789)
    0.000386273206473679
    
    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Normal_distribution)
    """
    
    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != float and type(x) != int:
        raise TypeError("Parameter 'x' is not a float or integer.")
    
    if type(mu) != float and type(mu) != int:
        raise TypeError("Parameter 'mu' is not a float or integer.")
    
    if type(sigma) != float and type(sigma) != int:
        raise TypeError("Parameter 'sigma' is not a float or integer.")
    elif sigma <= 0:
        raise ValueError("Parameter 'sigma' must be greater than 0.")
    
    # Engine
    #-------------------------------------------------------------------------#
    pdf = (1 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    
    # Output
    #-------------------------------------------------------------------------#
    return pdf

def norm_cdf(x, mu = 0, sigma = 1):
    r"""
    Retrieves the cumulative distribution function (CDF) of the Normal distribution.
    
    Function
    ===========
    .. math::
        CDF_{norm}(x, \mu, \sigma) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{\frac{-t^2}{2}} \;dt = \frac{1}{2} \left[ 1 + erf \left( \frac{x - \mu}{\sigma\sqrt{2}} \right) \right]
    
    Parameters
    ===========
    x : integer or float
    
    Upper limit of the integration.
    
    mu : integer or float
    
    Central location of the distribution. It is the average or expected value of the distribution. Default value is 0.
    
    sigma : integer or float
    
    Spread or dispersion of the distribution. It indicates how much the values of the distribution deviate from the mean. Default value is 1.
    
    Examples
    ===========
    >>> norm_cdf(1, 0, 1)
    0.8413447460685428
    
    >>> norm_cdf(12345, -20, 6789)
    0.9657215407724926
    
    >>> norm_cdf(-123, 456, 789)
    0.23152303661378465
    
    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Normal_distribution)
    """
    
    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != float and type(x) != int:
        raise TypeError("Parameter 'x' is not a float or integer.")
    
    if type(mu) != float and type(mu) != int:
        raise TypeError("Parameter 'mu' is not a float or integer.")
    
    if type(sigma) != float and type(sigma) != int:
        raise TypeError("Parameter 'sigma' is not a float or integer.")
    elif sigma <= 0:
        raise ValueError("Parameter 'sigma' must be greater than 0.")
    
    # Engine
    #-------------------------------------------------------------------------#
    z = (x - mu) / (sigma * math.sqrt(2))
    cdf = 0.5 * (1 + math.erf(z))
    
    # Output
    #-------------------------------------------------------------------------#
    return cdf

def norm_invcdf(p, mu = 0, sigma = 1):
    r"""
    Retrieves the inverse of the cumulative distribution function (CDF) of the Normal distribution.
    
    The algorithm returns an approximation of the inverse cumulative standard normal distribution function with a relative error whose absolute value is less than 1.15e-9.
    
    Function
    ===========
    .. math::
        CDF^{-1}_{norm}(p, \mu, \sigma) = \mu + \sigma \sqrt{2} erf^{-1}(2p - 1)
    
    Parameters
    ===========
    p : integer or float
    
    Cumulative probability value of a Normal distribution with mean = mu and standard deviation = sigma.
    
    mu : integer or float
    
    Central location of the distribution. It is the average or expected value of the distribution. Default value is 0.
    
    sigma : integer or float
    
    Spread or dispersion of the distribution. It indicates how much the values of the distribution deviate from the mean. Default value is 1.
    
    Examples
    ===========
    >>> norm_invcdf(0.02, 0, 1)
    -2.0537489090030348
    
    >>> norm_invcdf(0.12345, -20, 6789)
    -7881.063735536719
    
    >>> norm_invcdf(0.98, 456, 789)
    2076.407889203394
    
    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Normal_distribution)
    .. [2] An algorithm for computing the inverse normal cumulative distribution function (Author: Peter John Acklam / Website: http://home.online.no/~pjacklam)
    .. [3] Modification of Peter John Acklam's original perl code by Dan Field (3rd May 2004)
    """
    
    # Input Validation
    #-------------------------------------------------------------------------#
    if type(p) != float and type(p) != int:
        raise TypeError("Parameter 'p' is not a float or integer.")
    elif p <= 0 or p >= 1:
        raise ValueError("Parameter 'p' must be within the interval ]0; 1[.")
    
    if type(mu) != float and type(mu) != int:
        raise TypeError("Parameter 'mu' is not a float or integer.")
    
    if type(sigma) != float and type(sigma) != int:
        raise TypeError("Parameter 'sigma' is not a float or integer.")
    elif sigma <= 0:
        raise ValueError("Parameter 'sigma' must be greater than 0.")
        
    # Engine
    #-------------------------------------------------------------------------#
    # Coefficients in rational approximations
    a = (-3.969683028665376e+01, 2.209460984245205e+02, \
    -2.759285104469687e+02, 1.383577518672690e+02, \
    -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, \
    -1.556989798598866e+02, 6.680131188771972e+01, \
    -1.328068155288572e+01 )
    c = (-7.784894002430293e-03, -3.223964580411365e-01, \
    -2.400758277161838e+00, -2.549732539343734e+00, \
    4.374664141464968e+00, 2.938163982698783e+00)
    d = ( 7.784695709041462e-03, 3.224671290700398e-01, \
    2.445134137142996e+00, 3.754408661907416e+00)
    
    # Define break-points
    plow = 0.02425
    phigh = 1 - plow
    
    # Rational approximation for lower region:
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        invcdf = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    
    # Rational approximation for central region:
    elif plow <= p <= phigh:
        q = p - 0.5
        r=q*q
        invcdf = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                 (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    
    # Rational approximation for upper region:
    else:
        q = math.sqrt(-2*math.log(1-p))
        invcdf = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    
    # Outcome
    #-------------------------------------------------------------------------#
    return mu + sigma * invcdf

def norm_rvs(mu = 0, sigma = 1, size = 1, seed = None):
    r"""
    Generate a random number or a list of random numbers based on the Normal distribution.
    
    Function
    ===========
    .. math::
        RVS_{norm}(\mu, \sigma) \sim N(\mu, \sigma^2)
    
    Parameters
    ===========
    mu : integer or float
    
    Central location of the distribution. It is the average or expected value of the distribution. Default value is 0.
    
    sigma : integer or float
    
    Spread or dispersion of the distribution. It indicates how much the values of the distribution deviate from the mean. Default value is 1.
    
    size : integer
    
    Number of random variates. If size = 1, the output is a float; otherwise, it is a list.
    
    seed : integer
    
    The seed determines the sequence of random numbers generated (i.e., the same seed will generate the exact same random number or list of random numbers).
    
    Examples
    ===========
    >>> norm_rvs(0, 1, 1, 12345)
    1.320614489253298
    
    >>> norm_rvs(0, 1, 5, 12345)
    [1.320614489253298,
    -0.18650633919367346,
    0.48986790430628746,
    0.5620930205892353,
    -1.863579311650145]
    
    >>> norm_rvs(-100, 1000, 1, 6789)
    1136.1983515770335
    
    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Normal_distribution)
    .. [2] Wikipedia (https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(mu) != float and type(mu) != int:
        raise TypeError("Parameter 'mu' is not a float or integer.")
    
    if type(sigma) != float and type(sigma) != int:
        raise TypeError("Parameter 'sigma' is not a float or integer.")
    elif sigma <= 0:
        raise ValueError("Parameter 'sigma' must be greater than 0.")
    
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
    
    # Generator (Box-Muller Transform)
    def rvs_gen(mu, sigma):
        u1 = random.uniform(0, 1)
        u2 = random.uniform(0, 1)
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return mu + sigma * z
    
    # Generate the random variates (rvs)
    if size == 1:
        rvs = rvs_gen(mu = mu, sigma = sigma)
    else:
        rvs = [rvs_gen(mu = mu, sigma = sigma) for _ in range(size)]
    
    # Outcome
    #-------------------------------------------------------------------------#
    return rvs
