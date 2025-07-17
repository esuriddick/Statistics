#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import math
import random

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def binom_pmf(x, n, p):
    r"""
    Retrieves the probability mass function (PMF) of the Binomial distribution.

    Function
    ===========
    .. math::
        PMF_{B}(x, n, p) = \frac{n!}{x!(n - x)!} p^x (1 - p)^{n - x}

    Parameters
    ===========
    x : integer

    Value at which the probability mass function is evaluated.

    n : integer

    Number of independent experiments or trials.

    p : integer or float

    Probability that a single trial results in a success.

    Examples
    ===========
    >>> binom_pmf(3, 20, 0.1)
    0.190119871376199

    >>> binom_pmf(0, 20, 0)
    1

    >>> binom_pmf(3, 5, 0.5)
    0.3125

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Binomial_distribution)
    """
   
    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != int:
        raise TypeError("Parameter 'x' is not an integer.")
    elif x < 0:
        raise ValueError("Parameter 'x' must be greater or equal to 0.")
   
    if type(n) != int:
        raise TypeError("Parameter 'n' is not an integer.")
    elif n < 0:
        raise ValueError("Parameter 'n' must be greater or equal to 0.")
   
    if type(p) != float and type(p) != int:
        raise TypeError("Parameter 'p' is not a float or integer.")
    elif p < 0 or p > 1:
        raise ValueError("Parameter 'p' must be within the interval [0; 1].")
       
    # Engine
    #-------------------------------------------------------------------------#
    num = math.factorial(n) * (p ** (x)) * ((1 - p) ** (n - x))
    den = math.factorial(x) * math.factorial(n - x)
    pmf = num / den
   
    # Output
    #-------------------------------------------------------------------------#
    return pmf

def binom_cdf(x, n, p):
    r"""
    Retrieves the cumulative distribution function (CDF) of the Binomial distribution.

    Function
    ===========
    .. math::
        CDF_{B}(x, n, p) = \Sigma^x_{i = 0} \left[ \frac{n!}{i!(n - i)!} p^i (1 - p)^{n - i} \right]

    Parameters
    ===========
    x : integer

    Value up to which the cumulative probability is calculated.

    n : integer

    Number of independent experiments or trials.

    p : integer or float

    Probability that a single trial results in a success.

    Examples
    ===========
    >>> binom_cdf(3, 20, 0.1)
    0.8670466765656654

    >>> binom_cdf(1, 20, 0)
    1.0

    >>> binom_cdf(3, 5, 0.5)
    0.8125

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Binomial_distribution)
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != int:
        raise TypeError("Parameter 'x' is not an integer.")
    elif x < 0:
        raise ValueError("Parameter 'x' must be greater or equal to 0.")
   
    if type(n) != int:
        raise TypeError("Parameter 'n' is not an integer.")
    elif n < 0:
        raise ValueError("Parameter 'n' must be greater or equal to 0.")
   
    if type(p) != float and type(p) != int:
        raise TypeError("Parameter 'p' is not a float or integer.")
    elif p < 0 or p > 1:
        raise ValueError("Parameter 'p' must be within the interval [0; 1].")
       
    # Engine
    #-------------------------------------------------------------------------#
    cdf = sum((math.factorial(n) * (p ** (i)) * ((1 - p) ** (n - i))) / \
    (math.factorial(i) * math.factorial(n - i)) for i in range(x + 1))
   
    # Output
    #-------------------------------------------------------------------------#
    return cdf
   
def binom_invcdf(q, n, p):
    r"""
    Retrieves the inverse of the cumulative distribution function (CDF) of the Binomial distribution.

    Function
    ===========
    .. math::
        CDF^{-1}_{B}(q, n, p) = x

    Parameters
    ===========
    q : integer or float
   
    Cumulative probability value of a Binomial distribution with n trials and probability of success equal to p per trial.

    n : integer

    Number of independent experiments or trials.

    p : integer or float

    Probability that a single trial results in a success.

    Examples
    ===========
    >>> binom_invcdf(0.86704667656566, 20, 0.1)
    3

    >>> binom_invcdf(0.75, 20, 0.75)
    16

    >>> binom_invcdf(0.8125, 5, 0.5)
    3

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Binomial_distribution)
    .. [2] Wikipedia (https://en.wikipedia.org/wiki/Binary_search)
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(q) != float and type(q) != int:
        raise TypeError("Parameter 'q' is not a float or integer.")
    elif q < 0 or q > 1:
        raise ValueError("Parameter 'q' must be within the interval [0; 1].")
   
    if type(n) != int:
        raise TypeError("Parameter 'n' is not an integer.")
    elif n < 0:
        raise ValueError("Parameter 'n' must be greater than 0.")
   
    if type(p) != float and type(p) != int:
        raise TypeError("Parameter 'p' is not a float or integer.")
    elif p <= 0 or p >= 1:
        raise ValueError("Parameter 'p' must be within the interval ]0; 1[.")
       
    # Engine (Binary Search)
    #-------------------------------------------------------------------------#
    left = 0
    right = n
    while left < right:
        mid = (left + right) // 2
        cdf = binom_cdf(mid, n, p)
        if cdf < q:
            left = mid + 1
        else:
            right = mid
    invcdf = left
   
    # Output
    #-------------------------------------------------------------------------#
    return invcdf

def binom_rvs(n, p, size = 1, seed = None):
    r"""
    Generate the total number of successes or a list of total number of successes (n trials repeated size times) based on the Binomial distribution.
   
    Function
    ===========
    .. math::
        RVS_{B}(n, p) \sim B(n, p)
   
    Parameters
    ===========
    n : integer

    Number of independent experiments or trials.

    p : integer or float

    Probability that a single trial results in a success.
   
    size : integer

    Number of random variates. If size = 1, the output is a float; otherwise, it is a list.

    seed : integer

    The seed determines the sequence of random numbers generated (i.e., the same seed will generate the exact same random number or list of random numbers).
   
    Examples
    ===========
    >>> binom_rvs(20, 0.1, 1, 12345)
    1
   
    >>> binom_rvs(20, 0.1, 5, 12345)
    [1, 2, 1, 1, 2]
   
    >>> binom_rvs(5, 0.5, 1, 6789)
    1
   
    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Binomial_distribution)
    """
   
    # Input Validation
    #-------------------------------------------------------------------------#
    if type(n) != int:
        raise TypeError("Parameter 'n' is not an integer.")
    elif n <= 0:
        raise ValueError("Parameter 'n' must be greater than 0.")
   
    if type(p) != float and type(p) != int:
        raise TypeError("Parameter 'p' is not a float or integer.")
    elif p <= 0 or p >= 1:
        raise ValueError("Parameter 'p' must be within the interval ]0; 1[.")
       
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
        rvs = random.binomialvariate(n, p)
    else:
        rvs = [random.binomialvariate(n, p) for _ in range(size)]
   
    # Output
    #-------------------------------------------------------------------------#
    return rvs
