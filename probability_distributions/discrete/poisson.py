#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import math
import random

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def poisson_pmf(x, lamb):
    r"""
    Retrieves the probability mass function (PMF) of the Poisson distribution.

    Function
    ===========
    .. math::
        PMF_{Pois}(x, \lambda) = \frac{\lambda^x e^{-\lambda}}{x!}

    Parameters
    ===========
    x : integer

    Value at which the probability mass function is evaluated.

    lamb: integer or float

    The average rate of events occurring in the given time interval. It is also equal to the variance in the Poisson distribution.

    Examples
    ===========
    >>> poisson_pmf(1, 5)
    0.03368973499542734

    >>> poisson_pmf(5, 1)
    0.00306566200976202

    >>> poisson_pmf(5, 5)
    0.17546736976785068

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Poisson_distribution)
    """
   
    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != int:
        raise TypeError("Parameter 'x' is not an integer.")
   
    if type(lamb) != float and type(lamb) != int:
        raise TypeError("Parameter 'lamb' is not a float or integer.")
    elif lamb < 0:
        raise ValueError("Parameter 'lamb' must be greater or equal to 0.")
       
    # Engine
    #-------------------------------------------------------------------------#
    if x < 0 or lamb == 0:
        pmf = 0
    else:
        num = math.log(lamb ** x) + (-lamb)
        den = math.log(math.factorial(x))
        pmf = math.exp(num - den)
   
    # Output
    #-------------------------------------------------------------------------#
    return pmf

def poisson_cdf(x, lamb):
    r"""
    Retrieves the cumulative distribution function (CDF) of the Poisson distribution.

    Function
    ===========
    .. math::
        CDF_{Pois}(x, \lambda) = e^{-\lambda} \Sigma_{i = 0}^x \frac{\lambda^i}{i!}

    Parameters
    ===========
    x : integer

    Value up to which the cumulative probability is calculated.

    lamb: integer or float

    The average rate of events occurring in the given time interval. It is also equal to the variance in the Poisson distribution.

    Examples
    ===========
    >>> poisson_cdf(1, 5)
    0.04042768199451281

    >>> poisson_cdf(5, 1)
    0.9994058151824183

    >>> poisson_cdf(5, 5)
    0.6159606548330633

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Poisson_distribution)
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != int:
        raise TypeError("Parameter 'x' is not an integer.")
   
    if type(lamb) != float and type(lamb) != int:
        raise TypeError("Parameter 'lamb' is not a float or integer.")
    elif lamb < 0:
        raise ValueError("Parameter 'lamb' must be greater or equal to 0.")
       
    # Engine
    #-------------------------------------------------------------------------#
    if x < 0 or lamb == 0:
        cdf = 0
    else:
        sigma = sum(lamb ** i / math.factorial(i) for i in range(x + 1))
        mult = math.e ** (-lamb)
        cdf = mult * sigma
   
    # Output
    #-------------------------------------------------------------------------#
    return cdf

def poisson_invcdf(p, lamb):
    r"""
    Retrieves the inverse of the cumulative distribution function (CDF) of the Poisson distribution.

    Function
    ===========
    .. math::
        CDF^{-1}_{Pois}(p, \lambda) = x

    Parameters
    ===========
    p : integer or float
   
    Cumulative probability value of a Poisson distribution with an average rate of events lamb in the given time interval.
    
    lamb: integer or float

    The average rate of events occurring in the given time interval. It is also equal to the variance in the Poisson distribution.

    Examples
    ===========
    >>> poisson_invcdf(0.04042768199451281, 5)
    1

    >>> poisson_invcdf(0.9994058151824183, 1)
    5

    >>> poisson_invcdf(0.6159606548330633, 5)
    5

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Poisson_distribution)
    .. [2] Wikipedia (https://en.wikipedia.org/wiki/Binary_search)
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(p) != float and type(p) != int:
        raise TypeError("Parameter 'p' is not a float or integer.")
    elif p < 0 or p > 1:
        raise ValueError("Parameter 'p' must be within the interval [0; 1].")
   
    if type(lamb) != float and type(lamb) != int:
        raise TypeError("Parameter 'lamb' is not a float or integer.")
    elif lamb < 0:
        raise ValueError("Parameter 'lamb' must be greater or equal to 0.")
       
    # Engine (Binary Search)
    #-------------------------------------------------------------------------#
    if lamb == 0:
        invcdf = 0
    else:
        left = 0
        right = int(lamb + 10 * math.sqrt(lamb) + 10)
        while left < right:
            mid = (left + right) // 2
            cdf = poisson_cdf(mid, lamb)
            if cdf < p:
                left = mid + 1
            else:
                right = mid
        invcdf = left
   
    # Output
    #-------------------------------------------------------------------------#
    return invcdf

def poisson_rvs(lamb, size = 1, seed = None):
    r"""
    Generate the total number of events happening in a given time period based on the Poisson distribution.
   
    Function
    ===========
    .. math::
        RVS_{Pois}(\lambda, p) \sim Pois(\lambda)
   
    Parameters
    ===========
    lamb: integer or float

    The average rate of events occurring in the given time interval. It is also equal to the variance in the Poisson distribution.
   
    size : integer

    Number of random variates. If size = 1, the output is a float; otherwise, it is a list.

    seed : integer

    The seed determines the sequence of random numbers generated (i.e., the same seed will generate the exact same random number or list of random numbers).
   
    Examples
    ===========
    >>> poisson_rvs(1, 1, 12345)
    1
   
    >>> poisson_rvs(2, 5, 12345)
    [1, 2, 1, 1, 2]
   
    >>> poisson_rvs(3, 1, 6789)
    1
   
    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Poisson_distribution)
    """
   
    # Input Validation
    #-------------------------------------------------------------------------#
    if type(lamb) != float and type(lamb) != int:
        raise TypeError("Parameter 'lamb' is not a float or integer.")
    elif lamb < 0:
        raise ValueError("Parameter 'lamb' must be greater or equal to 0.")
       
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
        
    # Generator (Knuth’s Algorithm)
    def rvs_gen(lamb):
        L = math.exp(-lamb)
        k = 0
        p = 1
        while p > L:
            k += 1
            p *= random.random()
        return k - 1
       
    # Generate the random variates (rvs)
    if size == 1:
        rvs = rvs_gen(lamb)
    else:
        rvs = [rvs_gen(lamb) for _ in range(size)]
   
    # Output
    #-------------------------------------------------------------------------#
    return rvs
