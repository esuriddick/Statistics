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
    poisson_pmf
    ===========
    Computes the probability mass function (PMF) of the Poisson distribution.

    The Poisson distribution expresses the probability of a given number of events :math:`x` 
    occurring in a fixed interval of time or space, given a known constant average rate :math:`\lambda`
    of occurrence and independence between events.

    Mathematical Definition
    ----------
    The PMF of the Poisson distribution is defined as:

    .. math::
        \text{PMF}_{\text{Pois}}(x; \lambda) = \frac{\lambda^x e^{-\lambda}}{x!}

    where:
        
    - :math:`x` is the number of occurrences (a non-negative integer),
    - :math:`\lambda` is the expected number of occurrences (rate parameter),
    - :math:`e` is Euler’s number (approximately 2.71828).

    Parameters
    ----------
    x : int
        The non-negative integer value at which the PMF is evaluated (number of occurrences).

    lamb : float or int
        The rate parameter :math:`\lambda`, representing the average number of events per interval.
        Must be a positive real number.

    Returns
    ----------
    float
        The value of the Poisson PMF evaluated at :math:`x` for the given :math:`\lambda`.

    Examples
    ----------
    >>> poisson_pmf(1, 5)
    0.03368973499542734

    >>> poisson_pmf(5, 1)
    0.00306566200976202

    >>> poisson_pmf(5, 5)
    0.17546736976785068

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Poisson_distribution
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
    poisson_cdf
    ===========
    Computes the cumulative distribution function (CDF) of the Poisson distribution 
    for a given non-negative integer value :math:`x` and rate parameter :math:`\lambda`.

    The Poisson CDF gives the probability of observing up to and including :math:`x` 
    events given a Poisson distribution with mean rate :math:`\lambda`.

    Mathematical Definition
    ----------
    .. math::
        \text{CDF}_{\text{Poisson}}(x; \lambda) = e^{-\lambda} \sum_{i=0}^{x} \frac{\lambda^i}{i!}

    Parameters
    ----------
    x : int
        The non-negative integer value up to which the cumulative probability 
        is computed. Represents the number of events.

    lamb : float or int
        The rate parameter :math:`(\lambda)` of the Poisson distribution. Must be non-negative.
        It represents the expected number of occurrences in a fixed interval and
        is equal to both the mean and the variance of the distribution.
        
    Returns
    -------
    float
        The cumulative probability of observing up to :math:`x` events in a Poisson 
        distribution with rate :math:`\lambda`.

    Examples
    ----------
    >>> poisson_cdf(1, 5)
    0.04042768199451281

    >>> poisson_cdf(5, 1)
    0.9994058151824183

    >>> poisson_cdf(5, 5)
    0.6159606548330633

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Poisson_distribution
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
    poisson_invcdf
    ===========
    Computes the inverse cumulative distribution function (quantile function) for the Poisson distribution.

    This function returns the smallest integer :math:`x` such that the cumulative probability
    :math:`P(X ≤ x)` is greater than or equal to the given probability :math:`p`, for a Poisson
    distribution with rate parameter :math:`\lambda`.
    
    .. math::
        P(X \leq x) \geq p

    where :math:`X \sim \text{Poisson}(\lambda)`.

    Mathematical Definition
    ----------
    .. math::
        CDF^{-1}_{\text{Poisson}}(p, \lambda) = \min \{x \in \mathbb{N}_0 : P(X \leq x) \geq p\}
    
    where :math:`X \sim \text{Poisson}(\lambda)`.

    Parameters
    ----------
    p : float or int
        Cumulative probability :math:`(0 \leq p \leq 1)` at which to evaluate the inverse CDF.

    lamb : float or int
        The rate parameter :math:`(\lambda)` of the Poisson distribution.
        Represents the expected number of occurrences in a fixed interval.

    Returns
    -------
    x : int
        The smallest integer such that the Poisson CDF at :math:`x` is greater than or equal to :math:`p`.

    Examples
    ----------
    >>> poisson_invcdf(0.04042768199451281, 5)
    1

    >>> poisson_invcdf(0.9994058151824183, 1)
    5

    >>> poisson_invcdf(0.6159606548330633, 5)
    5

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Poisson_distribution
    .. [2] https://en.wikipedia.org/wiki/Binary_search
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
    poisson_rvs
    ===========
    Generate random variates representing the number of events occurring in a fixed time interval,
    based on the Poisson distribution.

    The Poisson distribution models the number of events occurring in a fixed interval of time or space,
    assuming these events happen with a known constant rate and independently of the time since the last event.
   
    Mathematical Definition
    ----------
    .. math::
        X \sim \text{Poisson}(\lambda)
   
    Parameters
    ----------
    lamb : int or float
        The expected number of events :math:`λ` in the given time interval. This is also the variance of
        the Poisson distribution.

    size : int, optional, default=1
        Number of random variates to generate. If `size=1`, a single integer is returned.
        If `size > 1`, a list of integers of length `size` is returned.

    seed : int, optional, default=None
        Random seed for reproducibility. Using the same seed will generate the same output.
    
    Returns
    -------
    int or list of int
        A single Poisson-distributed random variate if `size=1`, otherwise a list of variates.
    
    Examples
    ----------
    >>> poisson_rvs(1, 1, 12345)
    1
   
    >>> poisson_rvs(2, 5, 12345)
    [1, 2, 1, 1, 2]
   
    >>> poisson_rvs(3, 1, 6789)
    1
   
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Poisson_distribution
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
