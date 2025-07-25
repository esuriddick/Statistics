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
    binom_pmf
    ===========
    Compute the probability mass function (PMF) of the Binomial distribution.

    The Binomial PMF gives the probability of observing exactly :math:`x` successes in
    :math:`n` independent Bernoulli trials, each with success probability :math:`p`.
    
    Mathematical Definition
    ----------
    .. math::
        \text{PMF}_B(x; n, p) = \binom{n}{x} p^x (1 - p)^{n - x}
    
    Where:
        
    - :math:`\binom{n}{x}` is the binomial coefficient: :math:`\frac{n!}{x!(n - x)!}`
    - :math:`x` is the number of successes
    - :math:`n` is the number of trials
    - :math:`p` is the probability of success in each trial
    
    Parameters
    ----------
    x : int
        The number of successes. Must be in the range :math:`[0, n]`.
        
    n : int
        The total number of independent trials (must be non-negative).
        
    p : float or int
        Probability of success in a single trial. Must be in the interval :math:`[0, 1]`.
    
    Returns
    ----------
    float
        The probability of observing exactly :math:`x` successes in :math:`n` trials.
    
    Examples
    ----------
    >>> binom_pmf(3, 20, 0.1)
    0.190119871376199

    >>> binom_pmf(0, 20, 0)
    1

    >>> binom_pmf(3, 5, 0.5)
    0.3125
    
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Binomial_distribution
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
    binom_cdf
    ===========
    Compute the cumulative distribution function (CDF) of the Binomial distribution.

    The CDF gives the probability of obtaining at most :math:`x` successes in :math:`n` independent
    Bernoulli trials, each with success probability :math:`p`. Mathematically, the CDF is defined as:

    Mathematical Definition
    ----------
    .. math::
        \text{CDF}_{B}(x; n, p) = \sum_{i=0}^{x} \binom{n}{i} p^i (1 - p)^{n - i}

    Parameters
    ----------
    x : int
        The number of successes (upper bound for the cumulative sum). Must satisfy :math:`0 \leq x \leq n`.

    n : int
        The total number of independent trials. Must be non-negative.

    p : float or int
        Probability of success on a single trial. Must be in the interval :math:`[0, 1]`.

    Returns
    -------
    float
        The cumulative probability of getting at most :math:`x` successes in :math:`n` trials.

    Examples
    --------
    >>> binom_cdf(3, 20, 0.1)
    0.8670466765656654

    >>> binom_cdf(1, 20, 0)
    1.0

    >>> binom_cdf(3, 5, 0.5)
    0.8125

    References
    ----------
    https://en.wikipedia.org/wiki/Binomial_distribution
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
    binom_invcdf
    ===========
    Compute the inverse of the cumulative distribution function (CDF) for the Binomial distribution.

    This function returns the smallest integer :math:`x` such that the cumulative probability
    :math:`P(X ≤ x)` is greater than or equal to the given probability :math:`q`, for a Binomial
    distribution with :math:`n` trials and success probability :math:`p`.

    Mathematical Definition
    ----------
    .. math::
        CDF^{-1}_{B}(q, n, p) = \min \{x \in \mathbb{N}_0 : P(X \leq x) \geq q\}
    
    where :math:`X \sim \text{Binomial}(n, p)`

    Parameters
    ----------
    q : float or int
        The cumulative probability (0 ≤ q ≤ 1) for which to compute the inverse CDF.

    n : int
        The number of independent Bernoulli trials :math:`(n \geq 0)`.

    p : float
        The probability of success in each trial :math:`(0 < p < 1)`.

    Returns
    ----------
    int
        The smallest integer :math:`x` such that the CDF of the Binomial distribution at :math:`x`
        is at least :math:`q`.

    Examples
    ----------
    >>> binom_invcdf(0.86704667656566, 20, 0.1)
    3

    >>> binom_invcdf(0.75, 20, 0.75)
    16

    >>> binom_invcdf(0.8125, 5, 0.5)
    3

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Binomial_distribution
    .. [2] https://en.wikipedia.org/wiki/Binary_search
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
        raise ValueError("Parameter 'n' must be greater or equal to 0.")
   
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
    binom_rvs
    ===========
    Generate random variates representing the number of successes from a Binomial distribution.

    This function simulates `size` independent experiments, each consisting of :math:`n` Bernoulli trials
    with     success probability :math:`p`. The output is either a single value or a list of values
    representing the total number of successes in each experiment.
   
    Mathematical Definition
    ----------
    .. math::
        X \sim \text{Binomial}(n, p)
        
    Where:
        
    - :math:`n` is the number of trials per experiment
    - :math:`p` is the probability of success in a single trial
   
    Parameters
    ----------
    n : int
        Number of independent Bernoulli trials in each experiment (must be non-negative).

    p : float
        Probability of success in each trial :math:`(0 < p < 1)`.

    size : int, optional, default=1
        Number of experiments to simulate. If `size=1`, returns a single integer; otherwise,
        returns a list of integers of length `size`.

    seed : int, optional, default=None
        Seed for the random number generator. Setting the seed ensures reproducibility.
    
    Returns
    ----------
    int or list of int
        Total number of successes from the Binomial distribution. Returns an integer if `size=1`,
        otherwise returns a list of integers.
    
    Examples
    ----------
    >>> binom_rvs(20, 0.1, 1, 12345)
    1
   
    >>> binom_rvs(20, 0.1, 5, 12345)
    [1, 2, 1, 1, 2]
   
    >>> binom_rvs(5, 0.5, 1, 6789)
    1
   
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Binomial_distribution
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
