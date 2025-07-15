#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import random

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def cont_uniform_pdf(x, a = 0, b = 1, include_a = True, include_b = True):
    r"""
    Retrieves the probability density function (PDF) of the continuous Uniform distribution.

    Function
    ===========
    .. math::
        PDF_{U}(x, a, b) = \frac{1}{b - a}, \;where \;a ≤ x ≤ b

    If x is lower than a or greater than b, then:

    .. math::
        PDF_{U}(x, a, b) = 0

    It should be noted that if parameter include_a is equal to False, then:

    .. math::
        PDF_{U}(a, a, b) = 0

    Similarly, if parameter include_b is equal to False, then:

    .. math::
        PDF_{U}(b, a, b) = 0

    Parameters
    ===========
    x : integer or float

    Value at which the probability density function is evaluated.

    a : integer or float

    Lowest possible value to obtain in the distribution (if include_a = True).

    b : integer or float

    Highest possible value to obtain in the distribution (if include_b = True).

    include_a : bool

    Determines whether the value a can be obtained from the distribution.

    include_b : bool

    Determines whether the value b can be obtained from the distribution.

    Examples
    ===========
    >>> cont_uniform_pdf(2.5, 0.6, 12.2)
    0.08620689655172414

    >>> cont_uniform_pdf(-1, 0, 1)
    0

    >>> cont_uniform_pdf(0.5, 0, 1)
    1

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
    """
   
    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != float and type(x) != int:
        raise TypeError("Parameter 'x' is not a float or integer.")
   
    if type(a) != float and type(a) != int:
        raise TypeError("Parameter 'a' is not a float or integer.")
   
    if type(b) != float and type(b) != int:
        raise TypeError("Parameter 'b' is not a float or integer.")
    elif b < a:
        raise ValueError("Parameter 'b' cannot be smaller than parameter 'a'.")
    elif b == a:
        raise ValueError("Parameter 'b' cannot be equal to parameter 'a'.")
       
    if type(include_a) != bool:
        raise TypeError("Parameter 'include_a' is not a bool.")
       
    if type(include_b) != bool:
        raise TypeError("Parameter 'include_b' is not a bool.")
       
    # Engine
    #-------------------------------------------------------------------------#
    if x < a or x > b:
        pdf = 0
    elif x == a and include_a == False:
        pdf = 0
    elif x == b and include_b == False:
        pdf = 0
    else:
        pdf = 1 / (b - a)
   
    # Output
    #-------------------------------------------------------------------------#
    return pdf

def cont_uniform_cdf(x, a = 0, b = 1):
    r"""
    Retrieves the cumulative distribution function (CDF) of the continuous Uniform distribution.

    Function
    ===========
    .. math::
        CDF_{U}(x, a, b) = \frac{x - a}{b - a}, \;where \;a ≤ x ≤ b

    If x is lower than a, then:

    .. math::
        CDF_{U}(x, a, b) = 0
       
    If x is greater than b, then:
       
    .. math::
        CDF_{U}(x, a, b) = 1

    Parameters
    ===========
    x : integer or float

    Value up to which the cumulative probability is calculated.

    a : integer or float

    Lowest possible value to obtain in the distribution.

    b : integer or float

    Highest possible value to obtain in the distribution.

    Examples
    ===========
    >>> cont_uniform_cdf(2.5, 0.6, 12.2)
    0.16379310344827586

    >>> cont_uniform_cdf(-1, 0, 1)
    0

    >>> cont_uniform_cdf(0.5, 0, 1)
    0.5

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != float and type(x) != int:
        raise TypeError("Parameter 'x' is not a float or integer.")
   
    if type(a) != float and type(a) != int:
        raise TypeError("Parameter 'a' is not a float or integer.")
   
    if type(b) != float and type(b) != int:
        raise TypeError("Parameter 'b' is not a float or integer.")
    elif b < a:
        raise ValueError("Parameter 'b' cannot be smaller than parameter 'a'.")
    elif b == a:
        raise ValueError("Parameter 'b' cannot be equal to parameter 'a'.")
       
    # Engine
    #-------------------------------------------------------------------------#
    if x <= a:
        cdf = 0
    elif x >= b:
        cdf = 0
    else:
        cdf = (x - a) / (b - a)
   
    # Output
    #-------------------------------------------------------------------------#
    return cdf
   
def cont_uniform_invcdf(p, a = 0, b = 1):
    r"""
    Retrieves the inverse of the cumulative distribution function (CDF) of the continuous Uniform distribution.

    Function
    ===========
    .. math::
        CDF^{-1}_{U}(p, a, b) = p \times (b - a) + a, \;where \;a ≤ x ≤ b

    If p is equal to 0, then:

    .. math::
        CDF^{-1}_{U}(0, a, b) = a
       
    If p is equal to 1, then:
       
    .. math::
        CDF^{-1}_{U}(1, a, b) = b

    Parameters
    ===========
    p : integer or float
   
    Cumulative probability value of a Uniform distribution with the smallest value equal to a and the largest value equal to b.

    a : integer or float

    Lowest possible value to obtain in the distribution.

    b : integer or float

    Highest possible value to obtain in the distribution.

    Examples
    ===========
    >>> cont_uniform_invcdf(0.16379310344827586, 0.6, 12.2)
    2.5

    >>> cont_uniform_invcdf(0, 0, 1)
    0

    >>> cont_uniform_invcdf(0.5, 0, 1)
    0.5

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(p) != float and type(p) != int:
        raise TypeError("Parameter 'p' is not a float or integer.")
    elif p < 0 or p > 1:
        raise ValueError("Parameter 'p' must be within the interval [0; 1].")
   
    if type(a) != float and type(a) != int:
        raise TypeError("Parameter 'a' is not a float or integer.")
   
    if type(b) != float and type(b) != int:
        raise TypeError("Parameter 'b' is not a float or integer.")
    elif b < a:
        raise ValueError("Parameter 'b' cannot be smaller than parameter 'a'.")
    elif b == a:
        raise ValueError("Parameter 'b' cannot be equal to parameter 'a'.")
       
    # Engine
    #-------------------------------------------------------------------------#
    if p == 0:
        invcdf = a
    elif p == 1:
        invcdf = b
    else:
        invcdf = p * (b - a) + a
   
    # Output
    #-------------------------------------------------------------------------#
    return invcdf

def cont_uniform_rvs(a = 0, b = 1, size = 1, seed = None):
    r"""
    Generate a random number (x) or a list of random numbers based on the continuous Uniform distribution, where a ≤ x ≤ b.
   
    Function
    ===========
    .. math::
        RVS_{U}(a, b) \sim U(a, b)
   
    Parameters
    ===========
    a : integer or float

    Lowest possible value to obtain in the distribution.

    b : integer or float

    Highest possible value to obtain in the distribution.
   
    size : integer

    Number of random variates. If size = 1, the output is a float; otherwise, it is a list.

    seed : integer

    The seed determines the sequence of random numbers generated (i.e., the same seed will generate the exact same random number or list of random numbers).
   
    Examples
    ===========
    >>> cont_uniform_rvs(0, 1, 1, 12345)
    0.41661987254534116
   
    >>> cont_uniform_rvs(0, 1, 5, 12345)
    [0.41661987254534116,
     0.010169169457068361,
     0.8252065092537432,
     0.2986398551995928,
     0.3684116894884757]
   
    >>> cont_u_rvs(-100, 1000, 1, 6789)
    49.32491021823233
   
    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
    """
   
    # Input Validation
    #-------------------------------------------------------------------------#
    if type(a) != float and type(a) != int:
        raise TypeError("Parameter 'a' is not a float or integer.")
   
    if type(b) != float and type(b) != int:
        raise TypeError("Parameter 'b' is not a float or integer.")
    elif b < a:
        raise ValueError("Parameter 'b' cannot be smaller than parameter 'a'.")
    elif b == a:
        raise ValueError("Parameter 'b' cannot be equal to parameter 'a'.")
       
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
        rvs = random.uniform(a, b)
    else:
        rvs = [random.uniform(a, b) for _ in range(size)]
   
    # Output
    #-------------------------------------------------------------------------#
    return rvs
