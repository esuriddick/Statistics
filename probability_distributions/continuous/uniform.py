#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import random

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def cont_uniform_pdf(x, a = 0, b = 1, include_a = True, include_b = True):
    r"""
    cont_uniform_pdf
    ===========
    Computes the probability density function (PDF) of the continuous Uniform distribution 
    at a specified value :math:`x`, given lower and upper bounds :math:`a` and :math:`b`.

    Mathematical Definition
    ----------
    .. math::
        \text{PDF}_U(x; a, b) = 
        \begin{cases}
            \frac{1}{b - a}, & \text{if } a < x < b \\
            \frac{1}{b - a}, & \text{if } x = a \text{ and include\_a=True} \\
            \frac{1}{b - a}, & \text{if } x = b \text{ and include\_b=True} \\
            0, & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    x : float or int
        The value at which to evaluate the PDF.
    
    a : float or int, optional, default=0
        The lower bound of the uniform distribution.
    
    b : float or int, optional, default=1
        The upper bound of the uniform distribution.
    
    include_a : bool, optional, default=True
        If True, the value `a` is considered within the distribution (i.e., PDF is non-zero at :math:`x = a`).
        If False, :math:`x = a` yields a PDF of 0.
    
    include_b : bool, optional, default=True
        If True, the value `b` is considered within the support (i.e., PDF is non-zero at :math:`x = b`).
        If False, :math:`x = b` yields a PDF of 0.
    
    Returns
    ----------
    float
        The probability density at :math:`x`. Returns :math:`0` if :math:`x` is outside the defined distribution, 
        accounting for the `include_a` and `include_b` parameters.
    
    Examples
    ----------
    >>> cont_uniform_pdf(2.5, 0.6, 12.2)
    0.08620689655172414

    >>> cont_uniform_pdf(-1, 0, 1)
    0

    >>> cont_uniform_pdf(0.5, 0, 1)
    1

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Continuous_uniform_distribution
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
    cont_uniform_cdf
    ===========
    Computes the cumulative distribution function (CDF) of the continuous uniform distribution 
    for a given value :math:`x` over the interval :math:`[a, b]`.

    Mathematical Definition
    ----------
    .. math::
        \text{CDF}_{U}(x; a, b) = \begin{cases}
            0 & \text{if } x < a \\
            \frac{x - a}{b - a} & \text{if } a \leq x \leq b \\
            1 & \text{if } x > b
        \end{cases}

    Parameters
    ----------
    x : float or int
        The value up to which the cumulative probability is evaluated.
    
    a : float or int, optional, default=0
        The lower bound of the uniform distribution interval.
    
    b : float or int, optional, default=1
        The upper bound of the uniform distribution interval. Must satisfy `a < b`.
    
    Returns
    ----------
    float
        The value of the CDF at point :math:`x`, which will be between 0 and 1, inclusive.
    
    Examples
    ----------
    >>> cont_uniform_cdf(2.5, 0.6, 12.2)
    0.16379310344827586

    >>> cont_uniform_cdf(-1, 0, 1)
    0

    >>> cont_uniform_cdf(0.5, 0, 1)
    0.5

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Continuous_uniform_distribution
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
    cont_uniform_invcdf
    ===========
    Computes the inverse cumulative distribution function (inverse CDF or quantile function) 
    for the continuous Uniform distribution over the interval :math:`[a, b]`.

    Mathematical Definition
    ----------
    .. math::
        F^{-1}(p; a, b) = p \cdot (b - a) + a,\quad \text{for } 0 \leq p \leq 1

    Special cases:
        
    - When :math:`p = 0`, the result is :math:`a`
    - When :math:`p = 1`, the result is :math:`b`

    Parameters
    ----------
    p : float or int
        Cumulative probability (must be in the range :math:`[0, 1]`).

    a : float or int, optional, default=0
        Lower bound of the Uniform distribution.
    
    b : float or int, optional, default=1
        Upper bound of the Uniform distribution. Must satisfy :math:`b \geq a`.
    
    Returns
    ----------
    float
        The value :math:`x` such that the cumulative probability :math:`P(X ≤ x) = p`
        for a :math:`Uniform(a, b)` distribution.
    
    Examples
    ----------
    >>> cont_uniform_invcdf(0.16379310344827586, 0.6, 12.2)
    2.5

    >>> cont_uniform_invcdf(0, 0, 1)
    0

    >>> cont_uniform_invcdf(0.5, 0, 1)
    0.5

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Continuous_uniform_distribution
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
    cont_uniform_rvs
    ===========
    Generate random variates from a continuous Uniform distribution, where the values lie
    between a specified lower :math:`(a)` and upper :math:`(b)` bound.
   
    Mathematical Definition
    ----------
    .. math::
        RVS_{U}(a, b) \sim U(a, b)
   
    Parameters
    ----------
    a : float or int, optional, default=0
        The lower bound of the distribution, i.e., the smallest possible value the random variate can take.
    
    b : float or int, optional, default=1
        The upper bound of the distribution, i.e., the largest possible value the random variate can take.
    
    size : int, optional, default=1
        The number of random variates to generate. If `size` is 1, a single random variate is returned as a float; if `size` is greater than 1, a list of random variates is returned.
    
    seed : int, optional, default=None
        The seed for the random number generator. Providing the same seed value will ensure the same sequence of random numbers is generated, allowing for reproducibility.

    Returns
    ----------
       
    Examples
    ----------
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
    ----------
    .. [1] https://en.wikipedia.org/wiki/Continuous_uniform_distribution
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
