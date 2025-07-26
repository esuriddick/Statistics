#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import math

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def beta(a, b):
    r"""
    beta
    ===========
    Computes the Beta function, which is a key function in probability and statistics.

    Mathematical Definition
    ----------
    .. math::
        B(a, b) = \int_{0}^{1} t^{a - 1}(1 - t)^{b - 1} \;dt = \frac{\Gamma(a) \Gamma(b)}{\Gamma(a + b)}

    where :math:`\Gamma` is the Gamma function.

    Parameters
    ----------
    a : float or int
        The first shape parameter of the Beta distribution. It influences the behavior of the function
        for small values of :math:`t`.

    b : float or int
        The second shape parameter of the Beta distribution. It influences the behavior of the function
        for values of :math:`t` close to 1.

    Returns
    ----------
    float
        The value of the Beta function for the given parameters :math:`a` and :math:`b`.

    Examples
    ----------
    >>> beta(1, 20)
    0.05000000000000003

    >>> beta(175, 200)
    7.767746671879741e-114

    >>> beta(5, 3)
    0.009523809523809509

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta_function
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(a) != float and type(a) != int:
        raise TypeError("Parameter 'a' is not a float or integer.")
    elif a < 0:
        raise ValueError("Parameter 'a' must be greater than 0.")

    if type(b) != float and type(b) != int:
        raise TypeError("Parameter 'b' is not a float or integer.")
    elif b < 0:
        raise ValueError("Parameter 'b' must be greater than 0.")

    # Engine
    #-------------------------------------------------------------------------#
    # Numerator
    num = math.lgamma(a) + math.lgamma(b)

    # Denominator
    den = math.lgamma(a + b)

    # Output
    #-------------------------------------------------------------------------#
    return math.exp(num - den)

def incbeta(x, a, b, tol = 1e-10, floor = 1e-30):
    r"""
    incbeta
    ===========
    Calculates the value of the regularized incomplete beta function, which is a generalization of
    the beta function. When x = 1, the incomplete beta function equals the complete beta function.


    Mathematical Definition
    ----------
    .. math::
        I_x(a, b) = \frac{B_x(a, b)}{B(a, b)} = \frac{1}{B(a, b)} \times \int_{0}^{x} t^{a - 1}(1 - t)^{b - 1} \;dt

    Where:
    
    - :math:`B_x(a, b)` is the incomplete beta function
    - :math:`B(a, b)` is the complete beta function
    - :math:`a` and :math:`b` are shape parameters
    - :math:`x` is the upper limit of the integration
    
    Parameters
    ----------
    x : float or int
        The upper limit of integration. Must be in the range :math:`[0, 1]`.
    
    a : float or int
        The first shape parameter of the beta distribution, which affects the behavior of the
        function near 0.
    
    b : float or int
        The second shape parameter of the beta distribution, which influences the behavior near 1.
    
    tol : float or int, optional, default=1e-10
        The tolerance for the solution's error. A smaller value increases accuracy but may require
        more iterations for convergence.
    
    floor : float or int, optional, default=1e-30
        The minimum acceptable value during the iteration process. Prevents values from becoming
        too small (underflow) in the numerical method used (Lentz's algorithm).

    Returns
    ----------
    float
        The regularized incomplete beta function value for the given parameters.

    Examples
    ----------
    >>> incbeta(0.1, 1, 20)
    0.8784233454094308

    >>> incbeta(0.765, 1.5, 5)
    0.9982605350915675

    >>> incbeta(0.5, 20, 1)
    9.536743164062494e-07

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
    .. [2] https://codeplea.com/incomplete-beta-function-c
    .. [3] https://github.com/codeplea/incbeta
    .. [4] https://dlmf.nist.gov/8.17#SS5.p1
    """

    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != float and type(x) != int:
        raise TypeError("Parameter 'x' is not a float or integer.")
    elif x < 0 or x > 1:
        return float('inf')

    if type(a) != float and type(a) != int:
        raise TypeError("Parameter 'a' is not a float or integer.")
    elif a < 0:
        raise ValueError("Parameter 'a' must be greater than 0.")

    if type(b) != float and type(b) != int:
        raise TypeError("Parameter 'b' is not a float or integer.")
    elif b < 0:
        raise ValueError("Parameter 'b' must be greater than 0.")
        
    if type(tol) != float and type(tol) != int:
        raise TypeError("Parameter 'tol' is not a float or integer.")
    elif tol <= 0:
        raise ValueError("Parameter 'tol' must be greater than 0.")
        
    if type(floor) != float and type(floor) != int:
        raise TypeError("Parameter 'floor' is not a float or integer.")
    elif floor <= 0:
        raise ValueError("Parameter 'floor' must be greater than 0.")

    # Engine & Output
    #-------------------------------------------------------------------------#  
    # Special case
    if x == 0 or x == 1:
        return x

    # Continued fraction converges nicely for x < (a + 1) / (a + b + 2)
    if x > (a + 1) / (a + b + 2):
        return (1 - incbeta(1 - x, b, a)) # Use the fact that the beta is symmetrical

    # Find the first part before the continued fraction
    lbeta_ab = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta_ab) / a

    # Use Lentz's algorithm to evaluate the continued fraction
    f = 1
    c = 1
    d = 0
    for i in range(201):
        m = i // 2

        if i == 0: # First numerator
            numerator = 1
        elif i % 2 == 0: # Even term
            numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else: # Odd term
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))

        # Do an iteration of Lentz's algorithm
        d = 1 + numerator * d
        if math.fabs(d) < floor:
            d = floor
        d = 1 / d

        c = 1 + numerator / c
        if math.fabs(c) < floor:
            c = floor

        cd = c * d
        f *= cd

        # Check for stop
        if math.fabs(1 - cd) < tol:
            return front * (f - 1)

    # More loops are required as it did not converge
    return float('inf')

def incgamma(x, s, tol = 1e-10, max_iter = 100):
    r"""
    incgamma
    ===========
    Computes the value of the non-regularized lower incomplete gamma function.

    Mathematical Definition
    ----------
    .. math::
        \gamma_x(s) = \int_{0}^{x} t^{s - 1} e^{-t} \, dt

    This function represents the incomplete gamma function with the lower limit of integration
    set to 0 and the upper limit set to :math:`x`.

    Parameters
    ----------
    x : float or int
        The upper limit of the integral. It must be a non-negative real number.
    
    s : float or int
        The shape parameter of the incomplete gamma function. It must be greater than 0.
    
    tol : float or int, optional, default=1e-10
        The tolerance (error margin) for the convergence of the calculation. The method terminates
        when the absolute difference between consecutive iterations is smaller than `tol`.
        A smaller value results in higher precision but may increase the number of iterations.
    
    max_iter : int, optional, default=100
        The maximum number of iterations to perform. This parameter acts as a safeguard to avoid
        infinite loops in case the method fails to converge.

    Returns
    ----------
    float
        The computed value of the non-regularized lower incomplete gamma function :math:`\gamma_x(s)`.

    Examples
    ----------
    >>> incgamma(5.5, 3)
    1.82324713528309

    >>> incgamma(3, 2.5)
    0.9222712123020332

    >>> incgamma(10, 4)
    5.937983695940045

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Incomplete_gamma_function
    """
   
    # Input Validation
    #-------------------------------------------------------------------------#
    if type(x) != float and type(x) != int:
        raise TypeError("Parameter 'x' is not a float or integer.")
    elif x < 0:
        return float("Parameter 'x' must be greater or equal to 0.")
   
    if type(s) != float and type(s) != int:
        raise TypeError("Parameter 's' is not a float or integer.")
    elif s <= 0:
        return float("Parameter 's' must be greater than 0.")
   
    if type(tol) != float and type(tol) != int:
        raise TypeError("Parameter 'tol' is not a float or integer.")
    elif tol <= 0:
        raise ValueError("Parameter 'tol' must be greater than 0.")

    if type(max_iter) != int:
        raise TypeError("Parameter 'max_iter' is not an integer.")
    elif max_iter <= 0:
        raise ValueError("Parameter 'max_iter' must be greater than 0.")
       
    # Engine & Output
    #-------------------------------------------------------------------------#
    term = 1 / s
    total = term
    for n in range(1, max_iter):
        term *= x / (s + n)
        total += term
        if term < tol:
            break
    return total * math.exp(-x + s * math.log(x))
