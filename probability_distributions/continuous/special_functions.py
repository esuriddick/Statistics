#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import math

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def beta(a, b):
    r"""
    Retrieves the value of the beta function.

    Function
    ===========
    .. math::
        B(a, b) = \int_{0}^{1} t^{a - 1}(1 - t)^{b - 1} \;dt = \frac{\Gamma(a) \Gamma(b)}{\Gamma(a + b)}

    Parameters
    ===========
    a : float or integer

    Shape parameter of the beta distribution. It influences the behaviour of the function for small values of t.

    b : float or integer

    Shape parameter of the beta distribution. It influences the behaviour of the function for values of t close to 1.

    Examples
    ===========
    >>> beta(1, 20)
    0.05000000000000003

    >>> beta(175, 200)
    7.767746671879741e-114

    >>> beta(5, 3)
    0.009523809523809509

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Beta_function)
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
    Retrieves the value of the regularized incomplete beta function, which is a generalization of the beta function. For x = 1, the incomplete beta function is exactly equal to the (complete) beta function.

    Function
    ===========
    .. math::
        I_x(a, b) = \frac{B_x(a, b)}{B(a, b)} = \frac{1}{B(a, b)} \times \int_{0}^{x} t^{a - 1}(1 - t)^{b - 1} \;dt

    Parameters
    ===========
    x : float or integer

    Upper limit of the integration.

    a : float or integer

    Shape parameter of the beta distribution. It influences the behaviour of the function for small values of t.

    b : float or integer

    Shape parameter of the beta distribution. It influences the behaviour of the function for values of t close to 1.

    tol : float

    Acceptable error margin for the solution. It determines how close the function value must be to zero for the method to consider the solution as converged. A smaller tol value means higher accuracy but may require more iterations. Default value is 1e-10.

    floor : float

    Minimum acceptable value for the values obtained during the Lentz's algorithm iteration. Default value is 1e-30.

    Examples
    ===========
    >>> incbeta(0.1, 1, 20)
    0.8784233454094308

    >>> incbeta(0.765, 1.5, 5)
    0.9982605350915675

    >>> incbeta(0.5, 20, 1)
    9.536743164062494e-07

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function)
    .. [2] https://codeplea.com/incomplete-beta-function-c
    .. [3] https://github.com/codeplea/incbeta
    .. [4] Digital Library of Mathematical Functions (https://dlmf.nist.gov/8.17#SS5.p1)
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
    Retrieves the value of the (non-regularised) lower incomplete gamma function.

    Function
    ===========
    .. math::
        \gamma_x(s) = \int_{0}^{x} t^{s - 1} e^{-t} \;dt

    Parameters
    ===========
    x : float or integer

    Upper limit of the integration.

    s : float or integer

    Shape parameter of the function.

    tol : float

    Acceptable error margin for the solution. It determines how close the function value must be to zero for the method to consider the solution as converged. A smaller tol value means higher accuracy but may require more iterations. Default value is 1e-10.

    max_iter : float

    Maximum number of iterations until the convergence method is stopped. It acts as a safeguard to prevent infinite loops in case the method does not converge. Default value is 100.

    Examples
    ===========
    >>> incgamma(5.5, 3)
    1.82324713528309

    >>> incgamma(3, 2.5)
    0.9222712123020332

    >>> incgamma(10, 4)
    5.937983695940045

    References
    ===========
    .. [1] Wikipedia (https://en.wikipedia.org/wiki/Incomplete_gamma_function)
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
