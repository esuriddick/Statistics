#-----------------------------------------------------------------------------#
# ---- NATIVE MODULES
#-----------------------------------------------------------------------------#
import numpy as np
import dask.array as da

#-----------------------------------------------------------------------------#
# ---- CUSTOM MODULES
#-----------------------------------------------------------------------------#
import sys
import os
# Dynamically adjust the path to include the parent directory,
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from probability_distributions.continuous.f import f_cdf

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def r_squared(y_true, y_pred, has_const = True):
    """
    R-squared (Coefficient of Determination)
    ===========
    Calculate the coefficient of determination (R-squared) for a regression model. 
    This metric indicates how well the predicted values approximate the actual data. 
    The R-squared value ranges from 0 to 1, where 1 indicates perfect prediction, 
    and a value closer to 0 suggests poor model performance.

    This implementation allows for two variants of the R-squared formula:
        
    - **Centered R-squared**: The typical R-squared calculation, where the total
    sum of squares (SS_tot) is based on the variance of `y_true` around its mean.
    
    - **Uncentered R-squared**: A variant of R-squared where the total sum of
    squares (SS_tot) is based on the raw values of `y_true` without subtracting
    its mean. This can be useful in certain cases, such as when the regression
    model does not include an intercept term.

    Parameters
    ----------
    y_true : array-like
        Array of true (observed) target values.
    
    y_pred : array-like
        Array of predicted target values from the regression model.
        
    add_const : bool, optional, default = True
       Whether to use the centered (default) or uncentered R-squared formula:
       
       - If `True`, the calculation will use the centered formula (SS_tot is the
       variance of `y_true` around its mean).
       - If `False`, the uncentered formula will be used (SS_tot is based on the
       raw values of `y_true`).

    Returns
    ----------
    float
        The R-squared score, calculated as:
            
        - Centered R-squared: `1 - (SS_res / SS_tot)`
        - Uncentered R-squared: `1 - (SS_res / SS_tot)` (with SS_tot as the raw
        sum of squares of `y_true`).
    
    Notes
    ----------
    - This function assumes that `y_true` and `y_pred` are NumPy arrays or
    array-like objects of the same shape.
    
    - The uncentered R-squared formula is used in cases where the model does not
     include an intercept (e.g., when fitting a regression model through the origin).
    """
    
    # Engine
    #-------------------------------------------------------------------------#
    ss_res = np.sum((y_true - y_pred) ** 2)
    if has_const == True:
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    else:
        ss_tot = np.sum(y_true ** 2)
    
    # Output
    #-------------------------------------------------------------------------#
    res = 1 - ss_res / ss_tot
    if isinstance(res, da.Array):
        res = res.compute()
    if isinstance(ss_res, da.Array):
        ss_res = ss_res.compute()
    if isinstance(ss_tot, da.Array):
        ss_tot = ss_tot.compute()
    return res, ss_res, ss_tot

def adj_r_squared(model):
    """
    Adjusted R-squared
    ===========
    Calculate the adjusted R-squared for a regression model.

    Adjusted R-squared modifies the standard R-squared to account for the number 
    of predictors in the model. It provides a more accurate measure of model 
    performance, especially when comparing models with a different number of 
    independent variables.

    Parameters
    ----------
    model : object
        A fitted regression model object that must have the following attributes:
            
        - model.n_obs : int
            Number of observations in the dataset.
            
        - model.params : array-like
            Model parameters or coefficients (used to determine number of predictors).
            
        - model.r_squared : float
            The R-squared value of the fitted model.

    Returns
    ----------
    float
        The adjusted R-squared score, calculated as:

            1 - [(1 - R²) * (n - 1)] / (n - k - 1)

        where:
        - R² is the standard R-squared,
        - n is the number of observations,
        - k is the number of predictors (excluding the intercept).
    
    Notes
    ----------
    This function assumes the model object exposes `n_obs`, `params`, 
    and `r_squared` as accessible attributes.
    """
    
    # Engine
    #-------------------------------------------------------------------------#
    n = model.n_obs
    k = model.params.shape[0]
    r_squared = model.r_squared
    num = (1 - r_squared) * (n - 1)
    den = n - k - 1
    
    # Output
    #-------------------------------------------------------------------------#
    res = 1 - num/den
    if isinstance(res, da.Array):
        res = res.compute()
    return res

def f_statistic(model):
    """
    F-statistic
    ===========
    Computes the F-statistic for a given statistical model.

    The F-statistic is used to test the overall significance of a regression model. 
    It is the ratio of the Mean Square Regression (MSR) to the Mean Square Error
    (MSE).
    
    Parameters
    ----------
    model  : object
        A fitted regression model object that must have the following attributes:
            
        - model.n_obs : int
            Number of observations in the dataset.
        
        - model.params : array-like
            Model parameters or coefficients (used to determine number of predictors).
        
        - model.ssr : float
            The sum of squares due to regression.
        
        - model.sse : float
            The sum of squares due to error.

    Returns
    ----------
    float, float
        The F-statistic value and p-value.
        
    Notes
    ----------
        - The F-statistic is calculated as the ratio of the Mean Square Regression
         (MSR) to the Mean Square Error (MSE).
        - MSR is computed as SSR / k, where `k` is the number of parameters
         (without the intercept).
        - MSE is computed as SSE / (n - k - 1), where `n` is the number of
        observations.
    """
    
    # Engine
    #-------------------------------------------------------------------------#
    # Statistic
    n = model.n_obs
    k = model.params.shape[0]
    if type(model.const) == type(None):
        const_modifier = 0
    else:
        const_modifier = 1
    ssr = model.ssr
    sse = model.sse
    MSR = ssr / k
    MSE = sse / (n - k - const_modifier)
    stat = MSR / MSE
    if isinstance(stat, da.Array):
        stat = stat.compute()
    
    # P-value
    pval = 1 - f_cdf(float(stat), 1, n - k - const_modifier)
    
    # Output
    #-------------------------------------------------------------------------#
    return stat, pval