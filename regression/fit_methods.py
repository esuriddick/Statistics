#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import warnings
import dask.array as da
from regression.link_functions import link_Identity, link_Logit

#-----------------------------------------------------------------------------#
# ---- ORDINARY LEAST SQUARES (OLS)
#-----------------------------------------------------------------------------#
def OLS_pinv(y, X):
    """
    OLS_pinv
    ===========
    Perform Ordinary Least Squares (OLS) regression using pseudo-inverse method.

    Parameters
    ----------
    y : Dask Array
        A 1D array or vector of observed values (dependent variable).
        
    X : Dask Array
        A 2D array representing the design matrix of explanatory variables (independent variables).

    Returns
    ----------
    params : Dask Array
        A 1D array of the estimated regression coefficients (`beta`).
    """
    
    # Compute the SVD of the design matrix X
    u, s, vt = da.linalg.svd(X)
    
    # Compute the reciprocal of non-zero singular values with an adjustment to avoid infinite values
    s_inv = 1.0 / s
    s_inv[s_inv == float('inf')] = 0
    
    # Rebuild the pseudo-inverse using the formula: X' = u * s_inv * vt
    s_inv_matrix = da.diag(s_inv)
    
    # Compute the result: v * S_inv * u.T
    pinv_result = vt.T @ s_inv_matrix @ u.T  # Equivalent to v * s_inv * u.T
    
    # Determine regression coefficients
    params = pinv_result @ y

    # Output
    return params

def OLS_qr(y, X):
    """
    OLS_qr
    ===========
    Perform Ordinary Least Squares (OLS) regression using QR decomposition.

    Parameters:
    ----------
    y : Dask Array
        A 1D array or vector of observed values (dependent variable).
    
    X : Dask Array
        A 2D array representing the design matrix of explanatory variables (independent variables).

    Returns:
    ----------
    params : Dask Array
        A 1D array of the estimated regression coefficients (`beta`).
    """
    
    # Perform QR decomposition of the design matrix X
    Q, R = da.linalg.qr(X)
    
    # Determine regression coefficients
    params = da.linalg.solve(R, Q.T @ y)
    
    # Output
    return params

def OLS_svd(y, X):
    """
    OLS_svd
    ===========
    Perform Ordinary Least Squares (OLS) regression using Singular Value Decomposition (SVD).

    Parameters:
    ----------
    y : Dask Array
        A 1D array or vector of observed values (dependent variable).
    
    X : Dask Array
        A 2D array representing the design matrix of explanatory variables (independent variables).

    Returns:
    ----------
    params : Dask Array
        A 1D array of the estimated regression coefficients (`beta`).
    """
    
    # Perform SVD of the design matrix X
    U, S, VT = da.linalg.svd(X)
    
    # Create a diagonal matrix of the inverses of the singular value
    S_inv = da.diag(1 / S)
    
    # Determine regression coefficients
    params = VT.T @ S_inv @ U.T @ y

    # Output
    return params

def OLS_norm(y, X):
    """
    OLS_norm
    ===========
    Perform Ordinary Least Squares (OLS) regression using the analytical solution.

    Parameters:
    ----------
    y : Dask Array
        A 1D array or vector of observed values (dependent variable).
    
    X : Dask Array
        A 2D array representing the design matrix of explanatory variables (independent variables).

    Returns:
    ----------
    params : Dask Array
        A 1D array of the estimated regression coefficients (`beta`).
    """
    
    # Determine regression coefficients
    params = da.linalg.inv(X.T @ X) @ X.T @ y

    # Output
    return params

#-----------------------------------------------------------------------------#
# ---- GENERALIZED LINEAR MODELS (GLMs)
#-----------------------------------------------------------------------------#
def GLM_irls(y
             ,X
             ,dist = 'binomial'
             ,f = link_Logit
             ,add_const = True
             ,tol = 1e-8
             ,max_iter = 100
             ,warn = True):
    """
    GLM_irls
    ===========
    Generalized Linear Model (GLM) using Iteratively Reweighted Least Squares (IRLS).
    
    Parameters:
    ----------
    y : Dask Array
        A 1D array or vector of observed values (dependent variable).
        
    X : Dask Array
        A 2D array representing the design matrix of explanatory variables (independent variables).
        
    dist : string, optional, default='binomial'
        The distribution of the observed values (i.e., dependent variable).
        The options are:
            - 'gaussian' or 'normal';
            - 'binomial'.
        
    f : function, optional, default=link_Logit
        The link function for the GLM.
        The options are:
            - link_Identity;
            - link_Logit.
    
    add_const : bool, optional, default=True
        Whether the model has an intercept term or not.
    
    tol : float, optional, default=1e-8
        Convergence tolerance. The algorithm will stop if the log-likelihood change between
        iterations is less than `tol`.
    
    max_iter : int, default=100
        Maximum number of iterations for the IRLS algorithm.
    
    warn : bool, default=True
        Whether to issue a warning if the algorithm does not converge within `max_iter`.

    Returns:
    ----------
    params : Dask Array
        A 1D array of the estimated regression coefficients (`beta`).
    """
    
    # Input Validation
    #-------------------------------------------------------------------------#     
    if type(dist) != str:
        raise TypeError("Parameter 'dist' is not a string.")
    else:
        dist = dist.lower()
    
    if f not in [link_Identity, link_Logit]:
        raise TypeError("Parameter 'f' is not a recognised link function.")
        
    if type(add_const) != bool:
        raise TypeError("Parameter 'add_intercept' is not a bool.")
        
    if type(tol) != float and type(tol) != int:
        raise TypeError("Parameter 'tol' is not a float or integer.")
    elif tol <= 0:
        raise ValueError("Parameter 'tol' must be greater than 0.")
        
    if type(max_iter) != int:
        raise TypeError("Parameter 'max_iter' is not an integer.")
    elif max_iter <= 0:
        raise ValueError("Parameter 'max_iter' must be greater than 0.")
        
    if type(warn) != bool:
        raise TypeError("Parameter 'warn' is not a boolean.")
        
    # Distribution of Y - Gaussian/Normal
    #-------------------------------------------------------------------------#
    if dist in ['gaussian', 'normal']:
        if f == link_Identity:
            # Initialize parameters and variables
            conv = False # Convergence Flag
            n, p = X.shape
            params = da.zeros(p) # Initialize coefficients (parameters)
            ll = 0 # Initial log-likelihood
            
            # Add intercept if requested
            if add_const == True:
                X = da.concatenate([da.ones((n, 1)), X], axis = 1)
                X = X.rechunk('auto')
                intercept = da.mean(y) # Best guess for the intercept
                params = da.concatenate([intercept.reshape(1, ), params])
            params = params.reshape(-1, 1) # Reshape to column vector
                
            # Iterated Reweighted Least Squares (IRLS) Loop
            for itr in range(max_iter):
                ll_old = ll # Save the previous log-likelihood
                
                # Linear predictor and mean (fitted values)
                eta = X @ params
                mu = f(X, params)
                
                # Variance of the model
                s = da.ones_like(mu)
                
                # Weight matrix (diagonal matrix with variances)
                S = da.eye(len(s)) * s
                
                # Response Vector [eta + (y - mu) / s can be simplied to y]
                z = y
                
                # Compute the new regression coefficients
                XtS = (X.T @ S)
                XtSX = XtS @ X
                XtSz = XtS @ z
                params, _, _, _ = da.linalg.lstsq(XtSX, XtSz)
                
                # Compute the log-likelihood for convergence
                ll = -0.5 * da.sum((y - mu) ** 2)
                ll = da.sum(ll).compute()
                
                # Check for convergence
                if abs(ll - ll_old) < tol:
                    conv = True
                    break
                
            # Check for convergence failure
            if conv == False:
                if warn == True:
                    warnings.warn("Convergence was not achieved. Either increase the value of the parameter 'tol' or the value of the parameter 'max_iter'.", UserWarning)

        else:
            raise ValueError("Value of parameter 'f' is not recognised.")
    
    # Distribution of Y - Binomial
    #-------------------------------------------------------------------------#
    elif dist == 'binomial':
        if f == link_Logit:
            # Initialize parameters and variables
            conv = False # Convergence Flag
            n, p = X.shape
            params = da.zeros(p) # Initialize coefficients (parameters)
            ll = 0 # Initial log-likelihood
            
            # Add intercept if requested
            if add_const == True:
                X = da.concatenate([da.ones((n, 1)), X], axis = 1)
                X = X.rechunk('auto')
                intercept = da.log(da.mean(y) / (1 - da.mean(y))) # Best guess for the intercept
                params = da.concatenate([intercept.reshape(1, ), params])
            params = params.reshape(-1, 1) # Reshape to column vector
            
            # Iterated Reweighted Least Squares (IRLS) Loop
            for itr in range(max_iter):
                ll_old = ll # Save the previous log-likelihood
                
                # Linear predictor and mean (fitted values)
                eta = X @ params
                mu = f(X, params)
                
                # Variance of the model
                s = mu * (1 - mu)
                
                # Weight matrix (diagonal matrix with variances)
                S = da.eye(len(s)) * s
                
                # Response Vector
                z = eta + (y - mu) / s
                
                # Compute the new regression coefficients
                XtS = (X.T @ S)
                XtSX = XtS @ X
                XtSz = XtS @ z
                params, _, _, _ = da.linalg.lstsq(XtSX, XtSz)
                
                # Compute the log-likelihood for convergence
                ll = y * da.log(f(X, params)) + (1 - y) * da.log(1 - f(X, params))
                ll = da.sum(ll).compute()
                
                # Check for convergence
                if abs(ll - ll_old) < tol:
                    conv = True
                    break
                
            # Check for convergence failure
            if conv == False:
                if warn == True:
                    warnings.warn("Convergence was not achieved. Either increase the value of the parameter 'tol' or the value of the parameter 'max_iter'.", UserWarning)
                
        else:
            raise ValueError("Value of parameter 'f' is not recognised.")
    
    # Distribution of Y - Unknown
    #-------------------------------------------------------------------------#
    else:
        raise ValueError("Value of parameter 'dist' is not recognised.")
        
    # Output
    #-------------------------------------------------------------------------#
    return params

def GLM_sgd(y
            ,X
            ,dist = 'binomial'
            ,f = link_Logit
            ,add_const = True
            ,l_rate = 0.01
            ,tol = 1e-8
            ,max_iter = 100
            ,warn = True):
    """
    GLM_sgd
    ===========
    Generalized Linear Model (GLM) using Stochastic Gradient Descent (SGD).
    
    Parameters:
    ----------
    y : Dask Array
        A 1D array or vector of observed values (dependent variable).
        
    X : Dask Array
        A 2D array representing the design matrix of explanatory variables (independent variables).
        
    dist : string, optional, default='binomial'
        The distribution of the observed values (i.e., dependent variable).
        The options are:
            - 'gaussian' or 'normal';
            - 'binomial'.
        
    f : function, optional, default=link_Logit
        The link function for the GLM.
        The options are:
            - link_Identity;
            - link_Logit.
    
    add_const : bool, optional, default=True
        Whether the model has an intercept term or not.

    l_rate : float, optional, default=0.1
        How much you adjust your model's parameters in response to the estimated error each time the
        model's weights are updated.
    
    tol : float, optional, default=1e-8
        Convergence tolerance. The algorithm will stop if the log-likelihood change between
        iterations is less than `tol`.
    
    max_iter : int, default=100
        Maximum number of iterations for the IRLS algorithm.
    
    warn : bool, default=True
        Whether to issue a warning if the algorithm does not converge within `max_iter`.

    Returns:
    ----------
    params : Dask Array
        A 1D array of the estimated regression coefficients (`beta`).
    """
    
    # Input Validation
    #-------------------------------------------------------------------------#     
    if type(dist) != str:
        raise TypeError("Parameter 'dist' is not a string.")
    else:
        dist = dist.lower()
    
    if f not in [link_Identity, link_Logit]:
        raise TypeError("Parameter 'f' is not a recognised link function.")
        
    if type(add_const) != bool:
        raise TypeError("Parameter 'add_intercept' is not a bool.")

    if type(l_rate) != float and type(l_rate) != int:
        raise TypeError("Parameter 'l_rate' is not a float or integer.")
    elif l_rate <= 0:
        raise ValueError("Parameter 'l_rate' must be greater than 0.")
        
    if type(tol) != float and type(tol) != int:
        raise TypeError("Parameter 'tol' is not a float or integer.")
    elif tol <= 0:
        raise ValueError("Parameter 'tol' must be greater than 0.")
        
    if type(max_iter) != int:
        raise TypeError("Parameter 'max_iter' is not an integer.")
    elif max_iter <= 0:
        raise ValueError("Parameter 'max_iter' must be greater than 0.")
        
    if type(warn) != bool:
        raise TypeError("Parameter 'warn' is not a boolean.")
        
    # Distribution of Y - Gaussian/Normal
    #-------------------------------------------------------------------------#
    if dist in ['gaussian', 'normal']:
        if f == link_Identity:
            # Initialize parameters and variables
            conv = False # Convergence Flag
            n, p = X.shape
            params = da.zeros(p) # Initialize coefficients (parameters)
            ll = 0 # Initial log-likelihood
            
            # Add intercept if requested
            if add_const == True:
                X = da.concatenate([da.ones((n, 1)), X], axis = 1)
                X = X.rechunk('auto')
                intercept = da.mean(y) # Best guess for the intercept
                params = da.concatenate([intercept.reshape(1, ), params])
            params = params.reshape(-1, 1) # Reshape to column vector
                
            # Stochastic Gradient Descent (SGD) Loop
            for itr in range(max_iter):
                ll_old = ll # Save the previous log-likelihood
                
                # Linear predictor and mean (fitted values)
                eta = X @ params
                mu = f(X, params)
                
                # Gradient
                gradient = X.T @ (y - mu)

                # Update parameters
                params = params + l_rate * gradient
                
                # Compute the log-likelihood for convergence
                ll = -0.5 * da.sum((y - mu) ** 2)
                ll = da.sum(ll).compute()
                
                # Check for convergence
                if abs(ll - ll_old) < tol:
                    conv = True
                    break
                
            # Check for convergence failure
            if conv == False:
                if warn == True:
                    warnings.warn("Convergence was not achieved. Either increase the value of the parameter 'tol' or the value of the parameter 'max_iter'.", UserWarning)

        else:
            raise ValueError("Value of parameter 'f' is not recognised.")
    
    # Distribution of Y - Binomial
    #-------------------------------------------------------------------------#
    elif dist == 'binomial':
        if f == link_Logit:
            # Initialize parameters and variables
            conv = False # Convergence Flag
            n, p = X.shape
            params = da.zeros(p) # Initialize coefficients (parameters)
            ll = 0 # Initial log-likelihood
            
            # Add intercept if requested
            if add_const == True:
                X = da.concatenate([da.ones((n, 1)), X], axis = 1)
                X = X.rechunk('auto')
                intercept = da.log(da.mean(y) / (1 - da.mean(y))) # Best guess for the intercept
                params = da.concatenate([intercept.reshape(1, ), params])
            params = params.reshape(-1, 1) # Reshape to column vector
            
            # Stochastic Gradient Descent (SGD) Loop
            for itr in range(max_iter):
                ll_old = ll # Save the previous log-likelihood
                
                # Linear predictor and mean (fitted values)
                eta = X @ params
                mu = f(X, params)

                # Gradient
                gradient = X.T @ (y - mu)

                # Update parameters
                params = params + l_rate * gradient
                
                # Compute the log-likelihood for convergence
                ll = y * da.log(f(X, params)) + (1 - y) * da.log(1 - f(X, params))
                ll = da.sum(ll).compute()
                
                # Check for convergence
                if abs(ll - ll_old) < tol:
                    conv = True
                    break
                
            # Check for convergence failure
            if conv == False:
                if warn == True:
                    warnings.warn("Convergence was not achieved. Either increase the value of the parameter 'tol' or the value of the parameter 'max_iter'.", UserWarning)
                
        else:
            raise ValueError("Value of parameter 'f' is not recognised.")
    
    # Distribution of Y - Unknown
    #-------------------------------------------------------------------------#
    else:
        raise ValueError("Value of parameter 'dist' is not recognised.")
        
    # Output
    #-------------------------------------------------------------------------#
    return params
