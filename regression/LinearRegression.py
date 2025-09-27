#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import warnings
import datetime
import numpy as np
import dask.dataframe as dd
import dask.array as da
from data.loading import ddf_to_dda
from regression.dask_functions import pinv
from regression.metrics import r_squared, adj_r_squared, f_statistic

#-----------------------------------------------------------------------------#
# ---- CLASSES
#-----------------------------------------------------------------------------#
todo: WRITE ATTRIBUTES + MISSING VALUES TREATMENT
class OLS():
    r"""
    Ordinary Least Squares (OLS) Regression
    ===========
    Fits a linear regression model using the Ordinary Least Squares method, which minimizes
    the sum of squared differences between the observed dependent variable and the predictions
    from a linear function of the independent variables.
    
    Parameters
    ----------
    endog : array_like
        The endogenous (dependent) variable. Also known as the response variable or target (commonly
        denoted as `y`).
    
    exog : array_like
        The exogenous (independent) variables. Also known as features or predictors (commonly denoted as `X`).
    
    add_const : bool, optional, default=True
        If True, includes a constant (intercept) term in the regression model.
    
    missing : {None, 'drop', 'raise'}, optional, default=None
        How to handle missing values:
        - None: Do not check for missing values.
        - 'drop': Drop observations with at least one missing value.
        - 'raise': Raise an error if any missing values are detected.
        
    method : {'pinv', 'qr', 'svd', 'norm'}, optional, default='pinv'
        The method used to compute the regression coefficients:
        
        - 'pinv': Moore–Penrose pseudoinverse.
        - 'qr': Quadratic-Rectangular (QR) decomposition.
        - 'svd': Singular Value Decomposition (SVD).
        - 'norm': Normal equation (analytical solution).
    
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Ordinary_least_squares
    """
    
    # Setup
    #-------------------------------------------------------------------------#
    def __init__(self
                 ,endog = None
                 ,exog = None
                 ,add_const = True
                 ,missing = None
                 ,method = 'pinv'
                 ,use_t = True
                 ):
        
        # Input Initialization
        #---------------------------------------------------------------------#
        self.endog = endog
        self.exog = exog
        self.add_const = add_const
        if missing != None:
            self.missing = missing.lower()
        else:
            self.missing = missing
        self.method = method
        self.use_t = use_t
            
        # Attribute Initialization
        #---------------------------------------------------------------------#
        self.model_name = "OLS"
        self.n_obs = None
        self.const = None
        self.params = None
        self.ssr = None
        self.sse = None
        self.sst = None
        self.date_fit = None
        self.time_fit = None
        self.r_squared = None
        self.adj_r_squared = None
        self.f_stat = None
        self.f_pvalue = None
    
        # Input Validation
        #---------------------------------------------------------------------#
        if type(self.endog) == type(None):
            raise ValueError("Parameter 'endog' is equal to None.")
        elif [type(self.endog) != np.ndarray
              ,type(self.endog) != da.Array
              ,type(self.endog) != dd.dask_expr._collection.DataFrame
              ,type(self.endog) != dd.dask_expr._collection.Series].count(True) < 1:
            raise TypeError("Parameter 'endog' must be array-like.")
        elif [type(self.endog) != np.ndarray
              ,type(self.endog) != da.Array].count(True) > 0 and self.endog.ndim > 2:
            raise TypeError("Parameter 'endog' can only contain one variable.")
        elif [type(self.endog) != dd.dask_expr._collection.DataFrame
              ,type(self.endog) != dd.dask_expr._collection.Series].count(True) > 0 \
            and [type(self.endog) != dd.dask_expr._collection.DataFrame
                  ,type(self.endog) != dd.dask_expr._collection.Series].count(True) < 2 \
            and self.endog.ndim > 1:
            raise TypeError("Parameter 'endog' can only contain one variable.")
            
        if type(self.exog) == None:
            raise ValueError("Parameter 'exog' is equal to None.")
        elif [type(self.exog) != np.ndarray
              ,type(self.exog) != da.Array
              ,type(self.exog) != dd.dask_expr._collection.DataFrame
              ,type(self.exog) != dd.dask_expr._collection.Series].count(True) < 1:
            raise TypeError("Parameter 'exog' must be array-like.")

        if type(self.add_const) != bool:
            raise TypeError("Parameter 'add_intercept' is not a bool.")
            
        if type(self.missing) != type(None):
            if self.missing not in ['drop', 'raise']:
                raise ValueError("Parameter 'missing' must be equal to None, 'drop' or 'raise'.")
                
        if type(self.method) != str:
            raise TypeError("Parameter 'method' must be a string.")
        elif self.method.lower() not in ['pinv', 'qr', 'svd', 'norm']:
            raise ValueError("Parameter 'method' must be equal to 'pinv', 'qr', 'svd' or 'norm'.")
            
        if type(self.use_t) != bool:
            raise TypeError("Parameter 'use_t' must be a bool.")
        
        # Input Conversion
        #---------------------------------------------------------------------#
        if type(self.endog) == np.ndarray:
            self.endog = da.from_array(self.endog
                                       ,chunks='auto').persist()
        if type(self.endog) == da.Array:
            self.endog_name = "Y"
        elif [type(self.endog) != dd.dask_expr._collection.DataFrame
              ,type(self.endog) != dd.dask_expr._collection.Series].count(True) > 0 \
            and [type(self.endog) != dd.dask_expr._collection.DataFrame
                  ,type(self.endog) != dd.dask_expr._collection.Series].count(True) < 2:
            self.endog_name = self.endog.columns[0]
            self.endog = ddf_to_dda(self.endog).persist()
            
        if type(self.exog) == np.ndarray:
            self.exog = da.from_array(self.exog
                                      ,chunks='auto').persist()
        if type(self.exog) == da.Array:
            self.exog_name = [f"x_{i}" for i in range(self.exog.shape[1])]
        elif [type(self.exog) != dd.dask_expr._collection.DataFrame
              ,type(self.exog) != dd.dask_expr._collection.Series].count(True) > 0 \
            and [type(self.exog) != dd.dask_expr._collection.DataFrame
                  ,type(self.exog) != dd.dask_expr._collection.Series].count(True) < 2:
            self.exog_name = [i for i in self.exog.columns]
            self.exog = ddf_to_dda(self.exog).persist()
            
        self.method = self.method.lower()
        
        # Engine
        #---------------------------------------------------------------------#        
        # Determine the total number of observations
        self.n_obs = self.exog.shape[0]
        
        # Determine the regression coefficients
        self.fit()
        
        # Determine the date and time of the regression
        self.date_fit = datetime.date.today().strftime('%Y-%m-%d')
        self.time_fit = datetime.datetime.now().time().strftime('%H:%M:%S')
        
        # Calculate the r_squared and the sum of suares
        self.r_squared, self.sse, self.sst = r_squared(y_true = self.endog
                                                       ,y_pred = self.predict()
                                                       ,has_const = self.add_const)
        self.ssr = self.sst - self.sse
        
        # Calculate the adj_r_squared
        self.adj_r_squared = adj_r_squared(model = self)
        
        # Calculate the F-statistic
        self.f_stat, self.f_pvalue = f_statistic(model = self)

    # Regression Model
    #-------------------------------------------------------------------------#
    def fit(self):
        
        # Engine
        #---------------------------------------------------------------------#
        # Auxiliary Variables
        X = self.exog
        y = self.endog
        
        # Intercept Coefficient
        if self.add_const == True:
            
            if X.ndim == 1:
                dda_const = da.ones((X.shape[0], 1))
                X = da.concatenate([dda_const, X]
                                   ,axis = 1)
                X = X.rechunk('auto')
                
            else:
                if np.all(X[:, 0] == 1):
                    warnings.warn("The first variable of parameter 'exog' is already equal to 1 for each observation. Hence, it was not added.", RuntimeWarning)
                else:
                    dda_const = da.ones((X.shape[0], 1))
                    X = da.concatenate([dda_const, X]
                                       ,axis = 1)
                    X = X.rechunk('auto')
        else:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        
        # Missing Observations
        if type(self.missing) != type(None):
            
            # exog variable(s)
            if np.isnan(X).any() == True:
                if self.missing == 'drop':
                    mask = ~np.isnan(X).any(axis = 1)
                    X = X[mask]
                    y = y[mask]
                elif self.missing == 'raise':
                    raise ValueError("Parameter 'exog' has missing observations.")
            
            # endog variable
            if np.isnan(y).any() == True:
                if self.missing == 'drop':
                    mask = ~np.isnan(y).any(axis = 1)
                    X = X[mask]
                    y = y[mask]
                elif self.missing == 'raise':
                    raise ValueError("Parameter 'endog' has missing observations.")
        
        # Estimate the coefficients using OLS
        if self.method == 'pinv':
            self.params = pinv(X) @ y
            if self.add_const == True:
                self.const = self.params[0:1]
                self.params = self.params[1:]
            
        elif self.method == 'qr':
            Q, R = da.linalg.qr(X)
            self.params = np.linalg.solve(R, Q.T @ y)
            if self.add_const == True:
                self.const = self.params[0:1]
                self.params = self.params[1:]
                
        elif self.method == 'svd':
            U, S, VT = da.linalg.svd(X)
            S_inv = da.diag(1 / S)
            self.params = VT.T @ S_inv @ U.T @ y
            if self.add_const == True:
                self.const = self.params[0:1]
                self.params = self.params[1:]
            
        elif self.method == 'norm':
            self.params = da.linalg.inv(X.T @ X) @ X.T @ y
            if self.add_const == True:
                self.const = self.params[0:1]
                self.params = self.params[1:]
        
        # Convert the coefficients to readable arrays
        if isinstance(self.const, da.Array):
            self.const = self.const.compute()
            
        if isinstance(self.params, da.Array):
            self.params = self.params.compute()
                
    def predict(self, exog = None):
        
        # Engine
        #---------------------------------------------------------------------#
        # Auxiliary Variables
        if type(exog) == type(None):
            X = self.exog
        else:
            X = exog
        
        # Intercept Coefficient
        if self.add_const == True:
            
            if X.ndim == 1:
                dda_const = da.ones((X.shape[0], 1))
                X = da.concatenate([dda_const, X]
                                   ,axis = 1)
                X = X.rechunk('auto')
                
            else:
                if np.all(X[:, 0] == 1):
                    pass
                else:
                    dda_const = da.ones((X.shape[0], 1))
                    X = da.concatenate([dda_const, X]
                                       ,axis = 1)
                    X = X.rechunk('auto')
        else:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
                
        # Regression coefficients
        if type(self.const) != type(None):
            beta = np.concatenate((self.const, self.params))
        else:
            beta = self.params
                
        # Output
        #---------------------------------------------------------------------#
        return X @ beta