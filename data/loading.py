#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import dask.dataframe as dd
import dask.array as da

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def raw_to_ddf(filepath, sep = ';', decimal = '.'):
    """
    Loads a raw data file (CSV or Parquet) into a Dask DataFrame.
    ===========

    This function reads a `.csv` or `.parquet` file from the specified file path 
    and returns it as a Dask DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the input file. Must end with `.csv` or `.parquet`.
    
    sep : str, optional, default=';'
        Column delimiter used in the CSV file. Defaults to ';'.
    
    decimal : str, optional, default='.'
        Decimal delimiter used in the CSV file. Defaults to '.'.

    Returns
    ----------
    dask.dataframe.DataFrame:
        A Dask DataFrame if the file is successfully read.
    """
    
    # Input Validation
    #-------------------------------------------------------------------------#
    if type(filepath) == type(None) or filepath == '':
        raise ValueError("No file was selected.")
        
    if type(sep) != str:
        raise ValueError("The column delimiter must be a string.")
        
    if type(decimal) != str:
        raise ValueError("The decimal delimiter must be a string.")
        
    if [filepath.endswith('.csv')
        ,filepath.endswith('.parquet')].count(True) < 1:
        raise ValueError("The selected file's type is not recognised.")
    
    # Engine
    #-------------------------------------------------------------------------#
    if filepath.endswith('.csv'):
        df = dd.read_csv(urlpath = filepath
                         ,sep = sep
                         ,decimal = decimal)
    elif filepath.endswith('.parquet'):
        df = dd.read_parquet(path = filepath)

    # Output
    #-------------------------------------------------------------------------#
    return df

def ddf_to_dda(ddf):
    """
    Loads a Dask DataFrame into a Dask Array.
    ===========

    This function converts a Dask DataFrame or Dask Series to a Dask Array, while
    excluding all non-numeric variables.

    Parameters
    ----------
    ddf : Dask DataFrame or Dask Series
        Name of the Dask DataFrame or Dask Series to be converted to a Dask Array.

    Returns
    ----------
    dask.array.core.Array:
        A Dask Array.
    """
    
    # Input Validation
    #-------------------------------------------------------------------------#
    if [type(ddf) != dd.dask_expr._collection.DataFrame
        ,type(ddf) != dd.dask_expr._collection.Series].count(True) < 1:
        raise ValueError("The parameter 'ddf' is not a Dask Dataframe or Dask Series.")

    # Input Conversion
    #-------------------------------------------------------------------------#
    if type(ddf) == dd.dask_expr._collection.Series:
        ddf = ddf.to_frame()
    
    # Engine
    #-------------------------------------------------------------------------#
    # Exclude non-numerical variables
    ddf = ddf.select_dtypes(include = 'number')
    
    # Convert to Array
    dda = ddf.to_dask_array(lengths = True)
    dda = dda.rechunk('auto')
    
    # Output
    #-------------------------------------------------------------------------#
    return dda