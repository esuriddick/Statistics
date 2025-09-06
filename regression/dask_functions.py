#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import dask.array as da

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def pinv(dask_array):
    """
    pinv
    ===========
    Computes the Moore-Penrose pseudo-inverse of a matrix using Dask for distributed computing.

    The function performs Singular Value Decomposition (SVD) on the input matrix and uses the 
    resulting singular values to calculate the pseudo-inverse. The function handles large matrices 
    in a distributed manner using Dask, and applies a tolerance to avoid division by zero in the 
    inversion of singular values.

    Parameters
    ----------
        dask_array : Dask Array
            A Dask Array representing the matrix to compute the pseudo-inverse for.

    Returns
    ----------
        Dask Array
            The pseudo-inverse of the input matrix.

    Notes
    ----------
        - The function computes the SVD of the input matrix using Dask's `da.linalg.svd`.
        
        - Singular values that are zero or very close to zero are treated with a tolerance (set to zero in the inverse).
        - The pseudo-inverse is computed as `v * s_inv_matrix * u.T`, where:
            
            - `u`, `s`, `vt` are the components of the SVD.
            - `s_inv` is the reciprocal of the singular values, with zero values set to zero to avoid division by zero errors.
            
        - The result is returned as a Dask Array, allowing further distributed computation.
    """
    
    # Compute the SVD using Dask
    # This will calculate u, s, vt (V transpose) in a distributed manner
    u, s, vt = da.linalg.svd(dask_array)
    
    # Compute the reciprocal of non-zero singular values with a tolerance
    s_inv = 1.0 / s
    s_inv[s_inv == float('inf')] = 0  # Set any infinite values (from division by zero) to zero
    
    # Rebuild the pseudo-inverse using u * s_inv * vt
    # We need to broadcast `s_inv` to the appropriate dimensions for matrix multiplication
    # First, construct s_inv as a diagonal matrix
    s_inv_matrix = da.diag(s_inv)
    
    # Compute the result: V * S_inv * U.T
    pinv_result = vt.T @ s_inv_matrix @ u.T  # Equivalent to v * s_inv * u.T
    
    # Compute the result eagerly (use compute() to trigger computation)
    pinv_matrix = pinv_result

    return pinv_matrix