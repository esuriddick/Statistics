#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import dask.array as da

#-----------------------------------------------------------------------------#
# ---- REGRESSION EQUATIONS
#-----------------------------------------------------------------------------#
def link_Identity(X, beta):
    return X @ beta

def link_Logit(X, beta):
    # Stable sigmoid implementation
    x = X @ beta
    pos_mask = x >= 0
    neg_mask = ~pos_mask

    # Compute z based on sign of X
    z = da.where(pos_mask, da.exp(-x), da.exp(x))

    # Compute numerator
    num = da.where(neg_mask, z, 1.0)

    # Result
    return num / (1 + z)
