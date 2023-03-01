from numpy.typing import NDArray
from typing import Tuple

import autograd.numpy as np
from scipy.linalg import solve, svd



def is_square(mat: NDArray) -> bool:
    """Whether `mat` is a square matrix."""

    return len(mat.shape) == 2 and mat.shape[0] == mat.shape[1]



def is_symmetric(mat: NDArray) -> bool:
    """Whether `mat` is a symmetric matrix."""

    return is_square(mat) and np.allclose(mat, mat.T)



def is_pos_def(mat: NDArray,
               return_eigs: bool = False) -> bool | Tuple[bool, NDArray]:
    """Whether `mat` is a positive definite matrix.
    
    If `return_eigs`, the eigenvalues of `mat` are also returned."""

    eigs = np.linalg.eigvals(mat)
    pos_def = is_symmetric(mat) and np.all(eigs > 0)
    
    if return_eigs:
        return pos_def, eigs
        
    return pos_def



def diag_regularise(mat: NDArray, eiglim: float = 1e-8) -> NDArray:
    """Adds a digonal matrix to `mat` so that the resulting matrix is positive
    definite with all eigenvalues >= `eiglim`.
    
    If `mat` is already positive definite, this returns `mat` unchanged."""

    pos_def, eigs = is_pos_def(mat, return_eigs=True)
    min_eig = np.min(eigs)

    if pos_def:
        return mat
    
    reg = -min_eig*np.ones(mat.shape[0]) + eiglim
    return mat + np.diag(reg)



def eig_regularise(mat: NDArray, eiglim: float = 1e-8) -> NDArray:
    """# DON'T USE THIS - SOMETIMES IT DOES NOT RETURN A POS-DEF MATRIX!
    
    Makes `mat` positive definite by performing an eigenvalue decomposition
    and setting all negative eigenvalues to `eiglim`.
    
    If `mat` is already positive definite, this returns `mat` unchanged."""

    E, V = np.linalg.eig(mat)
    E[E <= 0] = eiglim

    return V @ np.diag(E) @ V.T



def requad(x: NDArray | float) -> NDArray | float:
    """Evaluates the requad function of `x`:
    
    requad(x) = (x + sqrt(x^2 + 4))/2"""

    return (x + np.sqrt(4+x**2))/2



def pd_svd_inv(mat: NDArray, tiny: float = 1e-8) -> NDArray:
    """Inverts the positive definite matrix `mat` using an SVD, adding `tiny`
    to the eigenvalues for stability."""
    
    u, s, _ = svd(mat)
    return u/(s + tiny) @ u.T



def pd_inv(mat: NDArray) -> NDArray:
    """Inverts the positive definite matrix `mat`."""

    return solve(mat, np.eye(mat.shape[0]), assume_a="pos")



