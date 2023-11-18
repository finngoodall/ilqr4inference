from numpy.typing import NDArray

from .utils import is_pos_def



class Gaussian():
    """Class for a Gaussian distribuion over an inferred state."""

    def __init__(self, mean: NDArray , cov: NDArray):
        """Construct the Gaussian object.
        
        Parameters
        - `mean`: NDArray
            The mean of the distribution
        - `cov`: NDArray
            The covariance of the distribution
        """

        self.mean = mean
        self.cov = cov
        if self.mean.shape:
            self.Ndims = self.mean.shape[0]
        else:
            # An empty shape tuple means a size of 1
            self.Ndims = 1

        if len(self.mean.shape) > 1:
            raise ValueError(f"State mean must be 1 dimensional")
        if not is_pos_def(self.cov):
            raise ValueError("Covariance matrix must be positive definite")
        if self.Ndims != self.cov.shape[0]:
            raise ValueError(f"Mean and covariance must have equal dimensions")
        
    # Used for ease in commparing to NoneTypes
    def __bool__(self):
        return True