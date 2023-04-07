from typing import List

import numpy as np

from lib.base import Gaussian



def write_to_file(data: List[Gaussian], fname: str) -> None:
    """Write the means and variances (NOT covarainces!) of the passed data to a
    .csv file.
    
    Parameters
    - `data`: List[Gaussian]
        The data to be written out
    - `fname`: str
        The name of the output file
    """

    Ndims = data[0].Ndims
    data_out = np.empty((len(data), 2*Ndims))

    for i, g in enumerate(data):
        data_out[i, :Ndims] = g.mean
        data_out[i, Ndims:] = np.diag(g.cov)

    np.savetxt(f"{fname}", data_out, fmt="%f", delimiter=",")



def read_from_file(fname:str) -> List[Gaussian]:
    """Read the means and variances (NOT covariances!) from the passed file to a
    sequence of Gaussian states.
    
    Parameters
    - `fname`: str
        Name of the file to be read
    
    Returns
    - `data`: List[Gaussian]
        The data from the file as a list of Gaussian objects
    """

    file_data = np.loadtxt(f"{fname}", delimiter=",")
    Ndims = file_data.shape[1]//2
    data = []

    for row in file_data:
        g = Gaussian(row[:Ndims], np.diagflat(row[Ndims:]))
        data.append(g)

    return data