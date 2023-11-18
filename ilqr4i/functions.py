from .models.dynamics.base import Dynamics
from .models.measurements.base import MeasurementModel
from .models.priors.base import Prior



def sample_trajectory(
        dynamics: Dynamics,
        meas_model: MeasurementModel,
        input_prior: Prior,
        x0_prior: Prior,
        T: int
    ):
    """Sample sequences of corresponding states, control inputs and observations
    from a generative model.
    
    Parameters
    - `dynamics`:
        The dynamics of the latent states in the generative model
    - `meas_model`:
        The observation emission model of the generative model
    - `input_prior`:
        The prior over the control inputs in the generative model
    - `x0_prior`:
        The prior over the initial state in the sequence in the generative model
    - `T`:
        The number of timesteps in the sampled sequences
        
    Returns
    - Tuple of...
        - `xs`:
            The sampled states
        - `us`:
            The sampled control inputs
        - `ys`:
            The sampled observations
    """

    xs = []
    us = []
    ys = []

    for t in range(T):
        if t == 0:
            x = x0_prior.sample(t)
        
        u = input_prior.sample(t)
        y = meas_model.sample(x, t)
        
        xs.append(x)
        us.append(u)
        ys.append(y)

        x = dynamics.f(x, u, t)

    return xs, us, ys