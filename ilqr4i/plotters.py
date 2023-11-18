from numpy.typing import NDArray
from typing import List

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import autograd.numpy as np

from .base import Gaussian
from .models.measurements import MeasurementModel

# Paul Tor - "vibrant" (colour-blind friendly)
# https://personal.sron.nl/~pault/
colour_map = {
    "orange" : "#EE7733",
    "blue" : "#0077BB",
    "cyan" : "#33BBEE",
    "magenta" : "#EE3377",
    "red" : "#CC3311",
    "teal" : "#009988",
    "grey" : "#BBBBBB"
}



def plot_spikes(ys: List[NDArray]) -> None:
    """Plots the spike trains of `ys`. Each vector in `ys[t+1][i]` contains the
    number of spikes record at time `t` by neuron `i`."""

    # Get the time indicies when a spike was recorded for each neuron
    ys_ndarray = np.array(ys)
    spike_times = [1 + np.argwhere(y > 0).flatten() for y in ys_ndarray.T]

    # Plot the spikes
    fig, ax = plt.subplots(1, 1)
    
    ax.eventplot(spike_times, linelengths=0.8, linewidths=1.0, colors="k")
    ax.set_ylabel("Neuron")
    ax.set_xlabel("Time Index [a.u.]")
    ax.set_yticklabels([])

    plt.show()



def plot_variances(gaussians: List[Gaussian]) -> None:
    """Plot the variances of the given gaussian objects at each timestep."""

    variances = np.zeros((len(gaussians), gaussians[0].Ndims))
    for t, g in enumerate(gaussians):
        variances[t] = np.diag(g.cov)

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    for i in range(gaussians[0].Ndims):
        ax.plot(np.sqrt(variances[:, i]), label=f"$\sigma_{i}^2$")
    ax.legend()



def plot_groundtruths(
        axes: plt.Axes,
        states: List[NDArray],
        markers: bool = False
    ):
    
    means = np.array(states)

    for i, ax in enumerate(axes):
        if markers:
            ax.scatter(
                np.arange(len(states)),
                means[:, i],
                c="k",
                marker=".",
                label="truths",
                zorder=2
            )
        else:
            ax.plot(
                means[:, i],
                c="k",
                ls="--",
                label="truths"
            )



def plot_gaussians(
        axes: plt.Axes,
        states: List[Gaussian],
        c: str,
        label: str,
        ls: str = "-",
        sd: float = 2.0,
    ):
    
    T = len(states)
    N = states[0].Ndims

    means = np.zeros((T, N))
    stdevs = np.zeros((T, N))
    for t in range(T):
        means[t] = states[t].mean
        stdevs[t] = np.sqrt(np.diag(states[t].cov))

    for i, ax in enumerate(axes):
        ax.plot(means[:, i], c=c, ls=ls, label=label)
        ax.fill_between(
            x=range(T),
            y1=means[:, i] + sd*stdevs[:, i],
            y2=means[:, i] - sd*stdevs[:, i],
            color=c,
            alpha=0.2
        )



def plot_measurements(
        axes: plt.Axes,
        model: MeasurementModel,
        states: List[Gaussian],
        c: str,
        label: str,
        ls: str = "-",
        sd: float = 2.0,
    ):

    T = len(states)
    N = model.Ny

    means = np.zeros((T, N))
    uppers = np.zeros((T, N))
    lowers = np.zeros((T, N))
    for t in range(T):
        means[t] = model.h(states[t].mean, t)
        uppers[t] = model.h(
            states[t].mean + sd*np.sqrt(np.diag(states[t].cov)),
            t
        )
        lowers[t] = model.h(
            states[t].mean - sd*np.sqrt(np.diag(states[t].cov)),
            t
        )

    for i, ax in enumerate(axes):
        ax.plot(means[:, i], c=c, ls=ls, label=label)
        ax.fill_between(
            x=range(T),
            y1=uppers[:, i],
            y2=lowers[:, i],
            color=c,
            alpha=0.2
        )