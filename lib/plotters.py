from numpy.typing import NDArray
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import autograd.numpy as np

from lib.base import Gaussian, MeasurementModel

# Use LaTeX to make plots look pretty
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.serif"] = "cm"
plt.rcParams["font.size"] = 11



class Plotter():
    """Class to handle plotting inferred trajectories. The trajectories have
    `Nx` states, `Nu` control inputs and `Ny` measurements, and were generated
    from `system`.

    The error bars show `sd` standard deviations. To have no error bars, set
    `sd = 0.0`."""

    def __init__(self, Nx: int, Nu: int, Ny: int, sd: float = 2.0):
        self.Nx = Nx
        self.Nu = Nu
        self.Ny = Ny
        self.sd = sd

    def _initialise_axes(self, N: int, name: str,
                         symbol: str) -> Tuple[plt.Figure, plt.Axes]:
        """Creates a subplot of `N` vertically stacked axes with the given
        `name` and `symbol`."""

        fig, axs = plt.subplots(N, 1, figsize=(8, 1+2*N))

        # Allow iterating over a single subplot
        if N == 1:
            axs = [axs]

        # Formatting
        fig.suptitle(f"{name}s")
        for i, ax in enumerate(axs):
            ax.set_ylabel(f"${symbol}_{i}$")
            if i == N-1:
                ax.set_xlabel("Time Index")
            else:
                ax.set_xticklabels([])
            ax.spines["bottom"].set_linewidth(1.5)
            ax.spines["left"].set_linewidth(1.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(width=1.5, length=3)
            ax.yaxis.set_major_locator(
                MaxNLocator(integer=True, nbins="auto", symmetric=True))

        return fig, axs

    def _plot_gtruths(self, axs: plt.Axes, states: List[Gaussian],
                      markers: bool = False) -> None:
        """Plots the ground truth `states` onto `axs`. If `markers`, the ground
        truths are plotted as dots instead of lines (used for the ground truth
        measurements)."""

        for i, ax in enumerate(axs):
            if markers:
                ax.scatter(np.arange(len(states)), [s[i] for s in states],
                           c="dimgrey", marker=".", label="Truths")
            else:
                ax.plot([s[i] for s in states], c="dimgrey", ls="--",
                        label="Truths")

    def _plot_gaussians(self, axs: plt.Axes, states: List[Gaussian],
                        c: str = "k", ls: str = "-",
                        label: str = "States") -> None:
        """Plots the means and error bars of `states` onto `axs` with the passed
        styling."""

        est_means = []
        est_covs = []
        for x in states:
            est_means.append(x.mean)
            est_covs.append(x.cov)

        for i, ax in enumerate(axs):
            # Plot means
            means = [m[i] for m in est_means]
            ax.plot(means, c=c, ls=ls, label=label)

            # Plot error bars
            stdevs = [cov[i, i]**0.5 for cov in est_covs]
            ax.fill_between(
                x=np.arange(0, len(means)),
                y1=[m+self.sd*sigma for m, sigma in zip(means, stdevs)],
                y2=[m-self.sd*sigma for m, sigma in zip(means, stdevs)],
                color=c, alpha=0.2
            )

    def _plot_measurements(self, axs: plt.Axes, model: MeasurementModel,
                           states: List[Gaussian], c: str = "k", ls: str = "-",
                           label: str = "States") -> None:
        """Plots the means and error bars of the measurements onto `axs` with
        the passed styling. The measurements and covariances are calculated from
        the passed `states`."""

        # Turn the states into measurement means and upper & lower bounds
        ys = []
        uppers = []
        lowers = []
        for t, x in enumerate(states, 1):
            delta_x = self.sd * np.diag(x.cov)**0.5
            try:
                delta_y = self.sd * np.diag(model.cov)**0.5
            except AttributeError:
                delta_y = 0
            ys.append(model.h(x.mean, t))
            uppers.append(model.h(x.mean + delta_x, t) + delta_y)
            lowers.append(model.h(x.mean - delta_x, t) - delta_y)

        # Produce the plot
        for i, ax in enumerate(axs):
            # Plot means
            ax.plot([y[i] for y in ys], c=c, ls=ls, label=label)

            # Plot error bars
            ax.fill_between(
                x=np.arange(0, len(ys)),
                y1=[u[i] for u in uppers],
                y2=[l[i] for l in lowers],
                color=c, alpha=0.2
            )
            
    def plot_states(self, true_xs: List[NDArray] = None,
                    kalman_xs: List[Gaussian] = None,
                    lqr_xs: List[Gaussian] = None,
                    ilqr_xs: List[Gaussian] = None) -> None:
        """Plots the ground truth states `true_xs` as a dotted line, with the
        other states over the top. Each element in the state vector is shown
        separately."""

        self.states_fig, self.states_axs = self._initialise_axes(
            self.Nx,
            "State",
            "x"
        )

        if true_xs:
            self._plot_gtruths(self.states_axs, true_xs)
        if kalman_xs:
            self._plot_gaussians(self.states_axs, kalman_xs, c="r", ls="-",
                                 label="Kalman")
        if lqr_xs:
            self._plot_gaussians(self.states_axs, lqr_xs, c="g", ls="-",
                                 label="LQR")
        if ilqr_xs:
            self._plot_gaussians(self.states_axs, ilqr_xs, c="royalblue",
                                 ls="-", label="iLQR")    

        [ax.legend(loc="lower right") for ax in self.states_axs]

    def plot_inputs(self, true_us: List[NDArray] = None,
                    kalman_us: List[Gaussian] = None,
                    lqr_us: List[Gaussian] = None,
                    ilqr_us: List[Gaussian] = None) -> None:
        """Plots the ground truth inputs `true_us` as a dotted line, with the
        other inputs over the top. Each element in the input vector is shown
        separately."""

        self.inputs_fig, self.inputs_axs = self._initialise_axes(
            self.Nu,
            "Input",
            "u"
        )

        if true_us:
            self._plot_gtruths(self.inputs_axs, true_us)
        if kalman_us:
            self._plot_gaussians(self.inputs_axs, kalman_us, c="r", ls="-",
                                 label="Kalman")
        if lqr_us:
            self._plot_gaussians(self.inputs_axs, lqr_us, c="g", ls="-",
                                 label="LQR")
        if ilqr_us:
            self._plot_gaussians(self.inputs_axs, ilqr_us, c="royalblue",
                                 ls="-", label="iLQR")

        [ax.legend(loc="lower right") for ax in self.inputs_axs]

    def plot_measurements(self, model,
                          true_ys: List[NDArray] = None,
                          kalman_xs: List[Gaussian] = None,
                          lqr_xs: List[Gaussian] = None,
                          ilqr_xs: List[Gaussian] = None) -> None:
        """Plots the observations `true_ys` as markers, with the mean
        measurements from the passed states over the top. Each element in the
        measurement vector is shown separately."""

        self.meas_fig, self.meas_axs = self._initialise_axes(
            self.Ny,
            "Measurement",
            "y"
        )
        
        if true_ys:
            self._plot_gtruths(self.meas_axs, true_ys, markers=True)
        if kalman_xs:
            self._plot_measurements(self.meas_axs, model, kalman_xs, c="r",
                                    ls="-", label="Kalman")
        if lqr_xs:
            self._plot_measurements(self.meas_axs, model, lqr_xs, c="g",
                                    ls="-", label="LQR")
        if ilqr_xs:
            self._plot_measurements(self.meas_axs, model, ilqr_xs,
                                    c="royalblue", ls="-", label="iLQR")

        [ax.legend(loc="lower right") for ax in self.meas_axs]


        

def plot_spikes(ys: List[NDArray]) -> None:
    """Plots the spike trains of `ys`. Each vector in `ys[t+1][i]` contains the
    number of spikes record at time `t` by neuron `i`."""

    # Get the time indicies when a spike was recorded for each neuron
    ys_ndarray = np.array(ys)
    spike_times = [1 + np.argwhere(y > 0).flatten() for y in ys_ndarray.T]

    # Plot the spikes
    fig, ax = plt.subplots(1, 1)
    
    ax.eventplot(spike_times, linelengths=0.8, linewidths=2.0, colors="k")
    ax.set_ylabel("Neuron Number")
    ax.set_xlabel("Time Index [a.u.]")
    ax.set_yticks(np.arange(ys_ndarray.shape[1]))

    plt.show()



def plot_variances(gaussians: List[Gaussian]) -> None:
    """Plot the variances of the given gaussian objects at each timestep."""

    variances = np.zeros((len(gaussians), gaussians[0].Ndims))
    for t, g in enumerate(gaussians):
        variances[t] = np.diag(g.cov)

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    for i in range(gaussians[0].Ndims):
        ax.plot(variances[:, i], label=f"$\sigma_{i}^2$")
    ax.legend()
