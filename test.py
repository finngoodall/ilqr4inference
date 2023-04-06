import matplotlib.pyplot as plt
import numpy as np

from lib.functions import sample_trajectory
from lib.lqr import iLQR
from lib.models.dynamics import LinearDynamics
from lib.models.priors import GaussianPrior
from lib.models.measurements import GaussianMeasurement
from lib.plotters import Plotter



# General parameters
num_steps = 100
Nx = 2
Nu = 2
Ny = 2



# Create the generative model
rho = 0.8
theta = 0.2
A = rho*np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
B = np.eye(Nu)
u_cov = 0.1*np.eye(2)
y_cov = 0.5*np.eye(2)

dynamics = LinearDynamics(Nx, Nu, A, B)
input_prior = GaussianPrior(Nu, u_cov)
meas_model = GaussianMeasurement(Ny, y_cov)



# Sample a trajcetory generative model
true_xs, true_us, ys = sample_trajectory(
    dynamics,
    meas_model,
    input_prior,
    num_steps
)



# Initialise and run new iLQR
ilqr = iLQR(
    dynamics=dynamics,
    meas_model=meas_model,
    input_prior=input_prior,
    ys=ys
)
us_init = [np.zeros(2) for _ in range(100)]
ilqr_new_xs, ilqr_new_us = ilqr(us_init, print_iters=True)



# Show results
plotter = Plotter(Nx, Nu, Ny)
plotter.plot_states(true_xs=true_xs, ilqr_xs=ilqr_new_xs)
plotter.plot_inputs(true_us=true_us, ilqr_us=ilqr_new_us)
plotter.plot_measurements(model=meas_model, true_ys=ys, ilqr_xs=ilqr_new_xs)

plt.show()