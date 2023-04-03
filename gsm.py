import autograd.numpy as np
import matplotlib.pyplot as plt

from lib.functions import sample_trajectory
from lib.lqr import iLQR
from lib.models.dynamics import GSMDynamicsKnownContrast
from lib.models.inputs import GaussianPrior
from lib.models.measurements import GaussianMeasurement
from lib.plotters import Plotter, plot_variances



# General parameters
num_steps = 250
Nx = 5
Nu = 3
Ny = Nx



# Create the dynamics and distributions
B = np.random.uniform(-1, 1, size=(Nx, Nu))
tau_x = np.random.normal(loc=0.5, scale=0.05, size=Nx)
u_cov = np.eye(Nu)
meas_func = lambda x, t: 3*(np.exp(2*(np.sin(4*np.pi*t/num_steps)-1)))*x
y_cov = 0.1*np.eye(Ny)

dynamics = GSMDynamicsKnownContrast(Nx, Nu, B, tau_x=tau_x)
meas_model = GaussianMeasurement(Ny, y_cov, meas_func)
input_prior = GaussianPrior(Nu, np.zeros(Nu), u_cov)



# Generate ground truth
true_xs, true_us, ys = sample_trajectory(dynamics, meas_model, input_prior,
                                         num_steps)



# Run iLQR
ilqr = iLQR(dynamics, meas_model, input_prior, ys)
us_init = [np.random.multivariate_normal(np.zeros(Nu), u_cov)
           for _ in range(num_steps)]
ilqr_xs, ilqr_us = ilqr(us_init, tol=1e-3, print_iters=True)



# Plot results
plotter = Plotter(Nx, Nu, Ny)
plotter.plot_states(true_xs=true_xs, ilqr_xs=ilqr_xs)
plotter.plot_inputs(true_us=true_us, ilqr_us=ilqr_us)
plotter.plot_measurements(model=meas_model, true_ys=ys, ilqr_xs=ilqr_xs)

plot_variances(ilqr_us)

plt.show()

