import matplotlib.pyplot as plt
import autograd.numpy as np

from lib.functions import sample_trajectory
from lib.kalman import KalmanSmoother
from lib.lqr import LQR, iLQR
from lib.models.dynamics import LinearDynamics
from lib.models.priors import GaussianPrior
from lib.models.measurements import LinearGaussianMeasurement
from lib.plotters import Plotter



# General parameters
num_steps = 50
Nx = 1
Nu = 1
Ny = 1



# Build generative model
A = np.array([[0.9]])
B = np.array([[0.5]])
C = np.array([[1]])
u_cov = np.eye(Nu)
y_cov = np.array([[0.1]])

dynamics = LinearDynamics(Nx, Nu, A, B)
meas_model = LinearGaussianMeasurement(Ny, y_cov, C)
input_prior = GaussianPrior(Nu, np.zeros(Nu), u_cov)
x1_prior = GaussianPrior(Nx, np.zeros(Nx), np.eye(Nx))



# Generate the true states and measurements
true_xs, true_us, ys = sample_trajectory(
    dynamics,
    meas_model,
    input_prior,
    num_steps,
    x1_prior.sample(0)
)



# Run Kalman smoother, LQR and iLQR
kalman_smoother = KalmanSmoother(A, B, C, u_cov, y_cov)
kalman_xs, kalman_us = kalman_smoother(ys, x1_prior)

lqr = LQR(dynamics, meas_model, input_prior, ys, x1_prior)
lqr_xs, lqr_us = lqr()

ilqr = iLQR(dynamics, meas_model, input_prior, ys, x1_prior)
us_init = [input_prior.sample(t) for t in range(num_steps)]
ilqr_xs, ilqr_us = ilqr(us_init, print_iters=True)



# Plot results
plotter = Plotter(Nx, Nu, Ny)
plotter.plot_states(true_xs, kalman_xs, lqr_xs, ilqr_xs)
plotter.plot_inputs(true_us, kalman_us, lqr_us, ilqr_us)
plotter.plot_measurements(meas_model, ys, kalman_xs, lqr_xs, ilqr_xs)

plt.show()