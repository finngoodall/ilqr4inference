import matplotlib.pyplot as plt
import autograd.numpy as np

from lib.base import Gaussian
from lib.functions import sample_trajectory
from lib.kalman import KalmanSmoother
from lib.lqr import LQR, iLQR
from lib.models.dynamics import LinearDynamics
from lib.models.inputs import GaussianPrior
from lib.models.measurements import LinearGaussianMeasurement
from lib.plotters import Plotter



# General parameters
num_steps = 100
Nx = 2
Nu = 2
Ny = 2



# Build generative model
rho = 0.8
theta = 0.2
A = rho*np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
B = np.array([[0.7, 0.5], [-0.3, -0.5]])
C = np.array([[1.0, 1.0], [1.0, -1.0]])
u_cov = np.eye(Nu)
y_cov = 0.15*np.eye(Ny)

dynamics = LinearDynamics(Nx, Nu, A, B)
input_prior = GaussianPrior(Nu, u_cov)
meas_model = LinearGaussianMeasurement(Ny, y_cov, C)

x1_prior = Gaussian(np.zeros(Nx), np.eye(Nx))



# Generate the true states and measurements
true_xs, true_us, ys = sample_trajectory(
    dynamics,
    meas_model,
    input_prior,
    num_steps,
    np.random.multivariate_normal(x1_prior.mean, x1_prior.cov)
)



# Run Kalman smoother, LQR and iLQR
kalman_smoother = KalmanSmoother(A, B, C, u_cov, y_cov)
kalman_xs, kalman_us = kalman_smoother(ys, x1_prior)

lqr = LQR(dynamics, meas_model, input_prior, ys, x1_prior)
lqr_xs, lqr_us = lqr()

ilqr = iLQR(dynamics, meas_model, input_prior, ys, x1_prior)
us_init = [np.random.multivariate_normal(np.zeros(Nu), 0.1*np.eye(Nu))
           for _ in range(num_steps)]
ilqr_xs, ilqr_us = ilqr(us_init, print_iters=True)



# Plot results
plotter = Plotter(Nx, Nu, Ny)
plotter.plot_states(true_xs, kalman_xs, lqr_xs, ilqr_xs)
plotter.plot_inputs(true_us, kalman_us, lqr_us, ilqr_us)
plotter.plot_measurements(meas_model, ys, kalman_xs, lqr_xs, ilqr_xs)

plt.show()