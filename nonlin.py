import autograd.numpy as np
import matplotlib.pyplot as plt

from lib.functions import sample_trajectory
from lib.models.dynamics import LinearDynamics
from lib.models.inputs import StudentPrior
from lib.models.measurements import PoissonMeasurement
from lib.lqr import iLQR
from lib.plotters import Plotter, plot_variances



# General parameters
num_steps = 100
Nx = 2
Nu = 2
Ny = 6



# Create the generative model
rho = 0.8
theta = 0.2
A = rho*np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
B = np.eye(Nu)
dynamics = LinearDynamics(Nx, Nu, A, B)

nu = 2
S = 0.1*np.ones(Nu)
t0_weight=10.0
input_prior = StudentPrior(Nu, nu, S, t0_weight)

C = np.ones((Ny, Nx))
C[:3, 1] = 0
C[3:, 0] = 0
meas_model = PoissonMeasurement(Ny, h=lambda x, t: np.exp(C@x))



# Generate ground truths
true_xs, true_us, ys = sample_trajectory(
    dynamics,
    meas_model,
    input_prior,
    num_steps
)



# Run iLQR
ilqr = iLQR(
    dynamics,
    meas_model,
    input_prior,
    ys
)
us_init = [np.random.multivariate_normal(np.zeros(Nu), 0.1*np.eye(Nu))
           for _ in range(num_steps)]
ilqr_xs, ilqr_us = ilqr(us_init, tol=1e-3, print_iters=True)



# Plot results
plotter = Plotter(Nx, Nu, Ny)
plotter.plot_states(true_xs=true_xs, ilqr_xs=ilqr_xs)
plotter.plot_inputs(true_us=true_us, ilqr_us=ilqr_us)
plotter.plot_measurements(model=meas_model, true_ys=ys, ilqr_xs=ilqr_xs)

plot_variances(ilqr_us)

plt.show()
