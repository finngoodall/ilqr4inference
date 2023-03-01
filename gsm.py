import autograd.numpy as np
import matplotlib.pyplot as plt

from lib.functions import sample_trajectory
from lib.lqr import iLQR
from lib.models.dynamics import GSMDynamics
from lib.models.inputs import GaussianPrior
from lib.models.measurements import GSMExpMeasurement, GaussianMeasurement
from lib.plotters import Plotter
from lib.utils import requad



# General parameters
num_steps = 250
Nx = 5
Nu = 3
Ny = 12

"""
Breaking the degeneracy
Try:
- Passing an initial setting `x1_prior`
- Ny > Nx
- A = np.random(...)
- Changing tau_x for each latent
- Changing the noise for each measurement
"""

# Create the dynamics and distributions
dt = 0.05
tau_xs = np.random.normal(loc=0.5, scale=0.05, size=Nx-1)
tau_c = 10.0
A = np.random.uniform(-1, 1, size=(Ny, Nx-1))
B = np.random.uniform(-3, 3, size=(Nx, Nu))
u_cov = np.eye(Nu)
u_cov[-1, -1] = 0.05
y_cov = np.diag(np.random.uniform(0.1, 0.5, size=Ny))

dynamics = GSMDynamics(Nx, Nu, B, dt, tau_xs, tau_c)
# meas_model = GSMExpMeasurement(Ny, A, y_cov)
# meas_model = GaussianMeasurement(Ny, y_cov, lambda x, t: requad(0.2*x[-1])*A@x[:-1])
meas_model = GaussianMeasurement(Ny, y_cov, lambda x, t: requad(x[-1])*A@x[:-1])
# meas_model = GaussianMeasurement(Ny, y_cov, lambda x, t: requad(5*x[-1])*A@x[:-1])
# meas_model = GaussianMeasurement(Ny, y_cov, lambda x, t: np.exp(x[-1])*A@x[:-1])
# meas_model = GaussianMeasurement(Ny, y_cov, h=lambda x, t: x[-1]*A@x[:-1])
input_prior = GaussianPrior(Nu, u_cov, t0_weight=1/dt)



# Generate ground truth
true_xs, true_us, ys = sample_trajectory(
    dynamics=dynamics,
    meas_model=meas_model,
    input_prior=input_prior,
    T=num_steps
)



# Run iLQR
ilqr = iLQR(
    dynamics=dynamics,
    meas_model=meas_model,
    input_prior=input_prior,
    ys=ys
)
us_init = [np.random.multivariate_normal(np.zeros(Nu), u_cov)
           for _ in range(num_steps)]
ilqr_xs, ilqr_us = ilqr(us_init, tol=1e-3, print_iters=True)



def cost(xs, us):
    cost = 0.0
    for t in range(1, num_steps+1):
        cost -= meas_model.ll(xs[t-1], ys[t-1], t)
        cost -= input_prior.ll(us[t-1], t)
    return cost
print(f"ilQR cost without u0: {cost([x.mean for x in ilqr_xs], [u.mean for u in ilqr_us])}")
print(f"Cost of truths: {cost(true_xs, true_us)}")



# Plot results
plotter = Plotter(Nx, Nu, Ny)
plotter.plot_states(true_xs=true_xs, ilqr_xs=ilqr_xs)
plotter.plot_inputs(true_us=true_us, ilqr_us=ilqr_us)
plotter.plot_measurements(model=meas_model, true_ys=ys, ilqr_xs=ilqr_xs)

fig, ax = plt.subplots(1, 1)
ax.plot([requad(x[-1]) for x in true_xs], label="True")
ax.plot([requad(x.mean[-1]) for x in ilqr_xs], label="iLQR")
plt.legend()

plt.show()

