from numpy.typing import NDArray
from typing import List, Tuple

import numpy as np

from lib.base import Gaussian, Dynamics, MeasurementModel, InputPrior
from lib.utils import diag_regularise, pd_svd_inv



# TODO: Handle passing `x1` as a prior over the first state in the sequence
class iLQR():
    def __init__(self, dynamics: Dynamics, meas_model: MeasurementModel,
                 input_prior: InputPrior, ys: List[NDArray],
                 ls_gamma: float = 0.5, ls_beta: float = 0.5,
                 ls_iters: int | None = 25):
        self.dynamics = dynamics
        self.meas_model = meas_model
        self.input_prior = input_prior
        self.ys = ys
        self.ls_gamma = ls_gamma
        self.ls_beta = ls_beta
        self.ls_iters = ls_iters
        self.T = len(self.ys)

    def _rollout(self, us_init: List[NDArray]) -> Tuple[List[Gaussian]]:
        """Computes the initial guess of the optimal state trajectory.
        
        Returns the states and inputs as Gaussian objects with placeholder
        covariances of 0, as well as including the state x_0 and input u_0 used
        to set the initial condition x_1.

        Parameters
        - `us_init: List[NDArray]`
            The initial guess for the optimal control inputs [u_1, ..., u_T]

        Returns
        - `Tuple` of...
            - `xs: List[Gaussian]`
                The optimal state trajectory [x_0, x_1, ..., x_T]
            - `us: List[Gaussian]`
                The optimal control inputs [u_0, u_1, ..., u_T], where u_0 sets
                the state x_1 and u_1, ..., u_T are Gaussians matching the
                passed `us_init`
        """

        xs = []
        us = []

        # Placeholder covariances; these are never used in calculations
        cov_x = np.eye(self.dynamics.Nx)
        cov_u = np.eye(self.dynamics.Nu)

        # Timestep t = 0
        x = Gaussian(np.zeros(self.dynamics.Nx), cov_x)
        u0 = np.zeros(self.dynamics.Nu)

        for t, u_mean in enumerate([u0] + us_init):
            # Convert input to Gaussian
            u = Gaussian(u_mean, cov_u)
            
            xs.append(x)
            us.append(u)

            # Compute next state
            x = Gaussian(self.dynamics.f(x.mean, u_mean, t), cov_x)

        return xs, us

    def _cost(self, xs: List[Gaussian], us: List[Gaussian]):

        cost = -self.input_prior.ll(us[0].mean, 0)
        for t in range(1, self.T+1):
            cost -= self.meas_model.ll(xs[t].mean, self.ys[t-1], t)
            cost -= self.input_prior.ll(us[t].mean, t)

        return cost

    def _backward_pass(self, F_xs: NDArray, F_us: NDArray, c_xs: NDArray,
                       c_us: NDArray, C_xxs: NDArray, C_uxs: NDArray,
                       C_uus: NDArray) -> Tuple[NDArray]:
        """Computes the linear feedback terms and relevant Hessians at each
        timestep.

        Parameters
        - `F_xs: NDArray`
            The derivatives df/dx of the dynamics at each timestep
        - `F_us: NDArray`
            The derivatives df/du of the dynamics at each timestep
        - `c_xs: NDArray`
            The derivatives dc/dx of the cost at each timestep
        - `c_us: NDArray`
            The derivatives dc/du of the cost at each timestep
        - `C_xxs: NDArray`
            The Hessians d2c/dx2 of the cost at each timestep
        - `C_uxs: NDArray`
            The Hessians d2c/dxdu of the cost at each timestep
        - `C_uus: NDArray`
            The Hessians d2c/du2 of the cost at each timestep

        Returns
        - `Tuple` of...
            - `ks: NDArray`
                The linear feedback vectors
            - `Ks: NDArray`
                The linear feedback gain matrices
            - `Vs: NDArray`
                The Hessians of the value function
            - `Q_uu_invs: NDArray`
                The inverses of the cost-to-go input matrix
        """
        
        ks = np.zeros((self.T+1, self.dynamics.Nu))
        Ks = np.zeros((self.T+1, self.dynamics.Nu, self.dynamics.Nx))
        Vs = np.zeros((self.T+1, self.dynamics.Nx, self.dynamics.Nx))
        Q_uu_invs = np.zeros((self.T+1, self.dynamics.Nu, self.dynamics.Nu))

        V = C_xxs[-1]
        v = c_xs[-1]
        Vs[-1] = V
        Q_uu_invs[-1] = pd_svd_inv(C_uus[-1] + F_us[-1].T @ V @ F_us[-1])

        for t in range(self.T-1, -1, -1):
            Q_xx = C_xxs[t] + F_xs[t].T @ V @ F_xs[t]
            # Q_ux = C_uxs[t] + F_us[t].T @ V @ F_xs[t]
            Q_ux = F_us[t].T @ V @ F_xs[t]
            Q_uu = C_uus[t] + F_us[t].T @ V @ F_us[t]
            q_x = c_xs[t] + F_xs[t].T @ v
            q_u = c_us[t] + F_us[t].T @ v
            Q_uu_inv = pd_svd_inv(Q_uu)

            K = -Q_uu_inv @ Q_ux
            k = -Q_uu_inv @ q_u

            V = Q_xx + K.T @ Q_ux
            v = q_x + K.T @ q_u

            # Enforce symmetry of the important bits
            Q_xx = (Q_xx + Q_xx.T)/2
            Q_uu = (Q_uu + Q_uu.T)/2
            V = (V + V.T)/2

            ks[t] = k
            Ks[t] = K
            Vs[t] = V
            Q_uu_invs[t] = Q_uu_inv

        return ks, Ks, Vs, Q_uu_invs

    def _compute_trajectory(self, a: float, xs_old: List[Gaussian],
                            us_old: List[Gaussian], ks: NDArray, Ks: NDArray,
                            Vs: NDArray, Q_uu_invs: NDArray, F_xs: NDArray,
                            F_us: NDArray, C_xxs: NDArray,
                            C_uus: NDArray) -> Tuple[List[Gaussian]]:
        # This `x` has a placeholder covariance which is never used
        x = Gaussian(np.zeros(self.dynamics.Nx), np.eye(self.dynamics.Nx))
        u = Gaussian(us_old[0].mean + a * ks[0], Q_uu_invs[0])

        xs = [x]
        us = [u]

        F_x_inv = np.linalg.inv(F_xs[0])
        P = F_us[0] @ u.cov @ F_us[0].T
        P = (P + P.T)/2
        P1 = F_x_inv.T @ (P + C_xxs[0]) @ F_x_inv
        P2 = pd_svd_inv(C_uus[0] + F_us[0].T @ P1 @ F_us[0])

        for t in range(1, self.T+1):
            P = P1 - P1 @ F_us[t-1] @ P2 @ F_us[t-1].T @ P1.T
            P = (P + P.T)/2
            F_x_inv = np.linalg.inv(F_xs[t])
            P1 = F_x_inv.T @ (P + C_xxs[t]) @ F_x_inv
            P2 = pd_svd_inv(C_uus[t] + F_us[t].T @ P1 @ F_us[t])

            x = Gaussian(
                self.dynamics.f(x.mean, u.mean, t-1),
                pd_svd_inv(P + Vs[t])
            )
            u = Gaussian(
                us_old[t].mean + Ks[t] @ (x.mean - xs_old[t].mean) + \
                    a * ks[t],
                Ks[t] @ x.cov @ Ks[t].T + Q_uu_invs[t]
            )

            xs.append(x)
            us.append(u)

        return xs, us

    def _forward_pass(self, xs_old: List[Gaussian], us_old: List[Gaussian],
                      ks: NDArray, Ks: NDArray, Vs: NDArray, Q_uu_invs: NDArray,
                      F_xs: NDArray, F_us: NDArray, C_xxs: NDArray,
                      C_uus: NDArray) -> Tuple:
        """Performs an iLQR forwards pass to compute the new optimal states and
        inputs.
        
        Parameters
        - `xs_old: List[Gaussian`
            The current guess for the optimal state trajectory
        - `us_old: List[Gaussian`
            The current guess for the optimal control inputs
        - `ks: NDArray`
            Linear feedback vectors
        - `Ks: NDArray`
            Linear feedback gain matrices
        - `Vs: NDArray`
            The Hessians of the value function
        - `Q_uu_invs: NDArray`
            The inverses of the cost-to-go input gain matrices
        - `F_xs: NDArray`
            The derivatives df/dx of the dynamics at each timestep
        - `F_us: NDArray`
            The derivatives df/du of the dynamics at each timestep
        - `C_xxs: NDArray`
            The Hessians d2c/dx2 of the cost at each timestep
        - `C_uus: NDArray`
            The Hessians d2c/du2 of the cost at each timestep

        Returns
        - `Tuple` of...
            - `xs_new: List[Gaussian]`
                The updated estimate of the optimal state trajectory
            - `us_new: List[Gaussian]`
                The updated estimate of the optimal control inputs
            - `cost: float`
                The cost of updated optimal trajectory
        """
        
        old_cost = self._cost(xs_old, us_old)

        if self.ls_iters == None:
            xs, us = self._compute_trajectory(1.0, xs_old, us_old, ks, Ks, Vs,
                                              Q_uu_invs, F_xs, F_us, C_xxs,
                                              C_uus)

            return xs, us, self._cost(xs, us)

        a = 1.0
        for n in range(self.ls_iters + 1):
            # Force convergence if no improvement can be found
            if n == self.ls_iters:
                a = 0.0

            xs, us = self._compute_trajectory(a, xs_old, us_old, ks, Ks, Vs,
                                              Q_uu_invs, F_xs, F_us, C_xxs,
                                              C_uus)

            # Check linesearch convergence
            cost = self._cost(xs, us)
            if cost <= old_cost:
                break

            a *= self.ls_gamma

        return xs, us, cost
    
    def __call__(self, us_init: List[NDArray], print_iters: bool = False,
                 tol: float = 1e-4, num_iters: int = 100) -> Tuple[List[Gaussian]]:
        """Runs iLQR to find the states and inputs that minimise the given cost
        function.

        Parameters:
        - `us_init: List[NDArray]`
            Initial guess of the optimal control inputs [u_1, ..., u_T]
        - `tol: float = 1e-6`
            Difference in cost function value between consecutiev iterations
            needed for convergence
        - `num_iters: int = 100`
            Number of iterations of iLQR before termination
        - `print_iters: bool = False`
            Whether to print the cost after each iteration
            
        Returns:
        - `Tuple` of...
            - `xs_new: List[Gaussian]`
                The optimal state trajectory [x_1, ..., x_T] found by iLQR
            - `us_new: List[Gaussian]`
                The optimal control inputs [u_1, ..., u_T] found by iLQR
        """

        xs, us = self._rollout(us_init)
        cost = self._cost(xs, us)
        if print_iters:
            print(f"iLQR 0/{num_iters}: Cost = {cost}")

        # NOTE: Maybe better to hold these as private attributes?
        F_xs = np.empty((self.T+1, self.dynamics.Nx, self.dynamics.Nx))
        F_us = np.empty((self.T+1, self.dynamics.Nx, self.dynamics.Nu))
        c_xs = np.empty((self.T+1, self.dynamics.Nx))
        c_us = np.empty((self.T+1, self.dynamics.Nu))
        C_xxs = np.empty((self.T+1, self.dynamics.Nx, self.dynamics.Nx))
        C_uxs = np.empty((self.T+1, self.dynamics.Nu, self.dynamics.Nx))
        C_uus = np.empty((self.T+1, self.dynamics.Nu, self.dynamics.Nu))

        for i in range(num_iters):
            for t in range(self.T+1):
                F_xs[t] = self.dynamics.df_dx(xs[t].mean, us[t].mean, t)
                F_us[t] = self.dynamics.df_du(xs[t].mean, us[t].mean, t)

                if t == 0:
                    # Pass fake observations of y = 0
                    y = np.zeros(self.meas_model.Ny)
                    c_xs[t] = -self.meas_model.dll(xs[t].mean, y, t)
                    C_xxs[t] = diag_regularise(
                        -self.meas_model.d2ll(xs[t].mean, y, t))
                else:
                    c_xs[t] = -self.meas_model.dll(xs[t].mean, self.ys[t-1], t)
                    C_xxs[t] = diag_regularise(
                        -self.meas_model.d2ll(xs[t].mean, self.ys[t-1], t))
                c_us[t] = -self.input_prior.dll(us[t].mean, t)
                C_uus[t] = diag_regularise(
                    -self.input_prior.d2ll(us[t].mean, t))
                # C_uxs[t] = -self.model.d2c_dux(xs[t].mean, us[t].mean, t)

            ks, Ks, Vs, Q_uu_invs = self._backward_pass(F_xs, F_us, c_xs, c_us,
                                                        C_xxs, C_uxs, C_uus)

            xs_new, us_new, new_cost = self._forward_pass(xs, us, ks, Ks, Vs,
                                                          Q_uu_invs, F_xs, F_us,
                                                          C_xxs, C_uus)

            if print_iters:
                print(f"iLQR {i+1}/{num_iters}: Cost = {new_cost}")

            # Check convergence
            if cost - new_cost <= tol:
                break

            xs = xs_new
            us = us_new
            cost = new_cost

        # Don't return x_0 and u_0
        return xs_new[1:], us_new[1:]



# TODO: Handle passing `x1_prior` over the first state of the trajectory
class LQR(iLQR):
    """Class implementation of LQR. Finds the optimal trajectory of states and
    inputs through the `system` that minimises the `cost` function."""

    def __init__(self, dynamics: Dynamics, meas_model: MeasurementModel,
                 input_prior: InputPrior, ys: List[NDArray]):
        super().__init__(dynamics, meas_model, input_prior, ys,
                         ls_iters=None)

    def __call__(self) -> Tuple[List[Gaussian]]:
        """Runs LQR to find the states and inputs that minimise the given cost
        function."""

        us_init = [np.zeros(self.dynamics.Nu) for _ in range(self.T)]
        xs, us = super().__call__(us_init, num_iters=1)

        return xs, us


