from numpy.typing import NDArray
from typing import List, Tuple

import numpy as np

from lib.base import Gaussian, Dynamics, MeasurementModel, InputPrior
from lib.models.dynamics import LinearDynamics
from lib.models.measurements import GaussianMeasurement
from lib.models.inputs import GaussianPrior
from lib.utils import diag_regularise, pd_svd_inv



class iLQR():
    """Class implementation of iLQR for inference."""

    def __init__(
            self,
            dynamics: Dynamics,
            meas_model: MeasurementModel,
            input_prior: InputPrior,
            ys: List[NDArray],
            x0_prior: Gaussian = None,
            ls_gamma: float = 0.5,
            ls_iters: int | None = 25
        ) -> None:
        """Contruct the iLQR instance.
        
        Parameters
        - `dynamics`:
            The dynamics of the generative model
        - `meas_model`:
            The measurement model of the generative model
        - `input_prior`:
            The prior over the generative model's control inputs
        - `ys`:
            The seen observations
        - `x0_prior = None`:
            The prior over the initial state of the trajectory
        - `ls_gamma = 0.5`:
            The decrease factor of the linesearch scaling factor
        - `ls_iters = 25`:
            The number of linesearch iterations before termination. Using
            `ls_iters = None` disables the linesearch.
        """

        self.dynamics = dynamics
        self.meas_model = meas_model
        self.input_prior = input_prior
        self.ys = ys
        self.x0_prior = x0_prior
        self.ls_gamma = ls_gamma
        self.ls_iters = ls_iters
        self.T = len(self.ys)

    def _rollout(
            self,
            us_init: List[NDArray]
        ) -> Tuple[List[Gaussian]]:
        """Compute the initial guess of the optimal state trajectory. Returns
        the states and inputs as lists of `Gaussian` objects with placeholder
        identity covariances.

        Parameters
        - `us_init`:
            The initial guess for the optimal control inputs

        Returns
        - `Tuple` of...
            - `xs`:
                The state trajectory
            - `us`:
                The control inputs with means equal to `us_init`
        """

        # Placeholder covariances; these are never used in calculations
        cov_x = np.eye(self.dynamics.Nx)
        cov_u = np.eye(self.dynamics.Nu)

        xs = []
        us = []

        if self.x0_prior:
            x = self.x0_prior
        else:
            x = Gaussian(np.zeros(self.dynamics.Nx), cov_x)

        for t, u_mean in enumerate(us_init):
            u = Gaussian(u_mean, cov_u)
            
            xs.append(x)
            us.append(u)

            x = Gaussian(self.dynamics.f(x.mean, u_mean, t), cov_x)

        return xs, us

    def _cost(
            self,
            xs: List[Gaussian],
            us: List[Gaussian]
        ) -> float:
        """Calculate the cost of the passed trajectory.
        
        Parameters
        - `xs`:
            The state trajectory
        - `us`:
            The control inputs

        Returns
        - `cost`
            The cost of the trajectory
        """

        cost = 0
        if self.x0_prior:
            v = xs[0].mean - self.x0_prior.mean
            cost += 0.5 * v.T @ pd_svd_inv(self.x0_prior.cov) @ v

        for t in range(self.T):
            cost -= self.meas_model.ll(xs[t].mean, self.ys[t], t)
            cost -= self.input_prior.ll(us[t].mean, t)

        return cost

    def _backward_pass(
            self,
            F_xs: NDArray,
            F_us: NDArray,
            c_xs: NDArray,
            c_us: NDArray,
            C_xxs: NDArray,
            C_uus: NDArray
        ) -> Tuple[NDArray]:
        """Compute the linear feedback terms, optimal cost-to-go terms at each
        timestep.

        Parameters
        - `F_xs`:`
            The dynamic's state Jacobians
        - `F_us`:
            The dynamic's input Jacobians
        - `c_xs`:
            The cost function's state Jacobians
        - `c_us`:
            The cost function's input Jacobians
        - `C_xxs`:
            The cost function's state Hessians
        - `C_uus`:
            The cost function's input Hessians

        Returns
        - `Tuple` of...
            - `ks`:
                The linear feedback vectors
            - `Ks`:
                The linear feedback gain matrices
            - `v0`:
                The optimal cost-to-go vector at timestep 0
            - `Vs`:
                The optimal cost-to-go Hessians
            - `Q_uu_invs`:
                The inverses of the cost-to-go input Hessian
        """
        
        ks = np.zeros((self.T, self.dynamics.Nu))
        Ks = np.zeros((self.T, self.dynamics.Nu, self.dynamics.Nx))
        Vs = np.zeros((self.T, self.dynamics.Nx, self.dynamics.Nx))
        Q_uu_invs = np.zeros((self.T, self.dynamics.Nu, self.dynamics.Nu))

        V = C_xxs[-1]
        v = c_xs[-1]
        Vs[-1] = V
        Q_uu_invs[-1] = pd_svd_inv(C_uus[-1] + F_us[-1].T @ V @ F_us[-1])

        for t in range(self.T-2, -1, -1):
            Q_xx = C_xxs[t] + F_xs[t].T @ V @ F_xs[t]
            Q_ux = F_us[t].T @ V @ F_xs[t]
            Q_uu = C_uus[t] + F_us[t].T @ V @ F_us[t]
            q_x = c_xs[t] + F_xs[t].T @ v
            q_u = c_us[t] + F_us[t].T @ v
            Q_uu_inv = pd_svd_inv(Q_uu)

            K = -Q_uu_inv @ Q_ux
            k = -Q_uu_inv @ q_u

            V = Q_xx + K.T @ Q_ux
            v = q_x + K.T @ q_u

            # Enforce symmetry of the important matrices
            Q_xx = (Q_xx + Q_xx.T)/2
            Q_uu = (Q_uu + Q_uu.T)/2
            V = (V + V.T)/2

            ks[t] = k
            Ks[t] = K
            Vs[t] = V
            Q_uu_invs[t] = Q_uu_inv

        return ks, Ks, v, Vs, Q_uu_invs

    def _compute_trajectory(
            self,
            a: float,
            xs_old: List[Gaussian],
            us_old: List[Gaussian],
            ks: NDArray,
            Ks: NDArray,
            v1: NDArray,
            Vs: NDArray,
            Q_uu_invs: NDArray,
            F_xs: NDArray,
            F_us: NDArray,
            C_xxs: NDArray,
            C_uus: NDArray
        ) -> Tuple[List[Gaussian]]:
        """Compute the optimal state trajectory and control inputs.
        
        Parameters
        - `a`:
            The linesearch scaling factor on the control inputs
        - `xs_old`:
            The old state trajectory to be updated
        - `us_old`:
            The old state trajectory to be updated
        - `ks`:
            The linear feedback vectors for the control inputs
        - `Ks`:
            The linear feedback gains for the control inputs
        - `v1`:
            The optimal cost-to-go vector at timestep 0
        - `Vs`:
            The optimal cost-to-go Hessians
        - `Q_uu_invs`:
            The inverses of the cost-to-go input Hessian
        - `F_xs`:
            The dynamic's state Jacobians
        - `F_us`:
            The dynamic's input Jacobians
        - `C_xxs`:
            The cost function's state Hessians 
        - `C_uus`:
            The cost function's input Hessians
        
        Returns
        - `xs`:
            The updated optimal state trajectory
        - `us`:
            The updated optimal control inputs
        """
        
        xs = []
        us = []

        for t in range(self.T):
            if t == 0:
                if self.x0_prior:
                    P = pd_svd_inv(self.x0_prior.cov)
                    x_cov = pd_svd_inv(P + Vs[t])
                    x = Gaussian(
                        xs_old[0].mean + a*x_cov@(P@self.x0_prior.mean - v1),
                        x_cov
                    )
                else:
                    P = np.zeros((self.dynamics.Nx, self.dynamics.Nx))
                    x = Gaussian(
                        xs_old[0].mean - a*pd_svd_inv(Vs[t])@v1,
                        pd_svd_inv(Vs[t])
                    )
            else:
                x = Gaussian(
                    self.dynamics.f(x.mean, u.mean, t),
                    pd_svd_inv(P + Vs[t])
                )

            u = Gaussian(
                us_old[t].mean + Ks[t]@(x.mean - xs_old[t].mean) + a*ks[t],
                Ks[t]@x.cov@Ks[t].T + Q_uu_invs[t]
            )

            xs.append(x)
            us.append(u)

            F_x_inv = np.linalg.inv(F_xs[t])
            P1 = F_x_inv.T @ (P + C_xxs[t]) @ F_x_inv
            P2 = pd_svd_inv(C_uus[t] + F_us[t].T@P1@F_us[t])
            P = P1 - P1@F_us[t]@P2@F_us[t].T@P1.T
            P = (P + P.T)/2

        return xs, us

    def _forward_pass(
            self,
            xs_old: List[Gaussian],
            us_old: List[Gaussian],
            ks: NDArray,
            Ks: NDArray,
            v0: NDArray,
            Vs: NDArray,
            Q_uu_invs: NDArray,
            F_xs: NDArray,
            F_us: NDArray,
            C_xxs: NDArray,
            C_uus: NDArray
        ) -> Tuple[List[Gaussian], List[Gaussian], float]:
        """Compute the new optimal state trajectory and control inputs.
        
        Unless otherwise specified by `self.ls_iters`, this performs a
        linesearch to ensure a decrease in the cost of the trajectory. If no
        decrease can be found after `self.ls_iters` searches, then the
        linesearch terminates and `xs_old`, `us_old` are returned.
        
        Parameters
        - `xs_old`:
            The old state trajectory to be updated
        - `us_old`:
            The old state trajectory to be updated
        - `ks`:
            The linear feedback vectors for the control inputs
        - `Ks`:
            The linear feedback gains for the control inputs
        - `v1`:
            The optimal cost-to-go vector at timestep 0
        - `Vs`:
            The optimal cost-to-go Hessians
        - `Q_uu_invs`:
            The inverses of the cost-to-go input Hessian
        - `F_xs`:
            The dynamic's state Jacobians
        - `F_us`:
            The dynamic's input Jacobians
        - `C_xxs`:
            The cost function's state Hessians 
        - `C_uus`:
            The cost function's input Hessians
        
        Returns
        - `xs`:
            The updated optimal state trajectory
        - `us`:
            The updated optimal control inputs
        - `cost`:
            The cost of the updated trajectory
        """
        
        cost_old = self._cost(xs_old, us_old)

        trajectory_updated = False
        if self.ls_iters == None:
            xs, us = self._compute_trajectory(1.0, xs_old, us_old, ks, Ks, v0,
                                              Vs, Q_uu_invs, F_xs, F_us, C_xxs,
                                              C_uus)

            return xs, us, self._cost(xs, us)

        a = 1.0
        for _ in range(self.ls_iters + 1):
            xs, us = self._compute_trajectory(a, xs_old, us_old, ks, Ks, v0, Vs,
                                              Q_uu_invs, F_xs, F_us, C_xxs,
                                              C_uus)

            # Check linesearch convergence
            cost = self._cost(xs, us)
            if cost <= cost_old:
                trajectory_updated = True
                break

            a *= self.ls_gamma

        if not trajectory_updated:
            return xs_old, us_old, cost_old
        
        return xs, us, cost
        
    
    def __call__(
            self,
            us_init: List[NDArray],
            tol: float = 1e-4,
            num_iters: int = 100,
            print_iters: bool = False
        ) -> Tuple[List[Gaussian]]:
        """Run iLQR to find the MAP states and inputs given the generative
        model, observations and prior over the first state.

        Parameters:
        - `us_init`:
            Initial guess of the optimal control inputs
        - `tol = 1e-4`:
            Difference in cost function value between consecutive iterations
            needed for convergence
        - `num_iters = 100`:
            Number of iterations of iLQR before termination
        - `print_iters = False`
            Whether to print the cost after each iLQR iteration
            
        Returns:
        - `Tuple` of...
            - `xs_new`:
                The MAP state trajectory
            - `us_new`:
                The MAP control inputs
        """

        xs, us = self._rollout(us_init)
        cost = self._cost(xs, us)
        if print_iters:
            print(f"iLQR 0/{num_iters}: Cost = {cost}")

        # NOTE: Maybe better to hold these as private attributes?
        F_xs = np.empty((self.T, self.dynamics.Nx, self.dynamics.Nx))
        F_us = np.empty((self.T, self.dynamics.Nx, self.dynamics.Nu))
        c_xs = np.empty((self.T, self.dynamics.Nx))
        c_us = np.empty((self.T, self.dynamics.Nu))
        C_xxs = np.empty((self.T, self.dynamics.Nx, self.dynamics.Nx))
        C_uus = np.empty((self.T, self.dynamics.Nu, self.dynamics.Nu))

        for i in range(num_iters):
            for t in range(self.T):
                F_xs[t] = self.dynamics.df_dx(xs[t].mean, us[t].mean, t)
                F_us[t] = self.dynamics.df_du(xs[t].mean, us[t].mean, t)

                c_xs[t] = -self.meas_model.dll(xs[t].mean, self.ys[t], t)
                C_xxs[t] = diag_regularise(
                    -self.meas_model.d2ll(xs[t].mean, self.ys[t], t))
                c_us[t] = -self.input_prior.dll(us[t].mean, t)
                C_uus[t] = diag_regularise(
                    -self.input_prior.d2ll(us[t].mean, t))

            ks, Ks, v0, Vs, Q_uu_invs = self._backward_pass(F_xs, F_us, c_xs,
                                                            c_us, C_xxs, C_uus)

            xs_new, us_new, new_cost = self._forward_pass(xs, us, ks, Ks, v0,
                                                          Vs, Q_uu_invs, F_xs,
                                                          F_us, C_xxs, C_uus)

            if print_iters:
                print(f"iLQR {i+1}/{num_iters}: Cost = {new_cost}")

            if cost - new_cost <= tol:
                break

            xs = xs_new
            us = us_new
            cost = new_cost

        return xs, us



class LQR(iLQR):
    """Class implementation of LQR for inference. Inherits functionality from
    iLQR."""

    def __init__(
            self,
            dynamics: LinearDynamics,
            meas_model: GaussianMeasurement,
            input_prior: GaussianPrior,
            ys: List[NDArray],
            x0_prior: Gaussian = None
        ) -> None:
        """Contruct the LQR instance.
        
        Parameters
        - `dynamics`:
            The dynamics of the generative model
        - `meas_model`:
            The measurement model of the generative model
        - `input_prior`:
            The prior over the generative model's control inputs
        - `ys`:
            The seen observations
        - `x0_prior = None`:
            The prior over the initial state of the trajectory
        """
        
        linear_model = (isinstance(dynamics, LinearDynamics)
                        and isinstance(meas_model, GaussianMeasurement)
                        and isinstance(input_prior, GaussianPrior))
        if not linear_model:
            print("Warning: LQR can only infer over linear generative models correctly.")

        super().__init__(dynamics, meas_model, input_prior, ys, x0_prior,
                         ls_iters=None)

    def __call__(self) -> Tuple[List[Gaussian]]:
        """Run LQR to find the MAP states and inputs given the generative
        model, observations and prior over the first state.
        """

        us_init = [np.zeros(self.dynamics.Nu) for _ in range(self.T)]
        return super().__call__(us_init, num_iters=1)


