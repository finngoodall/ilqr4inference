from numpy.typing import NDArray
from typing import List, Tuple

import numpy as np

from lib.base import Gaussian, Dynamics, MeasurementModel, Prior
from lib.models.dynamics import LinearDynamics
from lib.models.measurements import LinearGaussianMeasurement
from lib.models.priors import GaussianPrior
from lib.utils import diag_regularise, pd_svd_inv



class iLQR():
    """Class implementation of iLQR for inference."""

    def __init__(
            self,
            dynamics: Dynamics,
            meas_model: MeasurementModel,
            input_prior: Prior,
            ys: List[NDArray],
            x0_prior: Prior = None,
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
        ) -> NDArray:
        """Compute the initial guess of the optimal state trajectory.

        Parameters
        - `us_init`:
            The initial guess for the optimal control inputs

        Returns
        - `xs`:
            The state trajectory
        """

        xs = np.zeros((self.T, self.dynamics.Nx))

        if self.x0_prior:
            x = self.x0_prior.mean
        else:
            x = np.zeros(self.dynamics.Nx)

        for t, u in enumerate(us_init):
            xs[t] = x
            x = self.dynamics.f(x, u, t)

        return xs

    def _cost(
            self,
            xs: NDArray,
            us: NDArray
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
            cost -= self.x0_prior.ll(xs[0], 0)

        for t in range(self.T):
            cost -= self.meas_model.ll(xs[t], self.ys[t], t)
            cost -= self.input_prior.ll(us[t], t)

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
        Q_uu_invs[-1] = pd_svd_inv(C_uus[-1])

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

    def _compute_means(
            self,
            a: float,
            xs_old: NDArray,
            us_old: NDArray,
            ks: NDArray,
            Ks: NDArray,
            v0: NDArray,
            V0: NDArray
        ) -> Tuple[NDArray]:
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
        - `v0`:
            The optimal cost-to-go vector at timestep 0
        - `V0`:
            The optimal cost-to-go Hessian at timestep 0
        
        Returns
        - `xs`:
            The updated optimal state trajectory
        - `us`:
            The updated optimal control inputs
        """
        
        xs = np.zeros((self.T, self.dynamics.Nx))
        us = np.zeros((self.T, self.dynamics.Nu))

        for t in range(self.T):
            if t == 0:
                x = xs_old[0] - a*pd_svd_inv(V0)@v0
            else:
                x = self.dynamics.f(x, u, t)

            u = us_old[t] + Ks[t]@(x - xs_old[t]) + a*ks[t]

            xs[t] = x
            us[t] = u

        return xs, us

    def _compute_covs(
            self,
            xs: NDArray,
            us: NDArray,
            Ks: NDArray,
            Vs: NDArray,
            Q_uu_invs: NDArray,
            F_xs: NDArray,
            F_us: NDArray,
            C_xxs: NDArray,
            C_uus: NDArray
        ) -> Tuple[List[Gaussian]]:
        """Compute the covariances around the optimal state trajectory and
        control inputs. Returns the Gaussian distributions over the states and
        inputs.
        
        Parameters
        - `xs`:
            The optimal state trajectory means
        - `us`:
            The optimal control input means
        - `Ks`:
            The linear feedback gains for the control inputs
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
        - Tuple of...
            - `xs_with_covs`:
                Gaussian distributions around the optimal state means
            - `us_with_covs`:
                Gaussian distributions around the optimal control input means
        """

        xs_with_covs = []
        us_with_covs = []

        P = np.zeros((self.dynamics.Nx, self.dynamics.Nx))
        for t in range(self.T):
            x = Gaussian(xs[t], pd_svd_inv(P + Vs[t]))
            u = Gaussian(us[t], Ks[t] @ x.cov @ Ks[t].T + Q_uu_invs[t])
            xs_with_covs.append(x)
            us_with_covs.append(u)

            F_x_inv = np.linalg.inv(F_xs[t])
            P1 = F_x_inv.T @ (P + C_xxs[t]) @ F_x_inv
            P2 = pd_svd_inv(C_uus[t] + F_us[t].T@P1@F_us[t])
            P = P1 - P1@F_us[t]@P2@F_us[t].T@P1.T
            P = (P + P.T)/2

        return xs_with_covs, us_with_covs

    def _forward_pass(
            self,
            xs_old: NDArray,
            us_old: NDArray,
            ks: NDArray,
            Ks: NDArray,
            v0: NDArray,
            V0: NDArray
        ) -> Tuple[NDArray, NDArray, float]:
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
        - `v0`:
            The optimal cost-to-go vector at timestep 0
        - `V0`:
            The optimal cost-to-go Hessian at timestep 0
        
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
            xs, us = self._compute_means(1, xs_old, us_old, ks, Ks, v0, V0)
            return xs, us, self._cost(xs, us)

        a = 1.0
        for _ in range(self.ls_iters):
            xs, us = self._compute_means(a, xs_old, us_old, ks, Ks, v0, V0)
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

        xs = self._rollout(us_init)
        us = np.array(us_init)
        # Set final input to 0 as it cannot be determined
        us[-1] = self.input_prior.mean

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
                F_xs[t] = self.dynamics.df_dx(xs[t], us[t], t)
                F_us[t] = self.dynamics.df_du(xs[t], us[t], t)

                c_xs[t] = -self.meas_model.dll(xs[t], self.ys[t], t)
                C_xxs[t] = diag_regularise(
                    -self.meas_model.d2ll(xs[t], self.ys[t], t))
                if t == 0 and self.x0_prior:
                    c_xs[t] += -self.x0_prior.dll(xs[t], t)
                    C_xxs[t] += -self.x0_prior.d2ll(xs[t], t)
                c_us[t] = -self.input_prior.dll(us[t], t)
                C_uus[t] = diag_regularise(-self.input_prior.d2ll(us[t], t))

            ks, Ks, v0, Vs, Q_uu_invs = self._backward_pass(F_xs, F_us, c_xs,
                                                            c_us, C_xxs, C_uus)

            xs_new, us_new, new_cost = self._forward_pass(xs, us, ks, Ks, v0,
                                                          Vs[0])

            if print_iters:
                print(f"iLQR {i+1}/{num_iters}: Cost = {new_cost}")

            if cost - new_cost <= tol:
                break

            xs = xs_new
            us = us_new
            cost = new_cost

        xs, us = self._compute_covs(xs, us, Ks, Vs, Q_uu_invs, F_xs, F_us,
                                    C_xxs, C_uus)
        
        return xs, us



class LQR(iLQR):
    """Class implementation of LQR for inference. Inherits functionality from
    iLQR."""

    def __init__(
            self,
            dynamics: LinearDynamics,
            meas_model: LinearGaussianMeasurement,
            input_prior: GaussianPrior,
            ys: List[NDArray],
            x0_prior: Prior = None
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
                        and isinstance(meas_model, LinearGaussianMeasurement)
                        and isinstance(input_prior, GaussianPrior))
        if not linear_model:
            print("Warning: LQR can only infer over linear Gaussian generative models correctly.")

        super().__init__(dynamics, meas_model, input_prior, ys, x0_prior,
                         ls_iters=None)

    def __call__(self) -> Tuple[List[Gaussian]]:
        """Run LQR to find the MAP states and inputs given the generative
        model, observations and prior over the first state.
        """

        us_init = [np.zeros(self.dynamics.Nu) for _ in range(self.T)]
        return super().__call__(us_init, num_iters=1)


