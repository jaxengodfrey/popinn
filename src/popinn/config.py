from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple
import abc
import equinox as eqx
from jaxtyping import Array
import jax.numpy as jnp

class LossWeights(NamedTuple):
    """Weights for each loss component. Set a weight to 0 to exclude it from the total."""
    pde: float = 1.0
    ic: float = 1.0
    bc_left: float = 1.0
    bc_right: float = 1.0
    non_negative: float = 0.1
    sol: float = 1.0


class Batch(NamedTuple):
    """Data container passed from sample_fn to loss_fn each training step.

    Attributes:
        colloc_xt:  Interior collocation points, shape (2, n_pts, n_gamma).
        x_ic:       x-values for initial condition, shape (n_ic, n_gamma).
        t_bc:       t-values for boundary conditions, shape (n_bc, n_gamma).
        extras:     Any additional per-batch data, e.g. {"gamma": array}.
    """
    colloc_xt: Any
    x_ic: Any
    t_bc: Any
    extras: dict


class AbstractPhysics(eqx.Module):
    @abc.abstractmethod
    def check_params(self):
        raise NotImplementedError
    
class PhysicsConfig(AbstractPhysics):
    gamma_range: tuple
    gamma_evol: Array
    t_max: float
    n_gamma: int
    gamma_init: float
    theta: float
    nu: float

    def __init__(self, gamma_range = None, gamma_evol = None, t_max = 1., n_gamma = 50, gamma_init = 0., theta = 1., nu = 1.):
        self.gamma_range = gamma_range
        self.gamma_evol = gamma_evol
        self.t_max = t_max
        self.n_gamma = n_gamma
        self.gamma_init = gamma_init
        self.theta = theta
        self.nu = nu
        self.check_params()
    
    def check_params(self):
        if self.gamma_range is not None and self.gamma_evol is not None:
            raise ValueError('`gamma_range` and `gamma_evol` cannot both be defined. Use `gamma_range` for training a PINN parametrized in terms of gamma, use `gamma_evol` for training a PINN on a fixed value of gamma')
        if self.gamma_evol is not None:
            if isinstance(self.gamma_evol, (float, int)):
                self.gamma_evol = jnp.array([self.gamma_evol], dtype = float)
            else:
                self.gamma_evol = jnp.asarray([self.gamma_evol], dtype = float)


@dataclass
class SamplingConfig:
    """Controls collocation point sampling.

    Attributes:
        n_interior: Grid points per axis; total interior pts = n_interior².
        uniform:    If True, use a uniform grid without noise (used for L-BFGS).
    """
    n_interior: int = 100
    x_crowd: float = 1.
    uniform: bool = False
    n_batch: int = 1000



@dataclass
class AdamConfig:
    """Configuration for the Adam optimisation phase.

    Attributes:
        num_epochs:  Number of training epochs.
        lr:          Initial learning rate.
        lr_schedule: "cosine" for cosine decay, "constant" for fixed lr.
        log_every:   Print progress every this many epochs.
    """
    num_epochs: int = 10_000
    lr: float = 1e-3
    lr_schedule: str = "cosine"
    log_every: int = 500


@dataclass
class LBFGSConfig:
    """Configuration for the L-BFGS optimisation phase.

    Attributes:
        num_epochs: Maximum number of L-BFGS iterations.
        tol:        Convergence tolerance on the gradient error.
        log_every:  Print progress every this many steps.
    """
    num_epochs: int = 2_000
    tol: float = 1e-9
    log_every: int = 500
