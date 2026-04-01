from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from .phi import g_equilibrium
from .pde import pde_residual
from ..network.models import eval_partial_model
from ..config import Batch, LossWeights, PhysicsConfig

# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def mse(residuals: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error."""
    return jnp.mean(residuals ** 2.)


def mae(residuals: jnp.ndarray) -> jnp.ndarray:
    """Mean absolute error."""
    return jnp.mean(jnp.abs(residuals))

# ---------------------------------------------------------------------------
# Per-component residual maps (internal)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PDE residual and loss

def _mapped_pde_residual(model, colloc_xt, gamma_evol, nu = 1.):
    def single_residual(xt, gamma):
        return pde_residual(model, xt[0], xt[1], gamma, nu = nu)
    map_sample = jax.vmap(single_residual, in_axes = (1, None))
    map_gamma = jax.vmap(map_sample, in_axes = (2, 0))
    return map_gamma(colloc_xt, gamma_evol)


def pde_loss(model, metric, *args, **kwargs):
    return metric(_mapped_pde_residual(model, *args, **kwargs))


# ---------------------------------------------------------------------------
# Initial condition residual and loss

def _mapped_ic_residual(model, x_ic, gamma_evol, gamma_init: float, theta = 1., nu = 1.):
    """Mean squared error of IC: g(x, 0) = g_eq(x).
    """
    def single_ic(x, gamma):
        g_xt = eval_partial_model(model, gamma)
        g_pred = g_xt(x, jnp.array(0.0))      
        g_true = g_equilibrium(x, gamma_init, theta = theta, nu = nu)
        return (g_pred - g_true)
    
    map_sample = jax.vmap(single_ic, in_axes = (0, None))
    map_gamma = jax.vmap(map_sample, in_axes = (1, 0))

    return map_gamma(x_ic, gamma_evol)

def ic_loss(model, metric, *args, **kwargs):
    return metric(_mapped_ic_residual(model, *args, **kwargs))


# ---------------------------------------------------------------------------
# Boundary condition residual and loss

def _mapped_bc_residual(model, t_bc, gamma_evol, theta = 1., nu = 1.):
    """Mean squared error of BCs.
    
    Left BC: g(0,t) = theta  
    Right BC: g(1,t) = 0     
    """

    def map_bc(bc):
        map_sample = jax.vmap(bc, in_axes = (0, None))
        map_gamma = jax.vmap(map_sample, in_axes = (1, 0))
        return map_gamma

    def single_bc_left(t, gamma):
        g_xt = eval_partial_model(model, gamma)
        g_left = g_xt(jnp.array(0.), t)
        return g_left - theta * nu

    def single_bc_right(t, gamma):
        g_xt = eval_partial_model(model, gamma)
        g_right = g_xt(jnp.array(1.0), t)
        return g_right

    return map_bc(single_bc_left)(t_bc, gamma_evol), map_bc(single_bc_right)(t_bc, gamma_evol)

def bc_loss(model, metric, *args, **kwargs):
    left_bc_res, right_bc_res = _mapped_bc_residual(model, *args, **kwargs)
    return metric(left_bc_res), metric(right_bc_res)


# ---------------------------------------------------------------------------
# Non-negative residual and loss

def _mapped_non_negative_residual(model, colloc_xt, gamma_evol):
    """Soft penalty for negative g values (inductive bias: g >= 0)."""
    def single_nn(xt, gamma):
        g_xt = eval_partial_model(model, gamma)
        g_val = g_xt(xt[0], xt[1])
        return jnp.minimum(g_val, 0.0)
    
    map_sample = jax.vmap(single_nn, in_axes = (1, None))
    map_gamma = jax.vmap(map_sample, in_axes = (2, 0))

    return map_gamma(colloc_xt, gamma_evol)

def non_negative_loss(model, metric, *args, **kwargs):
    return metric(_mapped_non_negative_residual(model, *args, **kwargs))


# ---------------------------------------------------------------------------
# True solution at t_f residual and loss
# NOT CURRENTLY IMPLEMENTED

def _mapped_solution_residual(model, x_tf, gamma_evol, true_sol, t_max):
    """Soft penalty for negative g values (inductive bias: g >= 0)."""
    def single(x, gamma, sol):
        g_xt = eval_partial_model(model, gamma)
        g_val = g_xt(x, t_max)
        return (jnp.log(sol + 1e-8) - jnp.log(g_val + 1e-8))
    
    map_sample = jax.vmap(single, in_axes = (0, None, 0))
    map_gamma = jax.vmap(map_sample, in_axes = (None, 0, 0))

    return map_gamma(x_tf, gamma_evol, true_sol)

def solution_loss(model, metric, *args, **kwargs):
    return metric(_mapped_solution_residual(model, *args, **kwargs))



# ---------------------------------------------------------------------------
# Make full loss funtion
# ---------------------------------------------------------------------------

def make_loss(
    physics_cfg: PhysicsConfig,
    weights: LossWeights = None,
    pde_metric: Callable = mse,
    ic_metric: Callable = mse,
    bc_metric: Callable = mse,
    nn_metric: Callable = mse,
) -> Callable:
    """Build a loss function for the population genetics PINN.

    The returned ``loss_fn`` has the signature::

        loss_fn(model, batch) -> (total_loss, loss_dict)

    where ``batch.extras["gamma"]`` provides the array of gamma values used
    for vmapping (shape ``(n_gamma,)``).  For a standard single-gamma PINN
    this will be ``jnp.array([gamma_evol])``.

    Args:
        physics_cfg:     Physical parameters (theta, nu, gamma_init, …).
        weights:         Per-component loss weights.  Defaults to LossWeights().
        pde_residual_fn: PDE residual function with signature
                         ``(model_fn, x, t, gamma, nu) -> scalar``.
                         Defaults to the built-in ``log_pde_residual``.
        pde_metric:      Aggregation metric applied to PDE residuals.  Defaults to ``mae``.
        ic_metric:       Aggregation metric applied to IC residuals.   Defaults to ``mse``.
        bc_metric:       Aggregation metric applied to BC residuals.   Defaults to ``mse``.

    Returns:
        A callable ``loss_fn(model, batch) -> (float, dict)``.
    """
    if weights is None:
        weights = LossWeights()

    theta = physics_cfg.theta
    nu = physics_cfg.nu
    gamma_init = physics_cfg.gamma_init

    def loss_fn(model, batch: Batch):
        gamma_evol = batch.extras["gamma"]

        l_pde = pde_loss(model, pde_metric, batch.colloc_xt, gamma_evol, nu = nu)

        l_ic = ic_loss(model, ic_metric, batch.x_ic, gamma_evol, gamma_init, theta = theta, nu = nu)

        l_bc_left, l_bc_right = bc_loss(model, bc_metric, batch.t_bc, gamma_evol, theta = theta, nu = nu)

        l_nn = non_negative_loss(model, nn_metric, batch.colloc_xt, gamma_evol)

        total = (
            weights.pde * l_pde
            + weights.ic * l_ic
            + weights.bc_left * l_bc_left
            + weights.bc_right * l_bc_right
            + weights.non_negative * l_nn
        )

        loss_dict = {
            "pde": l_pde,
            "ic": l_ic,
            "bc_left": l_bc_left,
            "bc_right": l_bc_right,
            "non_neg": l_nn,
        }
        return total, loss_dict

    return loss_fn