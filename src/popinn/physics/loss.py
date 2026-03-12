
import jax
import jax.numpy as jnp
from typing import NamedTuple
from .phi import g_equilibrium
from .pde import pde_residual
from .contraint import hard, soft


class LossWeights(NamedTuple):
    """Weights for each loss component."""
    pde: float = 1.0
    ic: float = 10.0
    bc_left: float = 10.0
    bc_right: float = 10.0
    non_negative: float = 0.1


def loss_pde(model, colloc_xt, gamma, g_fn, theta = 1., nu = 1.):
    """Mean squared PDE residual over collocation points."""
    def single_residual(xt):
        return pde_residual(model, xt[0], xt[1], gamma, g_fn, theta = theta, nu = nu)

    residuals = jax.vmap(single_residual)(colloc_xt)
    return jnp.mean(residuals ** 2)


def loss_ic(model, x_ic, gamma, g_fn, theta = 1., nu = 1.):
    """Mean squared error of IC: g(x, 0) = g_eq(x).
    Only needed for soft constraint mode.
    """
    def single_ic(x):
        g_pred = g_fn(model, x, jnp.array(0.0), gamma, theta = theta, nu = nu)        
        g_true = g_equilibrium(x, gamma, theta = theta, nu = nu)
        return (g_pred - g_true) ** 2

    return jnp.mean(jax.vmap(single_ic)(x_ic))


def loss_bc(model, t_bc, gamma, g_fn, theta = 1., nu = 1.):
    """Mean squared error of BCs.
    
    Left BC: g(0,t) = theta  (always needed, even in hard mode)
    Right BC: g(1,t) = 0     (only needed in soft mode; hard mode satisfies exactly)
    """
    def single_bc_left(t):
        g_left = g_fn(model, jnp.array(0.0), t, gamma, theta = theta, nu = nu)
        return (g_left - theta * nu) ** 2

    def single_bc_right(t):
        g_right = g_fn(model, jnp.array(1.0), t, gamma, theta = theta, nu = nu)
        return g_right ** 2

    return jnp.mean(jax.vmap(single_bc_left)(t_bc)), jnp.mean(jax.vmap(single_bc_right)(t_bc))


def loss_non_negative(model, colloc_xt, gamma, g_fn, theta = 1., nu = 1.):
    """Soft penalty for negative g values (inductive bias: g >= 0)."""
    def single_nn(xt):
        g_val = g_fn(model, xt[0], xt[1], gamma, theta = theta, nu = nu)
        return jnp.minimum(g_val, 0.0) ** 2

    return jnp.mean(jax.vmap(single_nn)(colloc_xt))


def total_loss(model, colloc_xt, x_ic, t_bc,
               gamma, weights: LossWeights, use_hard: bool, theta = 1., nu = 1.):
    """Compute total weighted loss."""
    g_fn = hard if use_hard else soft

    l_pde = loss_pde(model, colloc_xt, gamma, g_fn, theta = theta, nu = nu)
    l_bc_left, l_bc_right = loss_bc(model, t_bc, gamma, g_fn, theta = theta, nu = nu)

    if use_hard:
        # IC and right BC are exact; left BC still needs soft enforcement
        l_ic = 0.0
        l_bc_right = 0.0
    else:
        # l_ic and l_bc_right should be exactly zero w/ hard condition
        l_ic = loss_ic(model, x_ic, jnp.array(1e-5), g_fn, theta = theta, nu = theta) 

    l_nn = loss_non_negative(model, colloc_xt, gamma, g_fn, theta = theta, nu = nu)

    total = (weights.pde * l_pde
             + weights.ic * l_ic
             + weights.bc_left * l_bc_left
             + weights.bc_right * l_bc_right
             + weights.non_negative * l_nn)

    return total, {"pde": l_pde, "ic": l_ic,
                   "bc_left": l_bc_left, "bc_right": l_bc_right,
                   "non_neg": l_nn}