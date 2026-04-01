from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
if TYPE_CHECKING:
    from ..network.models import PINN
import inspect
from functools import partial
from ..network.models import eval_partial_model

def pde_residual(model: PINN, x: jnp.ndarray, t: jnp.ndarray, gamma_evol: float, nu: float = 1.) -> jnp.ndarray:
    """Compute PDE residual: dg/dt - gamma*x*(1-x)*dg/dx - x*(1-x)/(2*nu) * d²g/dx².
    """
    g_xt = eval_partial_model(model, gamma_evol)
        
    # Time derivative: dg/dt
    dg_dt = jax.grad(g_xt, argnums=1)(x, t)

    # Spatial derivatives: dg/dx, d²g/dx²
    dg_dx = jax.grad(g_xt, argnums=0)(x, t)
    d2g_dx2 = jax.grad(jax.grad(g_xt, argnums=0), argnums=0)(x, t)

    # Diffusion and advection coefficients
    diff = x * (1.0 - x) / (2.0 * nu)
    adv = gamma_evol * x * (1.0 - x)

    residual = dg_dt + adv * dg_dx - diff * d2g_dx2
    return residual


def log_pde_residual(model: PINN, x: jnp.ndarray, t: jnp.ndarray, gamma_evol: float, nu: float = 1.) -> jnp.ndarray:
    """Compute PDE residual of log model, g = exp(h) : dg/dt - gamma*x*(1-x)*dg/dx - x*(1-x)/(2*nu) * d²g/dx².
        the common factor exp(h) is divided out for numerical stability
    """
    h_xt = eval_partial_model(model, gamma_evol)
        
    # Time derivative: dg/dt
    dg_dt = jax.grad(h_xt, argnums=1)(x, t)

    # Spatial derivatives: dg/dx, d²g/dx²
    dh_dx = jax.grad(h_xt, argnums = 0)(x,t)
    d2h_dx2 = jax.grad(jax.grad(h_xt, argnums = 0), argnums = 0)(x,t)

    dg_dx = dh_dx
    d2g_dx2 = (d2h_dx2 + dh_dx**2.)

    # Diffusion and advection coefficients
    diff = x * (1.0 - x) / (2.0 * nu)
    adv = gamma_evol * x * (1.0 - x)

    residual = dg_dt + adv * dg_dx - diff * d2g_dx2
    return residual
