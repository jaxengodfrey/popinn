from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
if TYPE_CHECKING:
    from ..network.model import PINN


def pde_residual(model: PINN, x: jnp.ndarray, t: jnp.ndarray, gamma: float, constraint_fn, theta: float = 1., nu: float = 1., gamma_init = 1e-5) -> jnp.ndarray:
    """Compute PDE residual: dg/dt - gamma*x*(1-x)*dg/dx - x*(1-x)/(2*nu) * d²g/dx².
    
    """
    def g_xt(x_, t_):
        return constraint_fn(model, x_, t_, gamma_init, theta = theta, nu = nu)

    # Time derivative: dg/dt
    dg_dt = jax.grad(g_xt, argnums=1)(x, t)

    # Spatial derivatives: dg/dx, d²g/dx²
    dg_dx = jax.grad(g_xt, argnums=0)(x, t)
    d2g_dx2 = jax.grad(jax.grad(g_xt, argnums=0), argnums=0)(x, t)

    # Diffusion and advection coefficients
    diff = x * (1.0 - x) / (2.0 * nu)
    adv = gamma * x * (1.0 - x)

    residual = dg_dt + adv * dg_dx - diff * d2g_dx2
    return residual