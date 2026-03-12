from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from .phi import g_equilibrium
import jax
if TYPE_CHECKING:
    from ..network.model import PINN

def hard(model: PINN, x: jnp.ndarray, t: jnp.ndarray, gamma: float, theta: float = 1., nu: float = 1.) -> jnp.ndarray:
    """Compute g(x,t) with hard-coded IC and right BC.
    
    g(x,t) = g_IC(x) + t * (1 - x) * NN(x, t)
    
    This ensures:
        g(x, 0) = g_IC(x)          (IC satisfied exactly)
        g(1, t) = g_IC(1) + 0 = 0  (right BC satisfied exactly)
    
    The left BC g(0, t) = theta must be enforced via a soft loss.

    Args:
        model (PINN): equinox NN
        x (jnp.ndarray): frequency
        t (jnp.ndarray): time
        gamma (float): scaled selection coefficient, equal to 2*Nref*s, where s is the selective advantage and Nref is the reference population size, typically the ancestral size.
        theta (float, optional): Population scaled mutation rate, equal to 4*Nref*u, where u is the mutation event rate per generation and Nref is the reference population size, typically the ancestral size. Defaults to 1.0.
        nu (float, optional): Population size relative to the reference population size Nref, i.e. nu = N/Nref. Defaults to 1.0.

    Returns:
        jnp.ndarray: transformed network output
    """
    xt = jnp.stack([x, t])
    nn_out = model(xt)
    g_ic = g_equilibrium(x, gamma, theta = theta, nu = nu)
    return g_ic + t * (1.0 - x) * nn_out


def soft(model: PINN, x: jnp.ndarray, t: jnp.ndarray, gamma: float, theta: float = 1., nu: float = 1.) -> jnp.ndarray:
    """Compute g(x,t) with no hard constraints (all enforced via loss)."""
    xt = jnp.stack([x, t])
    return model(xt)

