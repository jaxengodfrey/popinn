from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import jax.random as jr

from ..config import Batch, PhysicsConfig, SamplingConfig

def xgrid(pts: int, crwd: float = 8.0) -> jnp.ndarray:
    """Non-uniform grid on [0, 1] with points crowded near the boundaries.
    
    Uses a logistic transform of a uniform grid on [-1, 1].
    Higher crwd => more points near x=0 and x=1.
    """
    unif = jnp.linspace(-1, 1, pts)
    grid = 1.0 / (1.0 + jnp.exp(-crwd * unif))
    grid = (grid - grid[0]) / (grid[-1] - grid[0])
    return grid


def _sample_collocation(key, n_interior: int, t_max: float, uniform = False, ic_bc_shape = None, x_crowd = 8.0):
    """Sample collocation points for one training batch.
    
    Following Brevi et al.: sample from normal distributions centered
    on a regular mesh to span the domain while adding randomness.
    
    Returns:
        colloc_xt: (n_interior, 2) interior points [x, t]
        x_ic: (n_ic,) x-values for IC evaluation
        t_bc: (n_bc,) t-values for BC evaluation
    """
    k1, k2, k3, k4, k5 = jr.split(key, 5)

    # x_interior_span = (1e-5, 1 - 1e-5)
    t_interior_span = (1e-5,  t_max - 1e-5)

    # x_grid = jnp.linspace(*x_interior_span, n_interior)
    x_grid = xgrid(n_interior + 2, crwd = x_crowd)[1:-1]
    x_interior_span = (x_grid[0], x_grid[-1])
    t_grid = jnp.linspace(*t_interior_span, n_interior)
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]

    x_mesh, t_mesh = jnp.meshgrid(x_grid, t_grid)
    xt_grid = jnp.stack([x_mesh.flatten(), t_mesh.flatten()])

    if uniform:
        colloc_xt = jnp.expand_dims(xt_grid, axis = -1)
        x_ic = jr.uniform(k3, shape=ic_bc_shape, minval=x_interior_span[0], maxval=x_interior_span[1])
        t_bc = jr.uniform(k4, shape=ic_bc_shape, minval=t_interior_span[0], maxval=t_interior_span[1])
        return colloc_xt, x_ic, t_bc
    
    else:

        new_x = xt_grid[0] + jr.normal(k1, shape = n_interior*n_interior) * dx / 5.
        new_t = xt_grid[1] + jr.normal(k2, shape=n_interior*n_interior) * dt / 5.

        x_pts = jnp.clip(new_x, *x_interior_span)
        t_pts = jnp.clip(new_t, *t_interior_span)

        colloc_xt = jnp.expand_dims(jnp.stack([x_pts, t_pts]), axis = -1) #add axis for gamma parameter so that vmapped residuals work

        # IC points: t=0, random x
        if ic_bc_shape is None:
            ic_bc_shape = (n_interior,1)

        x_ic = jr.uniform(k3, shape=ic_bc_shape, minval=x_interior_span[0], maxval=x_interior_span[1])

        # BC points: random t for x=0 and x=1
        t_bc = jr.uniform(k4, shape=ic_bc_shape, minval=t_interior_span[0], maxval=t_interior_span[1])

        return colloc_xt, x_ic, t_bc
    

def _sample_collocation_and_param(key, param_range: tuple, n_param_vals: int, n_xt_grid: int, t_max: float, sample_num = 1000, uniform = False, x_crowd = 8.):

    colloc_xt, x_ic, t_bc = _sample_collocation(key, n_xt_grid, t_max, uniform = uniform, ic_bc_shape = (n_xt_grid, n_param_vals), x_crowd = x_crowd)
    k1, _ = jr.split(key, 2)

    param = jnp.linspace(*param_range, n_param_vals)

    idxs = jr.choice(k1, jnp.arange(colloc_xt.shape[1]), shape = (sample_num,n_param_vals))
    colloc_xt_samples = jnp.squeeze(colloc_xt[:,idxs])

    return colloc_xt_samples, x_ic, t_bc, param


# ---------------------------------------------------------------------------
# sampler factories
# ---------------------------------------------------------------------------

def make_pinn_sampler(
    sampling_cfg: SamplingConfig,
    physics_cfg: PhysicsConfig,
) -> Callable:
    """Build a sampler for a standard single-gamma PINN.

    The returned ``sample_fn`` wraps gamma as a length-1 array so that the
    batch layout (with a trailing gamma axis) is identical to the P2INN case.

    Args:
        sampling_cfg: Controls grid size and uniformity.
        physics_cfg:  Provides ``gamma_evol`` and ``t_max``.

    Returns:
        ``sample_fn(key) -> Batch``
    """
    gamma_evol = physics_cfg.gamma_evol
    t_max = physics_cfg.t_max
    n = sampling_cfg.n_interior
    uniform = sampling_cfg.uniform

    def sample_fn(key):
        colloc_xt, x_ic, t_bc = _sample_collocation(
            key, n, t_max, uniform=uniform
        )
        return Batch(colloc_xt=colloc_xt, x_ic=x_ic, t_bc=t_bc,
                     extras={"gamma": gamma_evol})

    return sample_fn


def make_p2inn_sampler(
    sampling_cfg: SamplingConfig,
    physics_cfg: PhysicsConfig,
) -> Callable:
    """Build a sampler for a parametrized P2INN that trains over a gamma range.

    Each call to ``sample_fn`` draws a fresh set of collocation points and
    samples gamma values uniformly from ``gamma_range``.

    Args:
        sampling_cfg: Controls grid size and uniformity.
        physics_cfg:  Provides ``t_max``.
        gamma_range:  ``(gamma_min, gamma_max)`` interval.
        n_gamma:      Number of gamma values per batch.
        sample_num:   Interior collocation points per gamma value.

    Returns:
        ``sample_fn(key) -> Batch``
    """
    t_max = physics_cfg.t_max
    n = sampling_cfg.n_interior
    uniform = sampling_cfg.uniform
    n_batch = sampling_cfg.n_batch
    x_crowd = sampling_cfg.x_crowd

    def sample_fn(key):
        colloc_xt, x_ic, t_bc, gamma = _sample_collocation_and_param(
            key, physics_cfg.gamma_range, physics_cfg.n_gamma, n, t_max,
            sample_num=n_batch, uniform=uniform, x_crowd = x_crowd
        )
        return Batch(colloc_xt=colloc_xt, x_ic=x_ic, t_bc=t_bc,
                     extras={"gamma": gamma})

    return sample_fn