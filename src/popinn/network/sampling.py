import jax.numpy as jnp
import jax.random as jr

def xgrid(pts: int, crwd: float = 8.0) -> jnp.ndarray:
    """Non-uniform grid on [0, 1] with points crowded near the boundaries.
    
    Uses a logistic transform of a uniform grid on [-1, 1].
    Higher crwd => more points near x=0 and x=1.
    """
    unif = jnp.linspace(-1, 1, pts)
    grid = 1.0 / (1.0 + jnp.exp(-crwd * unif))
    grid = (grid - grid[0]) / (grid[-1] - grid[0])
    return grid


def sample_collocation(key, n_interior: int, t_max: float, uniform = False, ic_bc_shape = None):
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
    x_grid = xgrid(n_interior + 2)[1:-1]
    x_interior_span = (x_grid[0], x_grid[-1])
    t_grid = jnp.linspace(*t_interior_span, n_interior)
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]

    x_mesh, t_mesh = jnp.meshgrid(x_grid, t_grid)
    xt_grid = jnp.stack([x_mesh.flatten(), t_mesh.flatten()])

    if uniform:
        colloc_xt = jnp.expand_dims(xt_grid, axis = -1)
        return xt_grid, x_grid, t_grid
    
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
    

def sample_collocation_and_param(key, param_range: tuple, n_param_vals: int, n_xt_grid: int, t_max: float, sample_num = 1000, uniform = False):

    colloc_xt, x_ic, t_bc = sample_collocation(key, n_xt_grid, t_max, uniform = uniform, ic_bc_shape = (n_xt_grid, n_param_vals))
    k1, _ = jr.split(key, 2)

    param = jnp.linspace(*param_range, n_param_vals)

    idxs = jr.choice(k1, jnp.arange(colloc_xt.shape[1]), shape = (sample_num,n_param_vals))
    colloc_xt_samples = jnp.squeeze(colloc_xt[:,idxs])

    return colloc_xt_samples, x_ic, t_bc, param




# def sample_collocation(key, n_interior: int, n_bc: int, n_ic: int,
#                        t_max: float):
#     """Sample collocation points for one training batch.
    
#     Following Brevi et al.: sample from normal distributions centered
#     on a regular mesh to span the domain while adding randomness.
    
#     Returns:
#         colloc_xt: (n_interior, 2) interior points [x, t]
#         x_ic: (n_ic,) x-values for IC evaluation
#         t_bc: (n_bc,) t-values for BC evaluation
#     """
#     k1, k2, k3, k4, k5 = jr.split(key, 5)

#     # Interior collocation points in (0,1) x (0, t_max)
#     # Mesh centers
#     x_mesh = jnp.linspace(0.05, 0.95, int(jnp.sqrt(n_interior)))
#     t_mesh = jnp.linspace(0.01, t_max * 0.99, int(jnp.sqrt(n_interior)))
#     x_grid, t_grid = jnp.meshgrid(x_mesh, t_mesh)
#     x_centers = x_grid.flatten()[:n_interior]
#     t_centers = t_grid.flatten()[:n_interior]

#     # Add noise (Brevi et al. strategy)
#     dx = x_mesh[1] - x_mesh[0] if len(x_mesh) > 1 else 0.1
#     dt = t_mesh[1] - t_mesh[0] if len(t_mesh) > 1 else 0.01
#     x_pts = x_centers + jr.normal(k1, shape=x_centers.shape) * dx * 0.2
#     t_pts = t_centers + jr.normal(k2, shape=t_centers.shape) * dt * 0.2

#     # Clip to domain
#     x_pts = jnp.clip(x_pts, 1e-4, 1.0 - 1e-4)
#     t_pts = jnp.clip(t_pts, 1e-6, t_max)
#     colloc_xt = jnp.stack([x_pts, t_pts], axis=-1)

#     # IC points: t=0, random x
#     x_ic = jr.uniform(k3, shape=(n_ic,), minval=1e-4, maxval=1.0 - 1e-4)

#     # BC points: random t for x=0 and x=1
#     t_bc = jr.uniform(k4, shape=(n_bc,), minval=1e-6, maxval=t_max)

#     return colloc_xt, x_ic, t_bc