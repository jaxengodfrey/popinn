import jax.numpy as jnp
import jax.random as jr

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
#     x_interior_span = (1e-4, 1 - 1e-4)
#     t_interior_span = (0.01, t_max*0.99)
#     x_mesh = jnp.linspace(*x_interior_span, int(jnp.sqrt(n_interior)))
#     t_mesh = jnp.linspace(*t_interior_span, int(jnp.sqrt(n_interior)))
#     x_grid, t_grid = jnp.meshgrid(x_mesh, t_mesh)
#     x_centers = x_grid.flatten()[:n_interior]
#     t_centers = t_grid.flatten()[:n_interior]

#     # Add noise (Brevi et al. strategy)
#     dx = x_mesh[1] - x_mesh[0] if len(x_mesh) > 1 else 0.1
#     dt = t_mesh[1] - t_mesh[0] if len(t_mesh) > 1 else 0.01
#     x_pts = x_centers + jr.normal(k1, shape=x_centers.shape) * dx * 0.2
#     t_pts = t_centers + jr.normal(k2, shape=t_centers.shape) * dt * 0.2
#     # x_pts = jr.normal(k1, x_centers, dx/2., shape=x_centers.shape)
#     # t_pts = t_centers + jr.normal(k2, t_centers, dt/2., shape=t_centers.shape)

#     # Clip to domain
#     x_pts = jnp.clip(x_pts, *x_interior_span)
#     t_pts = jnp.clip(t_pts, *t_interior_span)
#     colloc_xt = jnp.stack([x_pts, t_pts], axis=-1)

#     # IC points: t=0, random x
#     x_ic = jr.uniform(k3, shape=(n_ic,), minval=x_interior_span[0], maxval=x_interior_span[1])

#     # BC points: random t for x=0 and x=1
#     t_bc = jr.uniform(k4, shape=(n_bc,), minval=t_interior_span[0], maxval=t_interior_span[1])

#     return colloc_xt, x_ic, t_bc

def sample_collocation(key, n_interior: int, n_bc: int, n_ic: int,
                       t_max: float):
    """Sample collocation points for one training batch.
    
    Following Brevi et al.: sample from normal distributions centered
    on a regular mesh to span the domain while adding randomness.
    
    Returns:
        colloc_xt: (n_interior, 2) interior points [x, t]
        x_ic: (n_ic,) x-values for IC evaluation
        t_bc: (n_bc,) t-values for BC evaluation
    """
    k1, k2, k3, k4, k5 = jr.split(key, 5)

    # Interior collocation points in (0,1) x (0, t_max)
    # Mesh centers
    x_mesh = jnp.linspace(0.05, 0.95, int(jnp.sqrt(n_interior)))
    t_mesh = jnp.linspace(0.01, t_max * 0.99, int(jnp.sqrt(n_interior)))
    x_grid, t_grid = jnp.meshgrid(x_mesh, t_mesh)
    x_centers = x_grid.flatten()[:n_interior]
    t_centers = t_grid.flatten()[:n_interior]

    # Add noise (Brevi et al. strategy)
    dx = x_mesh[1] - x_mesh[0] if len(x_mesh) > 1 else 0.1
    dt = t_mesh[1] - t_mesh[0] if len(t_mesh) > 1 else 0.01
    x_pts = x_centers + jr.normal(k1, shape=x_centers.shape) * dx * 0.2
    t_pts = t_centers + jr.normal(k2, shape=t_centers.shape) * dt * 0.2

    # Clip to domain
    x_pts = jnp.clip(x_pts, 1e-4, 1.0 - 1e-4)
    t_pts = jnp.clip(t_pts, 1e-6, t_max)
    colloc_xt = jnp.stack([x_pts, t_pts], axis=-1)

    # IC points: t=0, random x
    x_ic = jr.uniform(k3, shape=(n_ic,), minval=1e-4, maxval=1.0 - 1e-4)

    # BC points: random t for x=0 and x=1
    t_bc = jr.uniform(k4, shape=(n_bc,), minval=1e-6, maxval=t_max)

    return colloc_xt, x_ic, t_bc