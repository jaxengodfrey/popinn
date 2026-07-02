
[![CI-Tests](https://github.com/jaxengodfrey/popinn/actions/workflows/ci.yaml/badge.svg)](https://github.com/jaxengodfrey/popinn/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://jaxengodfrey.github.io/popinn/)


# `Poppin`

Popinn is a small, composable library for physics-informed deep learning with [JAX](https://github.com/jax-ml/jax/)/[Equinox](https://github.com/patrick-kidger/equinox).

Define your problem (coordinates, PDE, boundary/initial conditions, parameters, etc.) using a few standard signatures and easily compose with any of the built-in networks:

- PINNs ([Raissi et al. 2019](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125))
- Parametrized PINNs (P$^2$INN)([Cho et al. 2024](https://arxiv.org/abs/2408.09446))
- Deep Operator Networks ([Lu et al. 2020](https://arxiv.org/abs/1910.03193)).

Built-ins not enough? Easily build your own model by subclassing `popinn.AbstractModel`.

Built on the JAX/Equinox ecosystem, the model, residuals, and loss are end-to-end differentiable and JIT-compiled. The network is evaluated across entire coordinate/parameter grids at once or in batches, allowing the flexibility to trade compute for memory.

## Installation
Pip installation will be available soon. In the meantime, Popinn can be installed via:
```bash
git clone https://github.com/jaxengodfrey/popinn.git
cd popinn
python -m pip install .
```
##### Dependencies
Popinn requires `jax`, `equinox`, `optax`, `jaxopt`, `jaxtyping`

It may be necessary to work in 64-bit precision ([Xu, et. al. 2025](https://arxiv.org/abs/2505.10949)). Enable with `jax.config` at the beginning of your script:
  ```python
  import jax
  jax.config.update("jax_enable_x64", True)
  ```

## Quick Example

Below is a brief code example for a parametrized PINN. For more detailed examples, see the [Documentation](https://jaxengodfrey.github.io/popinn/).
```python
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array
from popinn import P2INN, ResidualTerm, Loss, train_model, AdamConfig, LBFGSConfig

# construct a data container with eqx.Module
class Data(eqx.Module):
    pde_coords: tuple[Array, ...]
    aux: tuple

# define 1D coordinate arrays and any auxiliary inputs
# here our coordinates are x & t and a single parameter a
data = Data(
    pde_coords=(jnp.linspace(0, 1, 100), jnp.linspace(0, 1, 100)),  # (x, t)
    aux=(jnp.linspace(0, 1, 10),),                                  # (a,)
)

# define per-point residuals following the signature
# res(model) -> r(x, t, aux) -> scalar
def pde_residual(model):
    def r(x, t, aux):
        # du/dt - d^2u/dx^2 - a
        a = aux[0]
        return model.D(1)(x, t, aux) - model.D(0, 0)(x, t, aux) - a
    return r

# wrap each residual with ResidualTerm and pass a list of all terms into Loss
loss = Loss([ResidualTerm("pde", pde_residual)])

# initialize your desired model
model = P2INN(jr.PRNGKey(0), num_params=1, num_coords=2)

# train: first Adam, then switch to L-BFGS after 1k Adam steps
model, history = train_model(
    model, data, loss,
    [AdamConfig(lr=1e-3, num_epochs=1000), LBFGSConfig(num_epochs=1000)],
)
