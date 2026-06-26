# `Poppin` - Physics-Informed Deep Learning with JAX/Equinox

`Popinn` is a small, composable library for physics-informed deep learning in JAX/Equinox. Define your problem (coordinates, PDE, boundary/initial conditions, parameters, etc.) using a few standard signatures and easily compose with any of the built-in algorithms: PINNs, Parametrized PINNs, and Deep Operator Networks. Built-ins not enough? Easily build your own algorithm by subclassing `popinn.AbstractModel`, following the `equinox` pattern.

Built on the JAX/Equinox ecosystem, the model, residuals, and loss are end-to-end differentiable and JIT-compiled. The network is evaluated across entire coordinate/parameter grids at once or in batches, allowing the flexibility to trade compute for memory.


---

## The one idea everything rests on

Every model, derivative, and residual shares a single signature:

```python
fn(*coords, aux) -> scalar
```

- **Coordinates** are passed as individual scalar arguments (`x, t, ...`) — not a packed array — so per-coordinate derivatives compose under `jax.grad`.
- **Auxiliary inputs** (PDE parameters, sensor-sampled functions, initial conditions) are passed as a single trailing **tuple** (a pytree), so `jax` can address leaves and scalar/vector/ragged components need no special-casing.

Everything else — batching, differentiation, loss composition — is built to operate on functions of this shape.

---


## Installation & requirements

- Core dependencies: `jax`, `equinox`, `optax`, `jaxopt`, `jaxtyping`.
- The examples additionally requires `dadi` for the reference solution.
- **Enable float64 before creating any arrays** — PINNs generally need it [(Xu, et. al. 2025)](https://arxiv.org/abs/2505.10949):
  ```python
  import jax
  jax.config.update("jax_enable_x64", True)
  ```

---

## Quick start

The typical workflow:

- **1. Build a data container**: an `eqx.Module` holding coordinate grids per term and additional inputs in `aux` (can be empty, but must be present).
- **2. Write per-point residuals** as factories `residual_fn(model) -> r(*coords, aux) -> scalar`, using `model.D(...)` for derivatives. A residual's `aux` may be a *superset* of what the model consumes (e.g. unpack a PDE coefficient the network never sees).
- **3. Wrap each residual in a `popinn.ResidualTerm`**: Each term `<name>` reads `data.<name>_coords`.
- **4. Compose a `Loss`**: Pass a list containing each residual term to `popinn.Loss`
- **4. Train** with one or more optimizer configs, optionally passing a `sample_fn` to resample collocation points.


<details>
<summary>Example Script</summary>

```python
from popinn import (
    DeepONet, AdamConfig, LBFGSConfig, train_model,
    ResidualTerm, Loss,
)
import equinox as eqx
from jaxtyping import Array
import jax.numpy as jnp

# Data container
def Data(eqx.module):
    pde_coords: tuple[Array]
    aux: tuple

# Fill data container
data = Data(
    pde_coords = (jnp.linspace(0,1,100), jnp.linspace(0,1,100)) # (x, t)
    aux = (jnp.linspace(0,1,10),) # (a,)
)

# PDE residual factory that returns the standard signature fn(*coords, aux) -> scalar
def pde_residual(model):
    def r(x, t, aux):
        # dy_dt = a * dy_dx
        a = aux[0]
        dy_dt = model.D(1)(x, t, aux)
        dy_dx = model.D(0)(x, t, aux)

        return dy_dt - dy_dx * a
    return r

# Do the same for the other residuals, like BC/ICs

# Compose loss (more residuals can be included, e.g. ResidualTerm('ic', ic_residual))
loss = Loss([ResidualTerm("pde",  pde_residual),])

# Define the model
model = P2INN(key, num_params = 1, num_coords = 2)

# Train the model, first with 1000 Adam steps, then 1000 L-BFGS steps
model, history = train_model(
    model, data, loss,
    [AdamConfig(lr=1e-3, num_epochs=1000), LBFGSConfig(num_epochs=1000)],
)
```
</details>


For a different problem, swap the model and modify the residual functions and data container — the /loss/train machinery is unchanged.
