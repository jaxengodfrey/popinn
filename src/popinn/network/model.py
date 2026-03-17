import equinox as eqx
import jax.random as jr
import jax.numpy as jnp
import jax

#TODO make PINN an abstract class and then define different PINNs with different architecture

class PINN(eqx.Module):
    """Feedforward neural network for g(x, t).
    
    Input: (x, t) -> 2 features
    Output: scalar (raw network output before any hard constraint transform)
    """
    layers: list

    def __init__(self, key, hidden_dims: list[int] = None):
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64]

        keys = jr.split(key, len(hidden_dims) + 1)
        dims = [2] + hidden_dims + [1]
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]))

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass. xt has shape (2,): [x, t]."""
        h = jnp.stack([x, t])
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        # Final layer: no activation
        return self.layers[-1](h).squeeze()
    
    
def evaluate_model(model, x, t_eval):
    """Evaluate the trained PINN at a specific time.
    
    Returns x, g_pred, f_pred arrays.
    """
    # x = jnp.linspace(1e-4, 1.0 - 1e-4, num_x)
    t = jnp.full_like(x, t_eval)
    g_pred = jax.vmap(model)(x, t)
    f_pred = g_pred / (x * (1.0 - x))

    return x, g_pred, f_pred