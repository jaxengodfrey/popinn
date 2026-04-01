import equinox as eqx
import jax.random as jr
import jax.numpy as jnp
import jax
import inspect
from functools import partial

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
    
#####################
# P^2INN
#####################

import abc
from jaxtyping import Float, Array


class ParameterEncoder(eqx.Module):
    """Encoder for PDE parameter gamma.
    Input: scalar gamma
    Output: h_param of dimension hidden_dim (default 150)
    """
    layers: list

    def __init__(self, key, hidden_dim: int = 150, num_layers: int = 4):
        keys = jr.split(key, num_layers)
        # First layer: scalar input (dim 1) -> hidden_dim
        # Intermediate layers: hidden_dim -> hidden_dim
        dims = [1] + [hidden_dim] * num_layers
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]))

    def __call__(self, gamma: jnp.ndarray) -> jnp.ndarray:
        h = jnp.atleast_1d(gamma)
        for layer in self.layers:
            h = jnp.tanh(layer(h))
        return h


class CoordinateEncoder(eqx.Module):
    """Encoder for spatiotemporal coordinates (x, t).
    Input: (x, t) -> 2 features
    Output: h_coord of dimension hidden_dim (default 64)
    """
    layers: list

    def __init__(self, key, hidden_dim: int = 64, num_layers: int = 3):
        keys = jr.split(key, num_layers)
        dims = [2] + [hidden_dim] * num_layers
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]))

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        h = jnp.stack([x, t])
        for layer in self.layers:
            h = jnp.tanh(layer(h))
        return h

class ManifoldNetwork(eqx.Module):
    """Manifold network that combines h_param and h_coord to predict solution.
    Input: concatenation of h_coord and h_param
    Output: scalar prediction
    """
    layers: list

    def __init__(self, key, input_dim: int, hidden_dim: int = 64, num_layers: int = 5):
        keys = jr.split(key, num_layers)
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [1]
        self.layers = []
        for i in range(num_layers):
            self.layers.append(eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]))

    def __call__(self, h_concat: jnp.ndarray) -> jnp.ndarray:
        h = h_concat
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        return self.layers[-1](h).squeeze()


class P2INN(eqx.Module):
    """Parameterized Physics-Informed Neural Network.
    Input: (x, t, gamma)
    Output: scalar solution estimate
    """
    param_encoder: ParameterEncoder
    coord_encoder: CoordinateEncoder
    manifold: ManifoldNetwork

    def __init__(
        self,
        key,
        param_hidden_dim: int = 150,
        param_num_layers: int = 4,
        coord_hidden_dim: int = 64,
        coord_num_layers: int = 3,
        manifold_hidden_dim: int = 64,
        manifold_num_layers: int = 5,
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.param_encoder = ParameterEncoder(k1, param_hidden_dim, param_num_layers)
        self.coord_encoder = CoordinateEncoder(k2, coord_hidden_dim, coord_num_layers)
        manifold_input_dim = param_hidden_dim + coord_hidden_dim
        self.manifold = ManifoldNetwork(
            k3, manifold_input_dim, manifold_hidden_dim, manifold_num_layers
        )

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
        h_param = self.param_encoder(gamma)
        h_coord = self.coord_encoder(x, t)
        h_concat = jnp.concatenate([h_coord, h_param])
        return self.manifold(h_concat)


def eval_partial_model(model, gamma):
    if inspect.isfunction(model):
        params = list(inspect.signature(model).parameters)
    else:
        params = list(inspect.signature(model.__call__).parameters)

    if len(params) == 2:
        g_xt = model
    else: 
        g_xt = partial(model, gamma = gamma)
    return g_xt
    

def evaluate_model(model, x, t_eval):
    """Evaluate the trained PINN at a specific time.
    
    Returns x, g_pred, f_pred arrays.
    """
    # x = jnp.linspace(1e-4, 1.0 - 1e-4, num_x)
    t = jnp.full_like(x, t_eval)
    g_pred = jax.vmap(model)(x, t)
    f_pred = g_pred / (x * (1.0 - x))

    return x, g_pred, f_pred


