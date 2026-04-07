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

# class CoordinateEncoder(eqx.Module):
#     """Encoder for spatiotemporal coordinates (x, t).
#     Input: (x, t) -> 2 features
#     Output: h_coord of dimension hidden_dim (default 64)
#     """
#     layers: list

#     def __init__(self, key, hidden_dim: int = 64, num_layers: int = 3):
#         keys = jr.split(key, num_layers)
#         dims = [2] + [hidden_dim] * num_layers
#         self.layers = []
#         for i in range(len(dims) - 1):
#             self.layers.append(eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]))

#     def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
#         h = jnp.stack([x, t])
#         for layer in self.layers:
#             h = jnp.tanh(layer(h))
#         return h

class CoordinateEncoder(eqx.Module):
    """Encoder for spatiotemporal coordinates (x, t)."""
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
            h = jax.nn.silu(layer(h))
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
        return jax.nn.softplus(self.layers[-1](h).squeeze())
        # out = self.layers[-1](h).squeeze()
        # return jnp.exp(jnp.clip(out, -20.,20.))

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
    
class P2INN_2p(eqx.Module):
    """Parameterized Physics-Informed Neural Network.
    Input: (x, t, gamma)
    Output: scalar solution estimate
    """
    param_encoder_neg: ParameterEncoder
    param_encoder_pos: ParameterEncoder
    coord_encoder: CoordinateEncoder
    manifold: ManifoldNetwork

    def __init__(
        self,
        key,
        param_hidden_dim: int = 150,
        param_num_layers: int = 2,
        coord_hidden_dim: int = 64,
        coord_num_layers: int = 3,
        manifold_hidden_dim: int = 64,
        manifold_num_layers: int = 5,
    ):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.param_encoder_neg = ParameterEncoder(k4, param_hidden_dim, param_num_layers)
        self.param_encoder_pos = ParameterEncoder(k1, param_hidden_dim, param_num_layers)
        self.coord_encoder = CoordinateEncoder(k2, coord_hidden_dim, coord_num_layers)
        manifold_input_dim = param_hidden_dim + coord_hidden_dim
        self.manifold = ManifoldNetwork(
            k3, manifold_input_dim, manifold_hidden_dim, manifold_num_layers
        )

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:

        h_param = jnp.where(jnp.less(gamma, 0.), self.param_encoder_neg(gamma), self.param_encoder_pos(gamma))
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


######################################


import equinox as eqx
import jax.random as jr
import jax.numpy as jnp
import jax


class FiLMParameterEncoder(eqx.Module):
    """Encoder that produces per-layer FiLM parameters (scale and shift).
    
    Input: scalar gamma
    Output: list of (gamma_i, beta_i) pairs, one per manifold layer
    """
    trunk: list  # shared layers that build a representation of gamma
    film_heads: list  # per-layer linear projections to (scale, shift)

    def __init__(self, key, hidden_dim: int = 150, num_trunk_layers: int = 3,
                 num_film_layers: int = 4, film_target_dim: int = 64):
        keys = jr.split(key, num_trunk_layers + num_film_layers)

        # Trunk: scalar -> shared representation
        dims = [1] + [hidden_dim] * num_trunk_layers
        self.trunk = []
        for i in range(num_trunk_layers):
            self.trunk.append(eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]))

        # Heads: one per manifold layer, each producing (scale, shift) of size film_target_dim
        self.film_heads = []
        for i in range(num_film_layers):
            # Output 2 * film_target_dim: first half is scale, second half is shift
            self.film_heads.append(
                eqx.nn.Linear(hidden_dim, 2 * film_target_dim, key=keys[num_trunk_layers + i])
            )

    def __call__(self, gamma: jnp.ndarray):
        h = jnp.atleast_1d(gamma)
        for layer in self.trunk:
            h = jnp.tanh(layer(h))

        film_params = []
        for head in self.film_heads:
            out = head(h)
            d = out.shape[0] // 2
            scale = out[:d]
            shift = out[d:]
            film_params.append((scale, shift))
        return film_params


# class CoordinateEncoder(eqx.Module):
#     """Encoder for spatiotemporal coordinates (x, t)."""
#     layers: list

#     def __init__(self, key, hidden_dim: int = 64, num_layers: int = 3):
#         keys = jr.split(key, num_layers)
#         dims = [2] + [hidden_dim] * num_layers
#         self.layers = []
#         for i in range(len(dims) - 1):
#             self.layers.append(eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]))

#     def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
#         h = jnp.stack([x, t])
#         for layer in self.layers:
#             h = jax.nn.silu(layer(h))
#         return h


class FiLMManifoldNetwork(eqx.Module):
    """Manifold network where each hidden layer is modulated by FiLM parameters.
    
    At each layer: h = activation(scale_i * W_i @ h + b_i + shift_i)
    
    The scale multiplies after the linear transform (pre-activation),
    giving gamma direct control over feature magnitudes.
    """
    layers: list

    def __init__(self, key, input_dim: int, hidden_dim: int = 64, num_layers: int = 5):
        keys = jr.split(key, num_layers)
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [1]
        self.layers = []
        for i in range(num_layers):
            self.layers.append(eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]))

    def __call__(self, h: jnp.ndarray, film_params: list) -> jnp.ndarray:
        # film_params modulate the hidden layers (all except the last)
        for layer, (scale, shift) in zip(self.layers[:-1], film_params):
            h = layer(h)
            # FiLM: elementwise scale and shift
            # scale is centered around 1 so that the default behavior (gamma=0)
            # is close to an unmodulated network
            h = (1.0 + scale) * h + shift
            h = jax.nn.silu(h)

        return jax.nn.softplus(self.layers[-1](h).squeeze())

        # raw = self.layers[-1](h).squeeze()

        # return jnp.exp(jnp.clip(raw, -20.0, 20.0))


class P2INN_FiLM(eqx.Module):
    """P2INN with FiLM conditioning instead of concatenation."""
    param_encoder: FiLMParameterEncoder
    coord_encoder: CoordinateEncoder
    manifold: FiLMManifoldNetwork

    def __init__(
        self,
        key,
        param_hidden_dim: int = 150,
        param_num_trunk_layers: int = 3,
        coord_hidden_dim: int = 64,
        coord_num_layers: int = 3,
        manifold_hidden_dim: int = 64,
        manifold_num_layers: int = 5,
    ):
        k1, k2, k3 = jr.split(key, 3)

        # Number of FiLM-modulated layers = manifold hidden layers (all except output)
        num_film_layers = manifold_num_layers - 1

        self.param_encoder = FiLMParameterEncoder(
            k1, param_hidden_dim, param_num_trunk_layers,
            num_film_layers, manifold_hidden_dim
        )
        self.coord_encoder = CoordinateEncoder(k2, coord_hidden_dim, coord_num_layers)
        self.manifold = FiLMManifoldNetwork(
            k3, coord_hidden_dim, manifold_hidden_dim, manifold_num_layers
        )

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
        film_params = self.param_encoder(gamma)
        h_coord = self.coord_encoder(x, t)
        return self.manifold(h_coord, film_params)