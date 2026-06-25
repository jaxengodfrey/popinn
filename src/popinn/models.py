from __future__ import annotations

import abc
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float


class AbstractModel(eqx.Module):

    @abc.abstractmethod
    def _eval(self, 
              coords: Float[Array, 'num_coords'],
              aux_inputs: tuple
        ) -> Float[Array, '']:
        raise NotImplementedError

    def __call__(self, *args):
        """Evaluates the model. Inputs should be coordinates as individual elements
        and auxilary inputs (e.g. PDE coefficients, functions, ICs) in a single array.
        Ex: 
            ```
            model = P2INN(key)
            model(x, t, params)
            ```
        """
        if args and isinstance(args[-1], tuple):
            *coords, aux_inputs = args
        else:
            coords, aux_inputs = args, ()
        return self._eval(jnp.stack(coords), aux_inputs)
    
    def D(self, *argnums):
        """Function to compute the derivative w.r.t. the indexed argument. 
        Ex:
        u_xx = model.D(0,0)(x, t, params)  #second derivative w.r.t. x
        """
        f = self.__call__
        for a in argnums:
            f = jax.grad(f, argnums = a)
        return f

# ──────────────────────────────────────────────────────────────
# Abstract Model Classes
# ──────────────────────────────────────────────────────────────

class AbstractP2INN(AbstractModel):
    """Parameterized Physics-Informed Neural Network.

    Composes one parameter encoder, one coordinate encoder,
    and one manifold network. Inputs are pre-stacked arrays.

    Subclasses must provide the three sub-networks.
    """

    param_encoder: eqx.AbstractVar[eqx.nn.MLP]
    coord_encoder: eqx.AbstractVar[eqx.nn.MLP]
    manifold: eqx.AbstractVar[eqx.nn.MLP]

    def _eval(self, coords: Float[Array, 'num_coords'], aux_inputs: tuple) -> Float[Array, '']:
        """
        Args:
            params: tuple of scalar parameters, shape (num_params,)
            coords: stacked coordinate array, shape (num_coords,)
        Returns:
            Scalar solution estimate.
        """
        # aux_inputs is a tuple of scalar parameters; stack into an array
        # for the encoder, same trick __call__ uses for the coordinates.
        h_param = self.param_encoder(jnp.stack(aux_inputs))
        h_coord = self.coord_encoder(coords)
        h_concat = jnp.concatenate([h_param, h_coord])
        return self.manifold(h_concat)
    


class AbstractDeepONet(AbstractModel):
    branches: eqx.AbstractVar[list[eqx.nn.MLP]]
    trunk: eqx.AbstractVar[eqx.nn.MLP]
    bias: eqx.AbstractVar[Array]

    def _eval(self, coords: Float[Array, 'num_coords'], aux_inputs: tuple) -> Float[Array, '']:
        h = self.trunk(coords)
        for idx, branch in enumerate(self.branches):
            h = h * branch(aux_inputs[idx])
        return jnp.sum(h) + self.bias


# ──────────────────────────────────────────────────────────────
# Concrete Model Classes
# ──────────────────────────────────────────────────────────────

class PINN(AbstractModel):
    mlp: eqx.nn.MLP
    
    def __init__(self,
                 key,
                 num_coords: int = 2,
                 hidden_dim: int = 64,
                 depth: int = 4,
                 inner_activation = jnp.tanh,
                 final_activation = jnp.tanh,
                 mlp_kwargs: dict = {}):
        
        self.mlp = eqx.nn.MLP(
                            in_size = num_coords,
                            out_size = 'scalar',
                            width_size = hidden_dim,
                            depth = depth,
                            activation = inner_activation,
                            final_activation = final_activation,
                            key = key,
                            **mlp_kwargs)
        
    def _eval(self, 
              coords: Float[Array, 'num_coords'],
              aux_inputs: Float[Array, '']
              ):
        # aux_inputs are not used, included for API consistency
        return self.mlp(coords)


class P2INN(AbstractP2INN):
    """Parametrized Physics Informed Neural Network. Specify num_params and num_coords to set input dims.

    Usage:
        model = P2INN(key, num_params=1, num_coords=2)
        u = model(params=jnp.array([gamma]), coords=jnp.array([x, t]))

        model = P2INN(key, num_params=2, num_coords=3)
        u = model(params=jnp.array([gamma, beta]), coords=jnp.array([x, y, t]))
    """

    param_encoder: eqx.nn.MLP
    coord_encoder: eqx.nn.MLP
    manifold: eqx.nn.MLP

    def __init__(
        self,
        key: jr.PRNGKey,
        num_params: int = 1,
        num_coords: int = 2,
        param_hidden_dim = 150,
        param_depth = 4,
        coord_hidden_dim = 64,
        coord_depth = 3,
        manifold_inner_dim = 64,
        manifold_depth = 5,
        param_activation = jnp.tanh,
        coord_activation = jax.nn.silu,
        manifold_inner_activation = jnp.tanh,
        manifold_final_activation = jax.nn.softplus,
        param_kwargs: dict = {},
        coord_kwargs: dict = {},
        manifold_kwargs: dict = {},
    ):
        k1, k2, k3 = jr.split(key, 3)

        self.param_encoder = eqx.nn.MLP(
            in_size = num_params, 
            out_size = param_hidden_dim, 
            width_size = param_hidden_dim, 
            depth = param_depth,
            activation = param_activation,
            final_activation = param_activation,
            key = k1, 
            **param_kwargs
            )
        
        self.coord_encoder = eqx.nn.MLP(
            in_size = num_coords, 
            out_size = coord_hidden_dim,
            width_size = coord_hidden_dim,
            depth = coord_depth,
            activation = coord_activation,
            final_activation = coord_activation,
            key = k2, 
            **coord_kwargs
            )

        manifold_input_dim = self.param_encoder.out_size + self.coord_encoder.out_size

        self.manifold = eqx.nn.MLP(
            in_size = manifold_input_dim,
            out_size = "scalar", # scalar output so jax.grad / D works
            width_size = manifold_inner_dim,
            depth = manifold_depth,
            activation = manifold_inner_activation,
            final_activation = manifold_final_activation,
            key = k3,
            **manifold_kwargs
            )



class DeepONet(AbstractDeepONet):

    branches: list[eqx.nn.MLP]
    trunk: eqx.nn.MLP
    bias: Array

    def __init__(self, 
                 key, 
                 branch_input_dim: tuple[int],
                 trunk_input_dim: int,
                 branch_trunk_output_dim: int = 100,
                 branch_depth: tuple = (5,),
                 trunk_depth: int = 3,
                 branch_kwargs: dict = {'activation': jnp.tanh},
                 trunk_kwargs: dict = {'activation': jnp.tanh}):
        
        self.branches = []
        for idx in range(len(branch_input_dim)):
            key, _ = jr.split(key)
            self.branches.append(
                eqx.nn.MLP(
                    in_size = branch_input_dim[idx],
                    out_size = branch_trunk_output_dim,
                    width_size = branch_trunk_output_dim,
                    depth = branch_depth[idx],
                    key = key,
                    **branch_kwargs)
                        )

        k1, k2 = jr.split(key, 2)
        self.trunk = eqx.nn.MLP(
                        in_size = trunk_input_dim,
                        out_size = branch_trunk_output_dim,
                        width_size = branch_trunk_output_dim,
                        depth = trunk_depth,
                        key = k1,
                        **trunk_kwargs
                        )
        
        self.bias = jr.uniform(k2, minval = -1., maxval = 1.)