from __future__ import annotations

import abc
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float


class AbstractModel(eqx.Module):
    """Base class for all models, defining the per-point evaluation API.

    A model maps a single point -- coordinates plus optional auxiliary
    inputs -- to a scalar. Coordinates are passed as individual scalar
    arguments; auxiliary inputs (PDE parameters, discretized functions,
    initial conditions, ...) are passed as a single trailing tuple. This
    per-point, scalar-output contract is what lets `D` take clean per-
    coordinate derivatives and lets the eval_grid utilities batch the model
    over arbitrary coordinate / parameter grids.

    Subclasses implement `_eval(coords, aux_inputs)`, where `coords` is the
    stacked coordinate array and `aux_inputs` is the (possibly empty) aux
    tuple.
    """

    @abc.abstractmethod
    def _eval(
        self,
        coords: Float[Array, "num_coords"],
        aux_inputs: tuple,
    ) -> Float[Array, ""]:
        """Evaluate the model at one point.

        Args:
            coords (Float[Array, 'num_coords']): stacked coordinate array,
                one entry per coordinate axis (e.g. [x, t]).
            aux_inputs (tuple): auxiliary inputs for this point (PDE
                parameters, sensor-sampled functions, ...). Empty for models
                with no auxiliary inputs.

        Returns:
            Float[Array, '']: scalar model output at the point.
        """
        raise NotImplementedError

    def __call__(self, *args):
        """Evaluate the model at one point.

        Coordinates are passed as individual scalars; auxiliary inputs are
        passed as a single trailing TUPLE. The aux tuple may be omitted for
        models with no auxiliary inputs -- since coordinates are scalars and
        aux is by convention always a tuple, the trailing argument's type
        disambiguates:

            model(x, t, (a, b))   # parametrized
            model(x, t, ())       # explicit empty aux -- equivalent to:
            model(x, t)           # pure PINN

        Note `aux` must be a tuple specifically (not a list or array) for
        this dispatch and for eval_grid's per-leaf vmapping.

        Args:
            *args: the coordinate scalars, optionally followed by the aux
                tuple.

        Returns:
            Float[Array, '']: scalar model output at the point.
        """
        if args and isinstance(args[-1], tuple):
            *coords, aux_inputs = args
        else:
            coords, aux_inputs = args, ()
        return self._eval(jnp.stack(coords), aux_inputs)

    def D(self, *argnums):
        """Build the derivative of the model w.r.t. one or more coordinates.

        Returns a function with the same call signature as the model whose
        output is the requested derivative. Each entry of `argnums` indexes
        a coordinate argument (0 -> first coordinate, 1 -> second, ...);
        chaining differentiates repeatedly, so D(0, 0) is the second
        derivative w.r.t. coordinate 0.

        Ex:
            u_x  = model.D(0)(x, t, params)      # d/dx
            u_xx = model.D(0, 0)(x, t, params)   # d2/dx2
            u_t  = model.D(1)(x, t, params)      # d/dt

        Args:
            *argnums (int): coordinate indices to differentiate with respect
                to, applied left to right.

        Returns:
            Callable: a function (same signature as the model) returning the
                requested scalar derivative. Differentiation is per
                coordinate; do not pass the aux-tuple index here.
        """
        f = self.__call__
        for a in argnums:
            f = jax.grad(f, argnums=a)
        return f


# ──────────────────────────────────────────────────────────────
# Abstract Model Classes
# ──────────────────────────────────────────────────────────────


class AbstractP2INN(AbstractModel):
    """Parameterized Physics-Informed Neural Network (abstract).

    Composes a parameter encoder, a coordinate encoder, and a manifold
    network: the parameters and coordinates are encoded separately, their
    embeddings concatenated, and the manifold network maps the result to a
    scalar. Subclasses must provide the three sub-networks as fields.
    """

    param_encoder: eqx.AbstractVar[eqx.nn.MLP]
    coord_encoder: eqx.AbstractVar[eqx.nn.MLP]
    manifold: eqx.AbstractVar[eqx.nn.MLP]

    def _eval(self, coords: Float[Array, "num_coords"], aux_inputs: tuple) -> Float[Array, ""]:
        """Encode parameters and coordinates separately, then combine.

        Args:
            coords (Float[Array, 'num_coords']): stacked coordinate array,
                shape (num_coords,).
            aux_inputs (tuple): tuple of scalar PDE parameters, stacked
                internally into a shape-(num_params,) array for the encoder.

        Returns:
            Float[Array, '']: scalar solution estimate.
        """
        # aux_inputs is a tuple of scalar parameters; stack into an array
        # for the encoder, same trick __call__ uses for the coordinates.
        h_param = self.param_encoder(jnp.stack(aux_inputs))
        h_coord = self.coord_encoder(coords)
        h_concat = jnp.concatenate([h_param, h_coord])
        return self.manifold(h_concat)


class AbstractDeepONet(AbstractModel):
    """Deep Operator Network (abstract).

    Evaluates a sum over a product of branch and trunk embeddings: the trunk
    encodes the coordinates, each branch encodes one auxiliary input (a
    sensor-sampled function), and their elementwise product is summed and
    offset by a bias. Subclasses must provide the branch list, trunk, and
    bias as fields.
    """

    branches: eqx.AbstractVar[list[eqx.nn.MLP]]
    trunk: eqx.AbstractVar[eqx.nn.MLP]
    bias: eqx.AbstractVar[Array]

    def _eval(self, coords: Float[Array, "num_coords"], aux_inputs: tuple) -> Float[Array, ""]:
        """Combine trunk (coordinate) and branch (function) embeddings.

        Args:
            coords (Float[Array, 'num_coords']): stacked coordinate array,
                shape (num_coords,), fed to the trunk.
            aux_inputs (tuple): one sensor-sampled function per branch;
                aux_inputs[i] is fed to branches[i]. Indexing works whether
                aux_inputs is a tuple of arrays or a grouped pytree.

        Returns:
            Float[Array, '']: scalar operator output at the point.
        """
        h = self.trunk(coords)
        for idx, branch in enumerate(self.branches):
            h = h * branch(aux_inputs[idx])
        return jnp.sum(h) + self.bias


# ──────────────────────────────────────────────────────────────
# Concrete Model Classes
# ──────────────────────────────────────────────────────────────


class PINN(AbstractModel):
    """Standard Physics-Informed Neural Network (no auxiliary inputs).

    A single MLP mapping coordinates to a scalar. Since it has no auxiliary
    inputs, call it as model(x, t) (or model(x, t, ()) explicitly); the aux
    tuple is accepted and ignored for API consistency with the parametrized
    models.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        key: jr.PRNGKey,
        num_coords: int = 2,
        hidden_dim: int = 64,
        depth: int = 4,
        inner_activation: Callable = jnp.tanh,
        final_activation: Callable = jax.nn.softplus,
        mlp_kwargs: dict = {},
    ):
        """Build the PINN.

        Args:
            key (jr.PRNGKey): PRNG key for MLP initialization.
            num_coords (int): number of coordinate inputs (e.g. 2 for (x, t)).
            hidden_dim (int): width of each hidden layer.
            depth (int): number of hidden layers.
            inner_activation (Callable): activation between hidden layers.
            final_activation (Callable): activation on the output. The default softplus
            keeps the solution positive; override for sign-changing solutions.
            mlp_kwargs (dict): extra keyword arguments forwarded to
                eqx.nn.MLP.
        """
        self.mlp = eqx.nn.MLP(
            in_size=num_coords,
            out_size="scalar",
            width_size=hidden_dim,
            depth=depth,
            activation=inner_activation,
            final_activation=final_activation,
            key=key,
            **mlp_kwargs,
        )

    def _eval(
        self,
        coords: Float[Array, "num_coords"],
        aux_inputs: tuple,
    ) -> Float[Array, ""]:
        """Map coordinates to a scalar; aux_inputs is ignored.

        Args:
            coords (Float[Array, 'num_coords']): stacked coordinate array.
            aux_inputs (tuple): accepted for API consistency and not used.

        Returns:
            Float[Array, '']: scalar solution estimate.
        """
        # aux_inputs are not used, included for API consistency
        return self.mlp(coords)


class P2INN(AbstractP2INN):
    """Parametrized Physics-Informed Neural Network.

    Concrete AbstractP2INN with explicit parameter/coordinate encoders and a
    manifold network. Set num_params and num_coords to match the problem.

    Usage:
        # one parameter gamma, coordinates (x, t)
        model = P2INN(key, num_params=1, num_coords=2)
        u = model(x, t, (gamma,))

        # two parameters, coordinates (x, y, t)
        model = P2INN(key, num_params=2, num_coords=3)
        u = model(x, y, t, (gamma, beta))
    """

    param_encoder: eqx.nn.MLP
    coord_encoder: eqx.nn.MLP
    manifold: eqx.nn.MLP

    def __init__(
        self,
        key: jr.PRNGKey,
        num_params: int = 1,
        num_coords: int = 2,
        param_hidden_dim=150,
        param_depth=4,
        coord_hidden_dim=64,
        coord_depth=3,
        manifold_inner_dim=64,
        manifold_depth=5,
        param_activation=jnp.tanh,
        coord_activation=jax.nn.silu,
        manifold_inner_activation=jnp.tanh,
        manifold_final_activation=jax.nn.softplus,
        param_kwargs: dict = {},
        coord_kwargs: dict = {},
        manifold_kwargs: dict = {},
    ):
        """Build the three sub-networks.

        Args:
            key (jr.PRNGKey): PRNG key; split three ways for the encoders and
                manifold.
            num_params (int): number of scalar PDE parameters (param-encoder
                input size).
            num_coords (int): number of coordinate inputs (coord-encoder
                input size).
            param_hidden_dim (int): width and output size of the parameter
                encoder.
            param_depth (int): number of hidden layers in the parameter
                encoder.
            coord_hidden_dim (int): width and output size of the coordinate
                encoder.
            coord_depth (int): number of hidden layers in the coordinate
                encoder.
            manifold_inner_dim (int): width of the manifold network's hidden
                layers.
            manifold_depth (int): number of hidden layers in the manifold
                network.
            param_activation (Callable): activation for the parameter encoder
                (used inner and final).
            coord_activation (Callable): activation for the coordinate
                encoder (used inner and final).
            manifold_inner_activation (Callable): activation between the
                manifold network's hidden layers.
            manifold_final_activation (Callable): activation on the manifold
                output. The default softplus keeps the solution positive;
                override for sign-changing solutions.
            param_kwargs (dict): extra kwargs forwarded to the parameter
                encoder MLP.
            coord_kwargs (dict): extra kwargs forwarded to the coordinate
                encoder MLP.
            manifold_kwargs (dict): extra kwargs forwarded to the manifold
                MLP.
        """
        k1, k2, k3 = jr.split(key, 3)

        self.param_encoder = eqx.nn.MLP(
            in_size=num_params,
            out_size=param_hidden_dim,
            width_size=param_hidden_dim,
            depth=param_depth,
            activation=param_activation,
            final_activation=param_activation,
            key=k1,
            **param_kwargs,
        )

        self.coord_encoder = eqx.nn.MLP(
            in_size=num_coords,
            out_size=coord_hidden_dim,
            width_size=coord_hidden_dim,
            depth=coord_depth,
            activation=coord_activation,
            final_activation=coord_activation,
            key=k2,
            **coord_kwargs,
        )

        manifold_input_dim = self.param_encoder.out_size + self.coord_encoder.out_size

        self.manifold = eqx.nn.MLP(
            in_size=manifold_input_dim,
            out_size="scalar",  # scalar output so jax.grad / D works
            width_size=manifold_inner_dim,
            depth=manifold_depth,
            activation=manifold_inner_activation,
            final_activation=manifold_final_activation,
            key=k3,
            **manifold_kwargs,
        )


class DeepONet(AbstractDeepONet):
    """Deep Operator Network with one branch per auxiliary function.

    Concrete AbstractDeepONet. Each entry of branch_input_dim adds a branch
    MLP whose input size is that entry's sensor count; the trunk takes the
    coordinates. Call as model(x, t, (f1, f2, ...)) where f_i is the i-th
    branch's sensor-sampled function and matches branches[i]'s input size.
    """

    branches: list[eqx.nn.MLP]
    trunk: eqx.nn.MLP
    bias: Array

    def __init__(
        self,
        key: jr.PRNGKey,
        branch_input_dim: tuple[int],
        trunk_input_dim: int,
        branch_trunk_output_dim: int = 100,
        branch_depth: tuple = (5,),
        trunk_depth: int = 3,
        branch_kwargs: dict = {"activation": jnp.tanh},
        trunk_kwargs: dict = {"activation": jnp.tanh},
    ):
        """Build the branch networks, trunk, and bias.

        Args:
            key (jr.PRNGKey): PRNG key; consumed iteratively to initialize
                each branch, the trunk, and the bias.
            branch_input_dim (tuple[int]): sensor count for each branch; its
                length is the number of branches (and the expected length of
                the aux tuple at call time).
            trunk_input_dim (int): number of coordinate inputs to the trunk.
            branch_trunk_output_dim (int): shared embedding width; the output
                size (and hidden width) of every branch and the trunk, so
                their embeddings can be multiplied elementwise.
            branch_depth (tuple): number of hidden layers per branch; one
                entry per branch (indexed alongside branch_input_dim).
            trunk_depth (int): number of hidden layers in the trunk.
            branch_kwargs (dict): extra kwargs forwarded to every branch MLP.
            trunk_kwargs (dict): extra kwargs forwarded to the trunk MLP.
        """

        key, trunk_key, bias_key = jr.split(key, 3)

        self.bias = jr.uniform(bias_key, minval=-1.0, maxval=1.0)

        self.trunk = eqx.nn.MLP(
            in_size=trunk_input_dim,
            out_size=branch_trunk_output_dim,
            width_size=branch_trunk_output_dim,
            depth=trunk_depth,
            key=trunk_key,
            **trunk_kwargs,
        )

        self.branches = []
        branch_keys = jr.split(key, len(branch_input_dim))
        for idx in range(len(branch_input_dim)):
            # Re-derive a fresh key each iteration (keep the first split half,
            # discard the second) so branches initialize independently.
            self.branches.append(
                eqx.nn.MLP(
                    in_size=branch_input_dim[idx],
                    out_size=branch_trunk_output_dim,
                    width_size=branch_trunk_output_dim,
                    depth=branch_depth[idx],
                    key=branch_keys[idx],
                    **branch_kwargs,
                )
            )
