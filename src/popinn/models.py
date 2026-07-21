from __future__ import annotations

import abc
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float


class AbstractModel(eqx.Module):
    """Abstract base class for all models.

    The `__call__` and `D` (derivative) methods are concrete and should not be overridden by subclasses.

    Subclasses must implement the abstract hook `_eval(coords, aux_inputs)`, which `__call__` dispatches to.
    """

    @abc.abstractmethod
    def _eval(
        self,
        coords: Float[Array, "num_coords"],
        aux_inputs: tuple,
    ) -> Float[Array, ""]:
        """Abstract method.

        Evaluates the model at one grid point.

        Args:
            coords (Float[Array, 'num_coords']): Stacked coordinate array,
                one entry per coordinate axis (e.g. `jnp.array([x, t])`).
            aux_inputs (tuple): Auxiliary inputs (e.g. PDE
                parameters, sensor-sampled functions, etc.). Empty for models
                with no auxiliary inputs (PINN).

        Returns:
            (Float[Array, '']): The model output as a scalar JAX array.
        """
        raise NotImplementedError

    def __call__(self, *args):
        """Evaluate the model at one point.

        Coordinates are passed as individual scalars; auxiliary inputs are
        passed as a single trailing *tuple*:

            model(x, t, (a, b))   # x & t are the coordinates; a & b are the auxiliary inputs.

        For models with no auxiliary inputs, e.g. `popinn.PINN`, the auxiliary tuple
        can be empty, or omitted completely:

            model(x, t, ())       # explicit empty aux -- equivalent to:
            model(x, t)           # no aux

        Note the auxiliary inputs *must be a tuple* specifically, not a list
        or array.

        Args:
            *args: The coordinate scalars, optionally followed by the aux
                tuple.

        Returns:
            (Float[Array, '']): The model output as a scalar JAX array.
        """
        if args and isinstance(args[-1], tuple):
            *coords, aux_inputs = args
        else:
            coords, aux_inputs = args, ()
        return self._eval(jnp.stack(coords), aux_inputs)

    def D(self, *argnums):
        """Build the derivative of the model with respect to one or more coordinates.

        Returns a function with the same call signature as the model whose
        output is the requested derivative.

        !!! Example "Derivative syntax"
            Each entry of `argnums` indexes a coordinate argument (`0` -> first coordinate, `1` -> second, ...);
            chaining differentiates repeatedly. For example, for a model with two coordinates `x` & `t`,
            derivatives are taken & evaluated like:
            ```python
                du_dx  = model.D(0)(x, t, aux)         # du/dx
                d2u_dx2 = model.D(0, 0)(x, t, aux)     # d2/dx2
                du_dt  = model.D(1)(x, t, aux)         # d/dt


            ```

        Args:
            *argnums (int): Coordinate indices to differentiate with respect
                to, applied left to right.

        Returns:
            (Callable): A function with the same signature as the model that returns the
                requested scalar derivative.
        """
        f = self.__call__
        for a in argnums:
            f = jax.grad(f, argnums=a)
        return f


# ──────────────────────────────────────────────────────────────
# Abstract Model Classes
# ──────────────────────────────────────────────────────────────


class AbstractP2INN(AbstractModel):
    """Abstract Parameterized Physics-Informed Neural Network.

    Composes a parameter encoder, a coordinate encoder, and a manifold
    network: the coordinates and parameters are encoded separately, their
    embeddings concatenated, and the manifold network maps to a
    scalar. Subclasses must provide the three sub-networks as fields.
    """

    param_encoder: eqx.AbstractVar[eqx.nn.MLP]
    coord_encoder: eqx.AbstractVar[eqx.nn.MLP]
    manifold: eqx.AbstractVar[eqx.nn.MLP]

    def _eval(self, coords: Float[Array, "num_coords"], aux_inputs: tuple) -> Float[Array, ""]:
        """Concatenate coordinate and parameter embeddings, then pass through manifold network.

        Args:
            coords (Float[Array, 'num_coords']): Array of coordinate values, one coordinate per axis, e.g. `jnp.array([x,t])` for scalars `x` and `t`.
            aux_inputs (tuple): Tuple of scalar PDE parameters, e.g. `(a,b)` for scalar values `a` and `b`.

        Returns:
            (Float[Array, '']): The model output as a scalar JAX array.
        """

        h_param = self.param_encoder(jnp.stack(aux_inputs))
        h_coord = self.coord_encoder(coords)
        h_concat = jnp.concatenate([h_param, h_coord])
        return self.manifold(h_concat)


class AbstractDeepONet(AbstractModel):
    """Abstract Deep Operator Network.

    Evaluates a sum over a product of branch and trunk embeddings: the trunk
    encodes the coordinates, each branch encodes one auxiliary input (e.g., a
    sensor-sampled function), and their elementwise product is summed and
    offset by a bias. Subclasses must provide the branch list, trunk, and
    bias as fields.
    """

    branches: eqx.AbstractVar[list[eqx.nn.MLP]]
    trunk: eqx.AbstractVar[eqx.nn.MLP]
    bias: eqx.AbstractVar[Array]

    def _eval(self, coords: Float[Array, "num_coords"], aux_inputs: tuple) -> Float[Array, ""]:
        """Sum element-wise product of trunk (coordinate) and branch (auxiliary) embeddings and add bias.

        Args:
            coords (Float[Array, 'num_coords']): Stacked coordinate array,
                shape (num_coords,), fed to the trunk.
            aux_inputs (tuple): Tuple containing arrays (sensor-sampled function) and/or scalars (PDE parameters).
                Each top level element of the tuple is mapped to its own branch network.

        Returns:
            (Float[Array, '']): The model output as a scalar JAX array.
        """
        h = self.trunk(coords)
        for idx, branch in enumerate(self.branches):
            h = h * branch(jnp.atleast_1d(aux_inputs[idx]))
        return jnp.sum(h) + self.bias


# ──────────────────────────────────────────────────────────────
# Concrete Model Classes
# ──────────────────────────────────────────────────────────────


class PINN(AbstractModel):
    """Physics-Informed Neural Network.

    A single MLP mapping coordinates to a scalar, with no auxiliary inputs.

    See [Raissi et al. (2019)](https://www.sciencedirect.com/science/article/pii/S0021999118307125) for more details on PINNs.
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
        """Initialize the PINN as a multi-layer perceptron using `equinox.nn.MLP`.

        Args:
            key (jr.PRNGKey): PRNG key for MLP initialization.
            num_coords (int): Number of coordinate inputs (e.g. 2 for x & t).
            hidden_dim (int): Width of each hidden layer.
            depth (int): Number of hidden layers.
            inner_activation (Callable): Activation between hidden layers.
            final_activation (Callable): Activation on the output.
            mlp_kwargs (dict): Extra keyword arguments forwarded to
                `equinox.nn.MLP`.
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

    Consists of 3 MLPs: a coordinate encoder, parameter encoder, and
    manifold network. The outputs of the two encoders are stacked together
    and then passed into the manifold network, which returns the scalar
    model output.

    The parameters can be scalar PDE parameters, e.g. the scalar parameters
    $\\beta$, $\\nu$, and $\\rho$ in this PDE:

    $$
    \\frac{\\partial u}{\\partial t} + \\beta \\frac{\\partial u}{\\partial x} - \\nu \\frac{\\partial^2 u}{\\partial x^2} - \\rho u (1 - u) = 0
    $$

    Example usage:

    ```python
    import jax.numpy as jnp
    import popinn

    beta = jnp.array(1.)
    nu = jnp.array(2.)
    rho = jnp.array(3.)

    x = jnp.array(1.)
    t = jnp.array(1.)

    # initialize
    model = popinn.P2INN(num_params = 3, num_coords = 2)

    # evaluate
    u = model(x, t, (beta, nu, rho))
    ```

    See [Cho et al. (2024)](https://arxiv.org/abs/2408.09446) for more details
    about P$^2$INNs.
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
        """Initialize the three sub-networks: coordinate encoder, parameter encoder, and manifold network,
        each as an `equinox.nn.MLP`.

        Args:
            key (jr.PRNGKey): PRNG key; split three ways for the encoders and
                manifold.
            num_params (int): Number of scalar PDE parameters; defines the input dimension of the parameter encoder.
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
    """Deep Operator Network consisting of a single trunk network and a branch network for each auxiliary input.
    The trunk and branch networks are all `equinox.nn.MLP`.

    The final model output is constructed by taking the sum over an element-wise product between the
    trunk $tr(\\vec{x})$ and branch $br_j(\\vec{\\mu_j})$ outputs, plus a learnable bias $b$. For example,
    for two branches, the output is:

    $$
    \\sum_i \\Big[br_{1, i}(\\vec{\\mu}_1) * br_{2, i}(\\vec{\\mu}_2) * tr_i(\\vec{x})\\Big] + b
    $$

    The trunk and branch outputs must therefore all be the same dimension, which is specified
    by `branch_trunk_output_dim` on initialization.

    The auxiliary inputs are functions evaluated at a fixed set of sensor points, e.g. the
    forcing term $F(t)$ in the PDE:

    $$
    \\frac{\\partial u}{\\partial t} + \\beta \\frac{\\partial u}{\\partial x} - \\nu \\frac{\\partial^2 u}{\\partial x^2} - \\rho u (1 - u) = F(t)
    $$

    and the initial condition $u_0(x)$ that may depend on $\\beta$, $\\nu$, and $\\rho$.

    Example usage:

    ```python
    import jax.numpy as jnp
    import popinn

    beta = jnp.array(1.)
    nu = jnp.array(2.)
    rho = jnp.array(3.)

    ts = jnp.linspace(0,1,10)
    F_t = # some function that depends t
    F_t_vals = F_t(ts)

    xs = jnp.linspace(0,1,10)
    u_0 = # some function that depends on x and may depend on beta, nu, and/or rho
    u_0_vals = u_0(xs)

    x = jnp.array(1.)
    t = jnp.array(1.)

    # initialize
    model = popinn.DeepONet(
                branch_input_dim = (xs.shape[0], ts.shape[0]),
                trunk_input_dim = 2,
                branch_depth = (5, 5)
            )

    # evaluate
    u = model(x, t, (u_0_vals, F_t_vals))
    ```

    See [Lu et al. (2020)](https://arxiv.org/abs/1910.03193) for more details on DeepONets.
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
        branch_kwargs: dict = {"activation": jnp.tanh, "final_activation": jnp.tanh},
        trunk_kwargs: dict = {"activation": jnp.tanh, "final_activation": jnp.tanh},
        bias_min: float = -1.0,
        bias_max: float = 1.0,
    ):
        """Build the branch networks, trunk, and bias.

        Args:
            key (jr.PRNGKey): PRNG key.
            branch_input_dim (tuple[int]): Tuple containing the sensor count for each
                branch.
            trunk_input_dim (int): Number of coordinate inputs to the trunk.
            branch_trunk_output_dim (int): The shared hidden and output width
                of the trunk and branch networks.
            branch_depth (tuple): Number of hidden layers per branch; one
                entry per branch (indexed alongside branch_input_dim).
            trunk_depth (int): Number of hidden layers in the trunk.
            branch_kwargs (dict): Extra kwargs forwarded to every branch MLP.
            trunk_kwargs (dict): Extra kwargs forwarded to the trunk MLP.
        """

        key, trunk_key, bias_key = jr.split(key, 3)

        self.bias = jr.uniform(bias_key, minval=bias_min, maxval=bias_max)

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
