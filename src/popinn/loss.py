import abc
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .eval import eval_grid
from .models import AbstractModel


def _reduce(values: Array, metric: str | Callable) -> Float[Array, ""]:
    """
    Reduce a grid of residual values to a scalar term loss.

    Args:
        values (Array): jax array of residual values.
        metric (str | Callable): metric to compute the loss. Can be 'mse'
            for mean-squared-err, 'mae' for mean-absolute-error, or a
            custom Callable mapping the grid of values to a scalar.

    Returns:
        Float[Array, '']: scalar loss value.

    Raises:
        ValueError: if a custom callable metric returns a non-scalar, or if
            `metric` is a string other than 'mse' or 'mae'.
    """
    if callable(metric):
        out = metric(values)
        if jnp.shape(out) != ():
            raise ValueError(f"ResidualTerm metrics should be scalar-output functions. Output had shape: {jnp.shape(out)}")
        return out
    if metric == "mse":
        return jnp.mean(values**2)
    if metric == "mae":
        return jnp.mean(jnp.abs(values))
    raise ValueError(f"unknown metric {metric!r}; use 'mse', 'mae', or a custom callable")


class ResidualTerm(eqx.Module):
    """
    Module for evaluating a residual function over a grid of coordinate and auxiliary values,
    then reducing the residual grid to a scalar value with a specified metric.
    """

    name: str = eqx.field(static=True)
    residual_fn: Callable = eqx.field(static=True)
    metric: str | Callable = eqx.field(static=True, default="mse")
    eval_fn: Callable = eqx.field(static=True, default=eval_grid)
    batch_size: int | None = eqx.field(static=True, default=None)

    def __call__(self, model: AbstractModel, data: eqx.Module) -> Float[Array, ""]:
        """
        Evaluate the residual over this term's coordinate and auxiliary grid and reduce to a scalar.

        Coordinates are read from `data.<name>_coords` (where `<name>` is the
        `name` specified at initialization) and the auxiliary
        inputs from `data.aux`.

        Args:
            model (AbstractModel): The network model.
            data (equinox.Module): data container with field carrying the 1D coordinate
            grid axes in `<name>_coords` and the auxiliary inputs in `aux`.

        Returns:
            (Float[Array, '']): the scalar loss for this term.
        """
        coords = getattr(data, self.name + "_coords")
        aux = data.aux
        kw = {}
        if self.batch_size is not None:
            key = "batch_size" if self.eval_fn is not eval_grid else "outer_batch_size"
            kw[key] = self.batch_size
        values = self.eval_fn(self.residual_fn(model), coords, aux, **kw)
        return _reduce(values, self.metric)


ResidualTerm.__init__.__doc__ = """.


Args:
    name (str): Name of the term. Should match the name of a coordinate field in the `data`
        container as `data.<name>_coords`.
    residual_fn (Callable): Function with the signature
        `residual_fn(model) -> Callable`, where the returned function is
        the per-point residual function and has the signature
        `r(*coords, aux) -> Scalar`.
    metric (str | Callable): Metric to compute the loss.
        Can be 'mse' for mean-squared-err, 'mae' for mean-absolute-error,
        or a custom Callable mapping a grid of values to a scalar.
    eval_fn (eval_grid | eval_grid_flat_aux):
        Function that evaluates the model over a grid of coordinates
        and auxiliary inputs. Can be `eval_grid` or `eval_grid_flat_aux`.
    batch_size (int | None): The batch sizes for the
        memory dial in `eval_fn`. Setting to `None` evaluates the whole grid at once (fastest, most
        memory).
"""


class AbstractWeights(eqx.Module):
    """
    Abstract base class for per-term loss weighting schemes.

    Subclasses store the weighting parameters in `values` and implement the
    `combine` method, which folds the per-term scalar residuals into the total weighted loss.

    **Fields**:

    - `values` (`dict`): the weights, keyed by the set of corresponding `ResidualTerm.name` strings. Whether the
        weights are static constants or trainable leaves is decided by the concrete subclass.
    """

    values: eqx.AbstractVar[dict]

    @abc.abstractmethod
    def combine(self, residuals: dict) -> Float[Array, ""]:
        """
        Combine per-term scalar residuals into the weighted total.

        Args:
            residuals (dict): A mapping from each `ResidualTerm.name` to that term's scalar
                value.

        Returns:
            (Float[Array, '']): The total weighted loss.
        """
        raise NotImplementedError


class FixedWeights(AbstractWeights):
    """
    Static (non-trainable) per-term loss weights.

    **Fields**:

    - `values` (`dict`): Mapping from term name to its scalar weight, keyed by the set of corresponding `ResidualTerm.name` strings.
    """

    values: dict = eqx.field(static=True)

    def combine(self, residuals: dict) -> Float[Array, ""]:
        """
        Combine per-term scalar losses into the weighted sum.

        Args:
            residuals (dict): A mapping from each `ResidualTerm.name` to that term's scalar
                value.

        Returns:
            (Float[Array, '']): The weighted sum over all loss terms.
        """
        return sum(self.values[k] * residuals[k] for k in self.values.keys())


class Loss(eqx.Module):
    """
    Composite physics-informed loss assembled from a list of initialized `ResidualTerm` objects and a specified weighting scheme.

    """

    res_terms: list[ResidualTerm]
    weights: AbstractWeights | None = None
    include_weights: bool = True

    def __init__(self, res_terms: list[ResidualTerm], weights: AbstractWeights | None = None, include_weights: bool = True):
        """.

        Args:
            res_terms (list[ResidualTerm]): A list of initialized `ResidualTerm` objects. The `name` fields of each `ResidualTerm` must be unique.
            weights (AbstractWeights | None): A weighting object whose `values` keys match the names of the `ResidualTerm` objects passed to `res_terms`. When `None`, defaults to `FixedWeights` with a weight equal to 1.0 for every term.
            include_weights (bool): If `True`, the individual loss terms in the dictionary returned by `__call__` will be scaled by their weights.
                If `False`, the weights will not be included.
        """
        self.include_weights = include_weights
        self.res_terms = res_terms
        keys = [t.name for t in res_terms]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate term names in res_terms: {keys}")
        if weights is None:
            weights = FixedWeights(values={k: 1.0 for k in keys})
        elif set(weights.values.keys()) != set(keys):
            raise ValueError(f"weight keys {sorted(weights.values)} do not match term names {sorted(keys)}")
        self.weights = weights

    def __call__(self, model: AbstractModel, data: eqx.Module) -> tuple[Float[Array, ""], dict]:
        """
        Evaluate all terms and return the weighted total plus a dictionary with
        the individual loss terms (weighted or un-weighted determined by `include_weights` kwarg specified at initialization).

        Args:
            model (AbstractModel): The network model called in the per-point residual functions.
            data (eqx.Module): Data container; each `ResidualTerm` reads its
                coordinates as `data.<name>_coords` and the shared auxiliary
                inputs as `data.aux`.

        Returns:
            (tuple[Float[Array, ''], dict]): The scalar weighted total loss
                and a dict mapping each term name to its (weighted or un-weighted) scalar
                loss.
        """
        residuals = {}
        for term in self.res_terms:
            residuals[term.name] = term(model, data)
        total = self.weights.combine(residuals)
        if self.include_weights:
            for term in self.res_terms:
                residuals[term.name] = self.weights.values[term.name] * residuals[term.name]
        return total, residuals
