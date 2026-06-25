from .eval import eval_grid, eval_grid_flat_aux
from .models import AbstractModel
from jaxtyping import Array, Float
from collections.abc import Callable
import equinox as eqx
import jax.numpy as jnp
import abc


def _reduce(values: Array, metric: str | Callable) -> Float[Array, '']:
    """
    Reduce a grid of residual values to a scalar term loss.

    Args:
        values (Array): jax array of residual values.
        metric (str | Callable): metric to compute the loss. Can be 'mse'
            for mean-squared-err, 'mae' for mean-absolute-error, or a
            custom Callable.

    Returns:
        Float[Array, '']: scalar loss value.
    """
    if callable(metric):
        return metric(values)
    if metric == "mse":
        return jnp.mean(values**2)
    if metric == "mae":
        return jnp.mean(jnp.abs(values))
    raise ValueError(f"unknown metric {metric!r}; use 'mse', 'mae', or a callable")


class ResidualTerm(eqx.Module):
    """
    Module for computing a grid of residual values and computing a loss term
    given a metric.

    All fields are static (compile-time constants), so a ResidualTerm holds
    no array data and is safe to embed in a jitted Loss. This term's
    coordinates are read from the runtime `data` object at call time as the
    attribute `<name>_coords`, so the same term works for any resampled data
    of matching shapes.

    __init__ args:
        name (str): Name of the loss term. Should match the coords name in
            the data module: the term reads its coordinates from
            `data.<name>_coords`, and `name` is also this term's key in the
            residual / weight dictionaries.
        residual_fn (Callable): function with the signature
            residual_fn(model) -> Callable, where the returned function is
            the per-point residual function and has the signature
            r(*coords, aux) -> Scalar.
        metric (str | Callable, default: 'mse'): metric to compute the loss.
            Can be 'mse' for mean-squared-err, 'mae' for mean-absolute-error,
            or a custom Callable mapping a grid of values to a scalar.
        eval_fn (eval_grid | eval_grid_flat_aux, default: eval_grid):
            function that evaluates the model over the grid of coordinates
            and auxiliary inputs. Can be `eval_grid` (nested vmap; chunks the
            outermost axis via `outer_batch_size`) or `eval_grid_flat_aux`
            (chunks over all parameter combinations jointly via `batch_size`).
        batch_size (int | None, default: None): the size of chunks for the
            memory dial. When set, it is forwarded to eval_fn under the
            correct keyword: `outer_batch_size` for eval_grid, `batch_size`
            otherwise. None evaluates the whole grid at once (fastest, most
            memory).
    """

    name: str = eqx.field(static=True)
    residual_fn: Callable = eqx.field(static=True)
    metric: str | Callable = eqx.field(static=True, default='mse')
    eval_fn: Callable = eqx.field(static=True, default=eval_grid)
    batch_size: int | None = eqx.field(static=True, default=None)

    def __call__(self, model: AbstractModel, data: eqx.Module) -> Float[Array, '']:
        """
        Evaluate the residual over this term's grid and reduce to a scalar.

        Coordinates are read from `data.<name>_coords` and the auxiliary
        inputs from `data.aux`; the per-point residual `residual_fn(model)`
        is batched over them by `eval_fn` and reduced by `metric`.

        Args:
            model (AbstractModel): the model the residual is built around.
            data (eqx.Module): runtime data container exposing this term's
                coordinates as `<name>_coords` and the auxiliary inputs as
                `aux`.

        Returns:
            Float[Array, '']: the scalar loss for this term.
        """
        coords = getattr(data, self.name + '_coords')
        aux = data.aux
        kw = {}
        if self.batch_size is not None:
            key = 'batch_size' if self.eval_fn is not eval_grid else "outer_batch_size"
            kw[key] = self.batch_size
        values = self.eval_fn(self.residual_fn(model), coords, aux, **kw)
        return _reduce(values, self.metric)


class AbstractWeights(eqx.Module):
    """
    Abstract base class for per-term loss weighting schemes.

    Subclasses store the weighting parameters in `values` and implement
    `combine`, which folds the per-term scalar losses into the weighted
    total. Keeping this abstract leaves room for trainable schemes (e.g.
    learned-uncertainty or softmax-normalized weights) alongside the
    fixed-weight default without changing the Loss interface.

    Fields:
        values (dict): the weighting data, keyed by term name. Whether these
            are static constants or trainable leaves is decided by the
            concrete subclass.
    """

    values: eqx.AbstractVar[dict]

    @abc.abstractmethod
    def combine(self, residuals: dict) -> Float[Array, '']:
        """
        Combine per-term scalar losses into the weighted total.

        Args:
            residuals (dict): mapping from term name to that term's scalar
                loss.

        Returns:
            Float[Array, '']: the weighted total loss.
        """
        raise NotImplementedError


class FixedWeights(AbstractWeights):
    """
    Static (non-trainable) per-term loss weights.

    `values` is a static field, so the weights are excluded from
    differentiation automatically and incur no trace-time cost. To change
    them, construct a new FixedWeights rather than mutating in place.

    Fields:
        values (dict): mapping from term name to its scalar weight. Keys
            must cover exactly the term names in the Loss.
    """

    values: dict = eqx.field(static=True)

    def combine(self, residuals: dict) -> Float[Array, '']:
        """
        Combine per-term scalar losses into the weighted sum.

        Args:
            residuals (dict): mapping from term name to that term's scalar
                loss, as returned by each ResidualTerm.

        Returns:
            Float[Array, '']: the weighted sum over all loss terms.
        """
        return sum(self.values[k] * residuals[k] for k in self.values.keys())


class Loss(eqx.Module):
    """
    Composite physics-informed loss assembled from a collection of ResidualTerms.

    Calling the initialized Loss evaluates every term against the model and data and
    returns (total, residuals), where `total` is the weighted sum and
    `residuals` is the per-term breakdown -- a shape suited to
    eqx.filter_value_and_grad(loss, has_aux=True)(model, data).

    __init__ args:
        res_terms (list[ResidualTerm]): the terms to evaluate. Each term's
            `name` is its key in the residual / weight dictionaries and
            selects its coordinates on the data module (`<name>_coords`).
            Names must be unique.
        weights (AbstractWeights | None, default: None): a weighting object
            whose `values` keys match the term names and which exposes
            `combine(residuals) -> scalar`. When None, defaults to
            FixedWeights with weight 1.0 for every term.
    """

    res_terms: list[ResidualTerm]
    weights: AbstractWeights | None

    def __init__(
        self,
        res_terms: list[ResidualTerm],
        weights: AbstractWeights | None = None
    ):
        self.res_terms = res_terms
        keys = [t.name for t in res_terms]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate term names in res_terms: {keys}")
        if weights is None:
            weights = FixedWeights(values={k: 1. for k in keys})
        elif set(weights.values.keys()) != set(keys):
            raise ValueError(
                f"weight keys {sorted(weights.values)} do not match "
                f"term names {sorted(keys)}"
            )
        self.weights = weights

    def __call__(
        self,
        model,
        data
    ) -> tuple[Float[Array, ''], dict]:
        """
        Evaluate all terms and return the weighted total plus a dictionary with
        the individual un-weighted loss terms.

        Args:
            model (AbstractModel): the model the residuals are built around.
            data (eqx.Module): runtime data container; each term reads its
                coordinates as `data.<name>_coords` and the shared auxiliary
                inputs as `data.aux`.

        Returns:
            tuple[Float[Array, ''], dict]: the scalar weighted total loss
                and a dict mapping each term name to its un-weighted scalar 
                loss.
        """
        residuals = {}
        for term in self.res_terms:
            residuals[term.name] = term(model, data)
        total = self.weights.combine(residuals)
        return total, residuals