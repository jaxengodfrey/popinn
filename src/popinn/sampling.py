"""sampling.py

Resamplers for use with train_adam / train_model via their `sample_fn`
argument.

A resampler implements the contract  key -> batch : given a fresh PRNG key it
returns a new training-data container (the same eqx.Module type passed to
training) with some fields redrawn. It is called in plain Python between
optimizer steps, so it is not traced by jit/grad/vmap.

`AbstractSampler` enforces the contract so users can define their own samplers
(e.g. Latin-hypercube, adaptive residual-based, or time-windowed schemes) by
subclassing and implementing `__call__`. The trainer accepts ANY callable with
the key -> batch signature, not only AbstractSampler subclasses -- a plain
closure works just as well -- so the base class documents and organizes the
contract without gatekeeping it. `UniformCollocationSampler` is the built-in
concrete implementation.
"""

import abc
from collections.abc import Sequence

import equinox as eqx
import jax.random as jr
from jaxtyping import PRNGKeyArray


class AbstractSampler(eqx.Module):
    """Interface for batch resamplers: key -> batch.

    A sampler returns a new training-data container (the same eqx.Module type
    passed to training) with some fields redrawn from the given PRNG key. It
    is invoked between optimizer steps in plain Python, so it is not traced;
    subclasses may still hold array fields (such as a reference batch to copy
    unchanged fields from) because eqx.Module gives that state a structured,
    optionally serialisable home.

    Subclasses must implement `__call__`. To keep the jitted training step
    from recompiling, a sampler MUST return batches of constant shape across
    calls (resample values, not sizes).

    Note: the trainer calls samplers structurally (`sample_fn(key)`) and does
    not require AbstractSampler specifically -- any callable with the same
    signature is accepted. Subclass this when you want the contract named and
    your sampler's configuration introspectable; use a plain closure when you
    just need a quick one-off resampler.
    """

    @abc.abstractmethod
    def __call__(self, key: PRNGKeyArray) -> eqx.Module:
        """Draw a new batch from `key`.

        Args:
            key (PRNGKeyArray): PRNG key supplied by the trainer (freshly
                split each resample).

        Returns:
            eqx.Module: a new training-data container with the resampled
                fields replaced and all others carried through unchanged.
        """
        raise NotImplementedError


class UniformCollocationSampler(AbstractSampler):
    """Resample one coordinate field with uniform random collocation points.

    The named `field` is assumed to be a tuple of 1-D coordinate arrays, one
    per coordinate axis (the convention eval_grid expects, e.g. (x, t)). Each
    axis is independently redrawn uniformly within its bounds; every other
    field of the reference batch -- in particular `aux`, which defines the
    population of PDE instances -- is carried through unchanged. Each call
    constructs a new batch (it never mutates the reference), so it is safe
    with the immutability of eqx.Module fields.

    Fields:
        data (eqx.Module): the reference training-data container. Its
            non-resampled fields are reused as-is in every returned batch.
            Held as an array-bearing (non-static) field, so this sampler is
            itself a pytree containing the reference batch.
        bounds (tuple): one (min, max) pair per coordinate axis of `field`.
        counts (tuple): points per coordinate axis (static).
        field (str): name of the coordinate field to resample in `data` (static).
    """

    data: eqx.Module
    bounds: tuple = eqx.field(static=True)
    counts: tuple = eqx.field(static=True)
    field: str = eqx.field(static=True)

    def __init__(
        self,
        data: eqx.Module,
        bounds: Sequence[tuple[float, float]],
        num_samples: int | Sequence[int] | None = None,
        field: str = "pde_coords",
    ):
        """Configure the sampler.

        Args:
            data (eqx.Module): the training-data container to resample from.
                Captured here, so it only needs to exist when the sampler is
                constructed -- not at module-import time.
            bounds (Sequence[tuple[float, float]]): one (min, max) pair per
                coordinate axis of `data.<field>`. Its length must equal the
                number of axes in that field.
            num_samples (int | Sequence[int] | None): number of collocation
                points per axis. An int applies to every axis; a sequence
                gives a per-axis count. None (default) reuses the current
                per-axis sizes of `data.<field>`. NOTE: if the resolved counts
                differ from the sizes in the batch first passed to training,
                the jitted step recompiles on the first resample, so keep them
                equal unless a recompile is intended.
            field (str): name of the coordinate field to resample. Defaults to
                'pde_coords'. Intended for interior collocation fields;
                pointing it at a boundary field would also resample that
                field's pinned axis, which is usually not wanted.

        Raises:
            ValueError: if `bounds` (or a sequence `num_samples`) does not
                have one entry per coordinate axis of `data.<field>`.
        """
        coords = getattr(data, field)
        n_axes = len(coords)

        if len(bounds) != n_axes:
            raise ValueError(f"bounds has {len(bounds)} entries but data.{field} has {n_axes} coordinate axes; expected one `(min, max)` per axis.")

        if num_samples is None:
            counts = tuple(c.shape[0] for c in coords)
        elif isinstance(num_samples, int):
            counts = (num_samples,) * n_axes
        else:
            if len(num_samples) != n_axes:
                raise ValueError(f"num_samples has {len(num_samples)} entries but data.{field} has {n_axes} coordinate axes.")
            counts = tuple(num_samples)

        self.data = data
        self.bounds = tuple(bounds)
        self.counts = counts
        self.field = field

    def __call__(self, key: PRNGKeyArray) -> eqx.Module:
        """Draw a new batch with `field` resampled uniformly.

        Args:
            key (PRNGKeyArray): PRNG key; split into one subkey per axis so
                the axes are sampled independently.

        Returns:
            eqx.Module: a new batch with `field` redrawn within `bounds` and
                all other fields (including `aux`) unchanged.
        """
        keys = jr.split(key, len(self.counts))
        new_coords = tuple(
            jr.uniform(keys[i], (self.counts[i],), minval=self.bounds[i][0], maxval=self.bounds[i][1]) for i in range(len(self.counts))
        )
        # Replace only `field`; tree_at returns a new module, leaving the
        # reference `data` (and its aux / other fields) untouched.
        return eqx.tree_at(lambda b: getattr(b, self.field), self.data, new_coords)
