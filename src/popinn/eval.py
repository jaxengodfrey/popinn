"""eval.py

Tensor-product evaluation utilities for the per-point models in `models.py`
(P2INN, DeepONet, and anything else with the signature fn(*coords, aux)).

Conventions
-----------
1. Per-point model signature:  fn(x, t, ..., aux) -> scalar, where `aux` is
   a TUPLE of auxiliary components (PDE parameters, discretized source
   functions, initial conditions, ...). Components may be scalars or arrays
   of any (possibly different) shapes -- JAX treats the tuple as a pytree.

2. Batching: each TOP-LEVEL element of `aux` is one grid axis (its leading
   axis); everything trailing is the per-call input for that component:

       P2INN:    aux = (a, b),       a.shape = (Na,),    b.shape = (Nb,)     -> per call: scalars
       DeepONet: aux = (f1, f2),  f1 = (Nf1, Nf1_sen), f2 = (Nf2, Nf2_sen)   -> per call: vectors

3. Because `aux` is a pytree, vmap's `in_axes` can address individual leaves
   (e.g. in_axes=(None, None, (0, None)) maps leaf 0 of the aux tuple).
   Each grid axis is one vmap that maps exactly one coordinate argument
   or one top-level aux element. This unifies product vs. paired batching
   through aux STRUCTURE: two separate top-level elements are crossed (their
   tensor product); a single top-level element that is itself a tuple is
   zipped (all its leaves share one axis). See `eval_grid` for examples.


Memory dials
------------
Fully vmapping everything materializes activations for the full product of
all coordinate and aux axis lengths. Two ways to trade compute for memory,
both jit-compatible and differentiable (the loops are lax.map):

  * `outer_batch_size` in eval_grid: chunk the OUTERMOST grid axis only --
    axis 0 of aux[-1] when aux is non-empty, otherwise the last coordinate
    axis (so pure PINNs with aux=() can chunk over a coordinate).

  * eval_grid_flat_aux: enumerate ALL aux combinations by integer index and
    chunk over them jointly. Peak memory scales with batch_size * coord-grid
    size, independent of how the combination count factors across axes. Uses
    jax.lax.map.

All paths return the SAME output layout:

    (*reversed(aux axis lengths), *reversed(coord lengths))
    e.g. coords = (x, t), aux = (a, b)  ->  (Nb, Na, Nt, Nx)
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array


# ──────────────────────────────────────────────────────────────
# Internal: one-vmap-per-axis construction
# ──────────────────────────────────────────────────────────────

def _vmap_axis(fn: Callable, n_coords: int, n_aux: int, axis: int) -> Callable:
    """Wrap fn in one vmap that maps exactly one grid axis.

    axis < n_coords        -> map coordinate argument `axis` at its axis 0
    axis >= n_coords       -> map aux tuple leaf `axis - n_coords` at axis 0

    All other coordinate arguments and aux leaves are broadcast (None).
    Uses pytree in_axes for the aux argument, so fn's signature
    fn(*coords, aux_tuple) is preserved at every level.
    """
    if axis < n_coords:
        # map a single coordinate axis
        in_axes = tuple(0 if j == axis else None for j in range(n_coords))
        in_axes += (None,)  # whole aux tuple broadcast
    else:
        # map a single auxiliary axis
        leaf = axis - n_coords
        aux_axes = tuple(0 if m == leaf else None for m in range(n_aux))
        in_axes = (None,) * n_coords + (aux_axes,)
    return jax.vmap(fn, in_axes=in_axes)


# ──────────────────────────────────────────────────────────────
# Nested-vmap tensor product
# ──────────────────────────────────────────────────────────────

def eval_grid(
    fn: Callable,
    coords_1d: tuple[Array, ...],
    aux: tuple = (),
    outer_batch_size: int | None = None,
) -> Array:
    """Evaluate fn on the tensor product of coordinate and auxiliary axes.

    Args:
        fn: per-point function fn(*coords, aux) -> scalar, with `aux` a
            tuple of components. Works for `model` and for coordinate
            derivatives `model.D(...)`, since D preserves the signature.
        coords_1d: tuple of 1-D coordinate arrays, e.g. (x, t).
        aux: tuple of auxiliary components. Each TOP-LEVEL element is one
            grid axis (its leading axis); trailing dims are the per-call
            input for that component. Independent top-level elements are
            crossed (their tensor product):

                aux = (a, b)        # a.shape=(Na,), b.shape=(Nb,) -> Na x Nb grid

            A top-level element that is itself a tuple is ZIPPED, not
            crossed -- all its leaves share one axis. Use this for
            correlated quantities, e.g. a DeepONet whose initial conditions
            were generated from PDE parameters (IC[k] from param[k]):

                # IC.shape = (N_IC, N_IC_PTS), param.shape = (N_IC,)
                aux = ((IC, param),)   # ONE axis of length N_IC, zipped

            so per call the function receives the pair (IC[k], param[k]).
            Defaults to () for models with no auxiliary inputs (pure PINNs).
        outer_batch_size: if given, the OUTERMOST grid axis is chunked
            with lax.map instead of vmapped, so only `outer_batch_size`
            slices of that axis have live activations at once. The
            outermost axis is axis 0 of aux[-1] when aux is non-empty,
            and the last coordinate axis (axis 0 of coords_1d[-1]) when
            aux is empty -- so pure PINNs (aux=()) can chunk over a
            coordinate. Note that chunking a coordinate shrinks the inner
            vectorized batch to the product of the REMAINING coordinate
            axes, so keep the largest coordinate grid innermost (first)
            to preserve GPU utilization. None -> fully vmapped (fastest,
            most memory).

    Returns:
        Array of shape (*reversed(aux lens), *reversed(coord lens)).
        Convention: the LAST axis vmapped becomes the LEADING output axis;
        axes are wrapped in argument order, so the outermost axis is
        aux[-1] if present, else coords_1d[-1]. The chunked path matches
        this layout exactly (lax.map's leading output axis lands where
        the outermost vmap would have put it).
    """
    n_c, n_a = len(coords_1d), len(aux)
    n_total = n_c + n_a

    g = fn
    n_vmapped = n_total if outer_batch_size is None else n_total - 1
    for i in range(n_vmapped):
        g = _vmap_axis(g, n_c, n_a, i)

    if outer_batch_size is None:
        return g(*coords_1d, aux)

    # Chunked outermost axis: sequentially map over its slices,
    # `outer_batch_size` at a time. lax.map(batch_size=k) vmaps within
    # each chunk and scans across chunks, so only one chunk's activations
    # exist at any moment.
    if n_a > 0:
        # Outermost axis is the last aux component.
        body = lambda last: g(*coords_1d, (*aux[:-1], last))
        return jax.lax.map(body, aux[-1], batch_size=outer_batch_size)

    # No aux components: the outermost grid axis is the last coordinate.
    body = lambda last_c: g(*coords_1d[:-1], last_c, aux)
    return jax.lax.map(body, coords_1d[-1], batch_size=outer_batch_size)


# ──────────────────────────────────────────────────────────────
# Joint chunking over ALL parameter combinations
# ──────────────────────────────────────────────────────────────

def eval_grid_flat_aux(
    fn: Callable,
    coords_1d: tuple[Array, ...],
    aux: tuple = (),
    batch_size: int | None = None,
) -> Array:
    """Like eval_grid, but FLATTENS the outer axis of each `aux` component
        and, if specified, evaluates `batch_size` chunks.

    Use this when even one full outermost-axis chunk is too big, or when
    the aux axes are very uneven (e.g. Na=2, Nb=10_000). Peak activation
    memory ~ batch_size * (coord-grid batch).

    Combinations are enumerated as INTEGER INDEX tuples; the aux data is
    never meshgridded, so components of any shape work unchanged, and the
    only materialized overhead is an (N_combos, n_aux) int array.

    Args:
        fn: per-point function fn(*coords, aux) -> scalar, with `aux` a
            tuple of components. Works for any function that preserves this
            signature, including a subclass of popinn.AbstractModel.
        coords_1d: tuple of 1-D coordinate arrays, e.g. (x, t).
        aux: tuple of auxiliary components. Each TOP-LEVEL element is one
            grid axis (its leading axis); trailing dims are the per-call
            input for that component. Independent top-level elements are
            crossed (their tensor product):

                aux = (a, b)        # a.shape=(Na,), b.shape=(Nb,) -> Na x Nb grid

            A top-level element that is itself a tuple is ZIPPED, not
            crossed -- all its leaves share one axis. Use this for
            correlated quantities, e.g. a DeepONet whose initial conditions
            were generated from PDE parameters (IC[k] from param[k]):

                # IC.shape = (N_IC, N_IC_PTS), param.shape = (N_IC,)
                aux = ((IC, param),)   # ONE axis of length N_IC, zipped

            so per call the function receives the pair (IC[k], param[k]).
            Defaults to () for models with no auxiliary inputs (pure PINNs).
        batch_size: If given, the flattened combination axis is chunked so
            only `batch_size` combinations have live activations at once.

    Returns:
        Array of shape (*reversed(aux lens), *reversed(coord lens)) --
        the SAME layout as eval_grid. The flattened combinations are
        enumerated first-component-major, evaluated via lax.map, then
        reshaped to (*aux lens, *coord-grid) and the aux axes reversed so
        the last component leads, matching eval_grid's convention.
    """
    n_c, n_a = len(coords_1d), len(aux)

    # Zero aux axes -> exactly one (empty) combination: chunking over
    # combinations is a no-op, so this is just the coord-grid evaluation.
    if n_a == 0:
        return eval_grid(fn, coords_1d, aux)

    # Axis length of a component = leading-axis length of its first leaf
    # (for grouped components, all leaves share it by convention).
    aux_lens = tuple(
        jax.tree_util.tree_leaves(comp)[0].shape[0] for comp in aux
    )

    # Vectorize the coordinate axes only; the aux tuple stays per-call.
    g = fn
    for i in range(n_c):
        g = _vmap_axis(g, n_c, n_a, i)
    # g now maps (coord_vecs..., aux_slices_tuple) -> coord grid

    # Enumerate combinations in 'ij' (first-component-major) index order.
    idx_mesh = jnp.meshgrid(*(jnp.arange(n) for n in aux_lens), indexing="ij")
    idx = jnp.stack([m.ravel() for m in idx_mesh], axis=-1)  # (N_combos, n_a)

    # Gather one slice per component inside the loop body (dynamic gather
    # is fine under lax.map). tree_map handles plain arrays and grouped
    # (sub-tuple) components uniformly: every leaf is indexed at axis 0
    # by the same index, preserving zip pairing within a group.
    def body(ix):
        slices = tuple(
            jax.tree_util.tree_map(lambda leaf, k=k: leaf[ix[k]], aux[k])
            for k in range(n_a)
        )
        return g(*coords_1d, slices)

    out = jax.lax.map(body, idx, batch_size=batch_size)
    # out: (N_combos, *reversed(coord lens)), combos first-component-major.

    # Match eval_grid's layout: unflatten, then reverse the aux axes so the
    # LAST component leads, as in the nested-vmap convention.
    out = out.reshape(*aux_lens, *out.shape[1:])
    perm = tuple(reversed(range(n_a))) + tuple(range(n_a, out.ndim))
    return jnp.transpose(out, perm)