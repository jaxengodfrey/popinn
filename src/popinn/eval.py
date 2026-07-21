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

    Args:
        fn (Callable): function to vmap
        n_coords (int): number of coordinates
        n_aux (int): number of independent auxiliary components
        axis (int): grid axis index to vmap. `axis <= n_coords + n_aux`

    Returns:
        Callable: fn vmapped over a single grid axis.
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
    """Evaluate `fn` on the cartesian product of coordinate and auxiliary axes.

    Args:
        fn (Callable): Per-point function with signature `fn(*coords, aux) -> scalar`, with `aux` a
            tuple of auxiliary inputs.
        coords_1d (tuple[Array, ...]): Tuple of 1-D coordinate arrays.
        aux (tuple): Tuple of auxiliary components. Each TOP-LEVEL
            element is one grid axis (its leading axis); trailing dims are the
            per-call input for that component. Independent top-level elements are
            crossed:

            ```python
                aux = (a, b)        # a.shape=(Na,), b.shape=(Nb,) -> Na x Nb grid
            ```

            A top-level element that is itself a tuple is zipped, not
            crossed: all its leaves share their leading axis. Use this for
            correlated quantities, e.g. a DeepONet whose initial conditions
            were generated from PDE parameters (IC[k] from param[k]):

            ```python
                # IC.shape = (N_IC, N_IC_PTS), param.shape = (N_IC,)
                aux = ((IC, param),)   # ONE axis of length N_IC, zipped
            ```

            so per call the function receives the pair `(IC[k], param[k])`.
        outer_batch_size (int | None): If not `None`, the OUTERMOST
            grid axis is batched with `jax.lax.map`, which dispatches `jax.vmap`
            on each batch. The outermost axis is axis 0 of `aux[-1]` when `aux` is non-empty
            and the last coordinate axis (axis 0 of `coords_1d[-1]`) when
            `aux` is empty. Note that batching an axis shrinks the inner
            vectorized batch to the product of the remaining
            axes, so keep the largest coordinate axis as the first element and the largest
             `aux` axis as the last element.

    Returns:
        (Array): Array of shape `(*reversed(aux lens), *reversed(coord lens))`.
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
    batch_size: int,
    aux: tuple = (),
) -> Array:
    """Like `eval_grid`, but enumerates the `aux` combinations by integer index and stacks them into an array with
    shape `(N_combinations, N_aux_elements)`. The computation is then performed on `batch_size` chunks of the combinations.

    `batch_size` is required, so if you don't intend to utilize the batching scheme, use `eval_grid` instead.

    Use this when even one full outermost-axis batch in `eval_grid` is too big, or when
    the aux axes are very uneven (e.g. `Na=2, Nb=10_000`).

    Args:
        fn (Callable): Per-point function with signature `fn(*coords, aux) -> scalar`, where `aux` is a
            tuple of auxiliary inputs.
        coords_1d (tuple[Array, ...]): Tuple of 1-D coordinate arrays.
        batch_size (int): The batch size for `jax.lax.map`.
        aux (tuple): Tuple of auxiliary components. Each TOP-LEVEL
            element is one grid axis (its leading axis); trailing dims are the
            per-call input for that component. Independent top-level elements are
            crossed:

            ```python
                aux = (a, b)        # a.shape=(Na,), b.shape=(Nb,) -> Na x Nb grid
            ```

            A top-level element that is itself a tuple is zipped, not
            crossed: all its leaves share their leading axis. Use this for
            correlated quantities, e.g. a DeepONet whose initial conditions
            were generated from PDE parameters (IC[k] from param[k]):

            ```python
                # IC.shape = (N_IC, N_IC_PTS), param.shape = (N_IC,)
                aux = ((IC, param),)   # ONE axis of length N_IC, zipped
            ```

            so per call the function receives the pair `(IC[k], param[k])`.

    Returns:
        (Array): Array of shape `(*reversed(aux lens), *reversed(coord lens))`.
    """
    n_c, n_a = len(coords_1d), len(aux)

    # Zero aux axes -> exactly one (empty) combination: chunking over
    # combinations is a no-op, so this is just the coord-grid evaluation.
    if n_a == 0:
        return eval_grid(fn, coords_1d, aux)

    # Axis length of a component = leading-axis length of its first leaf
    # (for grouped components, all leaves share it by convention).
    aux_lens = tuple(jax.tree_util.tree_leaves(comp)[0].shape[0] for comp in aux)

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
        slices = tuple(jax.tree_util.tree_map(lambda leaf, k=k: leaf[ix[k]], aux[k]) for k in range(n_a))
        return g(*coords_1d, slices)

    out = jax.lax.map(body, idx, batch_size=batch_size)
    # out: (N_combos, *reversed(coord lens)), combos first-component-major.

    # Match eval_grid's layout: unflatten, then reverse the aux axes so the
    # LAST component leads, as in the nested-vmap convention.
    out = out.reshape(*aux_lens, *out.shape[1:])
    perm = tuple(reversed(range(n_a))) + tuple(range(n_a, out.ndim))
    return jnp.transpose(out, perm)
