# Grid evaluation

Utilities for evaluating a per-point function over a cartesian product of grid points.

#### Conventions

- The function being evaluated must have the signature:  `fn(x1, x2, ..., aux) -> scalar`,
    where coordinates `x1, x2, ...` are scalar entries and `aux` is
    a **tuple** of auxiliary components (PDE parameters, discretized source
    functions, initial conditions, etc). Components of `aux` may be scalars or arrays
    of any (possibly different) shapes.

- The function is evaluated over a cartesian product of the 1D coordinate arrays
    and the leading axis of each `aux` element. The trailing axes of each `aux` element
    are treated as the per-call input for that component:

    ```python
    # scalar input
    a = jnp.linspace(0,1,10)

    # function input
    x = jnp.linspace(0,10,20) # sensor coordinate
    b = jnp.linspace(0,100,15) # function parameter
    func = lambda x, b: b * x**2
    f = jax.vmap(func, in_axes = (None, 0))(x, b) # shape = (15, 20)

    #the aux tuple
    aux = (a, f) # per-call shape: a = (1,), f = (20,)
    ```

 - An element of `aux` can be a tuple of arrays that share their leading axis length.
    This is useful when the function requires a parameter that a functional auxiliary
    input was calculated from. In the above example, this would be the `b` array of
    parameter values that were used to calculate `f`:

    ```python
    aux = (a, (f, b)) # per-call shape: a = (1,), f = (20,), b = (1,)

    residual_fn(model):
        r(x, aux):
            a, (f, b) = aux
            # du/dx = b*u
            res = model.D(0)(x, (a, f)) - b * model(x, (a, f)) # b isn't a model input, but it is required to calculate the PDE
            return res
        return r
    ```

    Tuple elements of `aux` are zipped across their shared axis to preserve their
    correlated structure.

- Each function returns the same output layout `(*reversed(aux axis lengths), *reversed(coord lengths))`. For
the example above, `aux = (a, (f, b))` and a single collocation coordinate `x = jnp.linspace(0,10,100)`, the output has a shape `(15, 10, 100)`.

***
#### Memory dials

By default, the cartesian product evaluation is performed by nested `jax.vmap` calls, which
materializes the full grid of evaluation points in memory at once.
Each utility has the ability to trade `jax.vmap` for `jax.lax.map` on a single axis in order
to execute the computation in batches to be more memory efficient:

  * `outer_batch_size` in `eval_grid`: batches the outermost grid axis only --
    axis 0 of `aux[-1]` when `aux` is non-empty, otherwise the last coordinate
    axis (so models with no auxiliary inputs can batch over a coordinate).

  * `eval_grid_flat_aux`: enumerate the `aux` combinations and stack them into an array with
    shape `(N_combinations, N_aux_elements)`, then batch over the first axis of this array.

***

::: popinn.eval_grid

::: popinn.eval_grid_flat_aux
