The previous page introduced the idea behind physics-informed learning: replace labeled data with residuals derived from the governing equations, and train a network to drive those residuals to zero. This page walks through what that looks like in practice with Popinn.

## The workflow at a glance

1. **Specify the problem.** Write down the differential equation, its initial and/or boundary conditions, and identify what you would like to vary: just the coordinates? A set of scalar parameters? Entire input functions?
2. **Choose a model.** The answer to "what varies" determines the architecture — a plain PINN, a parametrized PINN, or a DeepONet.
3. **Define the residuals.** Each equation and condition becomes a per-point residual function that should vanish when the model is correct.
4. **Compose the loss.** Each residual term is passed to `popinn.Loss` as a `popinn.ResidualTerm` object.
5. **Assemble the training data.** Each residual gets a set of points where it is enforced; parametrized problems additionally get a population of parameter values or input functions.
6. **Train.** Typically a first-order method like Adam first, optionally followed by a second-order method like L-BFGS. Collocation points can remain fixed or be resampled during the Adam phase.
7. **Evaluate.** Query the trained surrogate on dense grids — anywhere in the domain, for any parameter values in the range it was trained over.

## 1. Specify the Problem
Write down your differential equation(s) and any physical constraints that your system must satisfy. Identify all quantities that you would like to vary, e.g. values that will be inputs into the network. Popinn divides these quantities into two different categories: **coordinates** and **auxiliary inputs**. Coordinates are the quantities that your solution directly depends on, like time and spatial dimensions. Auxiliary inputs are values or functions that change the solution space, like scalar coefficients, forcing functions, or initial conditions.


## 2. Choose a Model
The variable quantities you identified in the previous section will determine what type of network architecture your model should have.

**Only coordinates vary.** You want the solution $u(\vec{x}, t)$ of a single DE instance with fixed parameters and conditions. Use a **PINN**: a single MLP that maps a set of coordinates to the solution value with no auxiliary inputs.

**Scalar parameters vary.** You want $u(\vec{x}, t; \vec{\mu})$, the solution across a *family* of problems indexed by one or more scalar coefficients $\vec{\mu}$. These can be quantities such as a diffusion constant, a reaction rate, or a forcing amplitude. Use a **Parametrized PINN** (P$^2$INN): the auxiliary input is $\vec{\mu}$ and is fed to the network alongside the coordinates. One training run covers the whole family; afterwards any parameter value in the trained range can be queried without retraining.

**Input functions vary.** The problem is indexed not by scalars but by *functions*, such as an initial condition or a source term. Use a **DeepONet**: each input function is sampled at a fixed set of sensor locations and fed to the network alongside the coordinates. The trained network approximates the *operator* mapping input functions to solutions.

**Note**: a DeepONet can take functions and scalar parameters as separate auxiliary inputs if desired.

!!! Tip "Creating Custom Models"
    The above are just a few examples of network architectures that have been developed for physics-informed learning and are the models currently implemented in Popinn. If you wish to use a different type of network, you can easily create your own by appropriately subclassing `popinn.AbstractModel`. Everything downstream will work unchanged. All models in Popinn are Equinox Modules (`equinox.Module`), so check out the [Equinox documentation](https://docs.kidger.site/equinox/) before attempting to create your own. A particularly important section is the Abstract vs. Final class pattern that Equinox formalizes [here](https://docs.kidger.site/equinox/pattern/), which you'll need to understand before creating your custom model.


## 3. Define the Residuals

Now that you've determined the appropriate model for your problem, you're ready to start writing some code. This section along with sections 4 and 5 can technically be done in any order, but we recommend following the order as it is currently laid out.

#### Evaluating the model and its derivatives

In Popinn, **all models and their derivatives are evaluated at a single set of inputs and they always output a single scalar**. The call signature for all models is `model(*coords, aux)`, where `*coords` are scalar coordinate values and `aux` is a tuple containing all of the auxiliary inputs for the problem. In practice (e.g. for a problem with 2 spatial dimensions `x1` and `x2` and 1 time dimension `t`), calling the model looks like:

```python
u = model(x1, x2, t)             # no auxiliary inputs
u = model(x1, x2, t, aux)        # with auxiliary inputs
```

where `x1`, `x2`, and `t` are scalars and `aux` is a tuple. This call signature is central to the Popinn API.

All models in Popinn have a method `D` to facilitate gradient computations that make it easy to chain multiple derivatives at once:

```python
u_t  = model.D(2)(x1, x2, t, aux)     # du/dt
u_x1x1 = model.D(0, 0)(x1, x2, t, aux)  # d^2u/dx1^2
```

Under the hood, these are equivalent to:

```python
u_t = jax.grad(model, argnums = 2)(x1, x2, t, aux)
u_x1x1 = jax.grad(jax.grad(model, argnums = 0), argnums = 0)(x1, x2, t, aux)
```

#### Creating residual factories

Each per-point residual function must be defined as a factory, i.e. a function that returns another function. These residual factories should have the signature `Callable(model) -> Callable(*coords, aux) -> scalar`. Below is an example of a PDE residual defined in this way:

```python
def pde_residual_fn(model):
    def res(x1, x2, t, aux):
        du_dt = model.D(2)(x1, x2, t, aux)
        du_dx1 = model.D(0)(x1, x2, t, aux)
        du_dx2 = model.D(1)(x1, x2, t, aux)

        p1, p2 = aux # unpack the individual parameters from aux, as they are needed in the residual equation
        return du_dt - p1 * du_dx1 - p2 * du_dx2
    return res
```

#### Define a `ResidualTerm`

In Popinn, each residual equation is packaged as a `ResidualTerm` object. This object is responsible for evaluating a per-point residual function over its grid of collocation points and auxiliary inputs and applying the error metric to return a single scalar value.

A `ResidualTerm` bundles four aspects of each residual equation:

1. **The name**: A string that links the term to its set of collocation points. A term named `"pde"` will grab its corresponding collocation points from a data container `data` as `data.pde_coords`.
2. **The residual factory**: A factory, like the one defined above, that takes the model and returns a per-point residual function with the same call signature as the model.
3. **The metric**: How a grid of residual values reduces to a single scalar. The default is a mean-squared error (MSE), but it can be any custom callable that reduces a grid of values to a scalar.
4. **The grid evaluation function**: A function like `popinn.eval_grid` or `popinn.eval_grid_flat_aux`. The default is `popinn.eval_grid`, which evaluates the residual equation over a cartesian product of the corresponding collocation points and auxiliary inputs.

For the example above, this looks like:

```python
from popinn import ResidualTerm, eval_grid

pde = ResidualTerm(name = 'pde', residual_fn = pde_residual_fn, metric = 'mse', eval_fn = eval_grid)
```

## 4. Compose the Loss

Once you've defined all of your residual terms, collect them into a list and pass it to `popinn.Loss`. This loss object will build a function that adds the output of each residual term together to return the total loss.

You can optionally specify additional hyperparameters called weights. These multiply each term of the loss and can be either fixed values, using `popinn.FixedWeights`, or can be learnable quantities. Popinn currently does not implement trainable weights, but you can subclass `popinn.AbstractWeights` to construct your own. By default, `Loss` assumes each term has a weight of 1.

```python
from popinn import Loss

# default weights = 1
loss = Loss([pde, ic, bc1, bc2]) # ic, bc1, bc2 should all be ResidualTerm objects, defined in the same manner as the pde example above

# custom fixed weights
from popinn import FixedWeights
weights = FixedWeights({'pde': 100., 'ic': 10., 'bc1': 1., 'bc2': 1.})
loss = Loss([pde, ic, bc1, bc2], weights = weights)
```

## 5. Assemble the Training Data

The training data should be placed in an `equinox.Module` data container. Each residual term in `Loss` will need its own set of coordinate values, i.e. collocation points, stored in a field called `<name>_coords`, where `<name>` is the name of the corresponding residual term. Additionally, the auxiliary inputs should be included as a single tuple in a field called `aux`. Below is an example:

```python
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

class TrainingData(eqx.Module):
    pde_coords: tuple[Array, ...]
    ic_coords: tuple[Array, ...]
    bc1_coords: tuple[Array, ...]
    bc2_coords: tuple[Array, ...]
    aux: tuple

data = TrainingData(
    pde_coords = (jnp.linspace(0,1,30), jnp.linspace(0,1,30), jnp.linspace(0,1,30)), # (x1, x2, t)
    ic_coords = (jnp.linspace(0,1,50), jnp.linspace(0,1,50), jnp.array([0.])), # t = 0 for an initial condition, but it should still be passed as an array
    bc1_coords = ... # similar code for bc1 and bc2
    ...
    aux = (jnp.linspace(1,4,10), jnp.linspace(10,20,10)), # (p1, p2)
)
```

!!! Tip "Grid Evaluation"

    You never store the full grid of collocation points, only the 1-D axes
    that generate them. The evaluation function specified for each `ResidualTerm` takes
    the cartesian product of the term's coordinate arrays and the auxiliary
    inputs, evaluating the per-point residual at every combination. For the
    example above, the `pde` term stores just 90 values (three arrays of 30)
    but is enforced on a grid of $30 \times 30 \times 30 = 27{,}000$
    collocation points. Since the two auxiliary axes are also crossed
    in, the residual is enforced at all $10 \times 10$ parameter
    combinations on top of that: 2.7 million evaluations from ~110 stored
    numbers.

    This is also why the initial condition pins time with the length-1
    array `jnp.array([0.])` rather than a scalar: it is a coordinate axis
    like any other, just with a single value, so the outer product places
    the $50 \times 50$ spatial grid on the $t = 0$ plane.

    If the size of the materialized array becomes too memory intensive, it is possible
    to batch the last auxiliary input axis so that only a small portion is materialized at once.
    In this example, a batch size of 1 would reduce the materialized size by an order of magnitude,
    but at the expense of extra compute time.


## 6. Train

Physics-informed losses are notoriously stiff, and the standard recipe is **two phases**: Adam to find the basin, then optionally L-BFGS to descend into it. Other optimizers are not currently implemented in Popinn.

```python
from popinn import AdamConfig, LBFGSConfig, train_model, warmup_cosine

model, history = train_model(
    model,
    data,
    loss,
    [AdamConfig(lr=warmup_cosine(1e-3, 5000), num_epochs=5000),
     LBFGSConfig(num_epochs=2000)],
)
```

Learning-rate schedules are delegated entirely to optax — the warmup-then-cosine-decay schedule is common enough in PINN practice that Popinn ships a helper for it, but any optax schedule can be passed as the `lr` argument of `AdamConfig`.

The returned history maps each loss term's name to its per-step values, so you can watch the terms individually. Whether the individual loss terms in `history` include their weighting factor or give just their raw values is toggled by the `include_weights` keyword argument in `popinn.Loss`.

#### Re-sampling collocation points
In some cases, it may be desirable to resample the collocation points for a particular residual term during the Adam training phase. The only sampler currently implemented by Popinn draws samples from a uniform grid, `popinn.UniformCollocationSampler`. To resample during training, simply initialize this sampler (or your custom sampler that subclasses `popinn.AbstractSampler`) by specifying your previously constructed data container, the name of the term coordinates you wish to resample, the sample sizes, and bounds for each coordinate axis. Then pass this sampler to `popinn.train_model`:

```python
from popinn import UniformCollocationSampler

sampler = UniformCollocationSampler(
    data = data,
    field = 'pde_coords',
    bounds = ((0,1), (0,1), (0,1)),
    num_samples = 30,
    )

model, history = train_model(
    model,
    data,
    loss,
    [AdamConfig(lr=warmup_cosine(1e-3, 5000), num_epochs=5000),
     LBFGSConfig(num_epochs=2000)],
     sample_fn = sampler,
)
```

The L-BFGS phase needs parameter values to remain fixed, so if a sampler is included it will draw one batch of values and hold it fixed during training.

## 7. Evaluate the Surrogate

After training, the surrogate is cheap to query anywhere. The same `eval_grid` function used inside the loss evaluates the trained model on dense grids, and for parametrized models, over whole parameter sweeps at once:

```python
u = eval_grid(model, (x1_dense, x2_dense, t_dense), aux=(p1_sweep, p2_sweep))
# shape: (len(p2_sweep), len(p1_sweep), len(t_dense), len(x_dense))
```
