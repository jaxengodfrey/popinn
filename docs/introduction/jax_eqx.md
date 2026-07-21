# JAX and Equinox Essentials

Popinn is built on [JAX](https://docs.jax.dev/) and [Equinox](https://docs.kidger.site/equinox/). This page is not a tutorial for either (their own documentation covers them very thoroughly) but a short summary of the essential ideas that Popinn utilizes.

## JAX in brief

JAX is a NumPy-style array library that includes a suite of composable function transformations and supports multiple backends like CPU and GPU. Define a function and easily compute its derivatives with `jax.grad`, vectorize it over a batch axis with `jax.vmap`, and [compile](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html#just-in-time-compilation-with-jax-jit) it with `jax.jit`. The price is that transformed functions must be *pure*, meaning they have no side effects and do not mutate arrays.

Key concepts:

- **`jit` expects static shapes.** A jitted function is traced and compiled for the specific input shapes it first sees; new shapes trigger recompilation. Popinn's training step is jitted, so resamplers must return batches of constant shape.
- **JAX arrays are immutable.** NumPy-style in-place updates like `x[0] = 1.0` will raise an error. The functional equivalent is `x = x.at[0].set(1.0)`, which returns a new array.
- **JAX functions can operate on pytrees, not just arrays.** A [PyTree](https://docs.jax.dev/en/latest/pytrees.html#working-with-pytrees) is any nested structure of containers like tuples, lists, and dicts. The "leaves" of the pytree can be anything, like arrays, functions, or floats, but transformations like `jax.jit` and `jax.grad` only operate on functions that input and output pytrees with array leaves.

### GPU support

JAX runs the same code on CPU, GPU, and TPU. No changes to your script are required, just make sure you have an accelerator-enabled JAX installation (see the [installation guide](https://docs.jax.dev/en/latest/installation.html)).

For scaling beyond a single device, JAX can also shard arrays across multiple devices and parallelize computation over them automatically. See [distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/parallel.html) in the JAX docs and the [autoparallelism example](https://docs.kidger.site/equinox/examples/parallelism/) in the Equinox docs.

### Precision

JAX defaults to float32, and PDE residuals, especially with higher-order derivatives, often need float64 to converge well. Enable it at the top of your script, before any arrays are created:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

**Note:** Float64 can be significantly slower than float32 on most GPUs.

## Equinox

### Classes as pytrees

Equinox is a very handy library that builds upon the JAX core. Most importantly, Equinox makes it easy to construct *classes* as pytrees with `equinox.Module`. This means we can apply transformations like `grad` with respect to a model and thus use gradient descent methods to tune the model's parameters. Everything stateful in Popinn is an `equinox.Module`: models, `Loss`, data containers, samplers, etc.

If you want to use Popinn as is, at minimum you'll need to know how to build a data container with `equinox.Module`. You can see how this is done in the supplied examples, so head over to the [PINN](../examples/pinn.ipynb) example. The rest of the page points out some important concepts for creating custom components in Popinn.

Two things to know about Modules:

- **Their fields are immutable.** Reassigning a field raises `FrozenInstanceError`. To change one, build a new Module with the relevant piece swapped — `eqx.tree_at(lambda d: d.pde_coords, data, fresh_coords)` returns a new container sharing every other field.
- **Use the `eqx.filter_*` transformations.** Transformations like `jax.grad` and `jax.jit` expect every leaf they process to be an array, but a realistic model also carries non-arrays: activation functions, layer sizes, names. `eqx.filter_jit` and `eqx.filter_grad` handle this by partitioning a Module into its array leaves (traced, differentiated) and everything else (treated as static), applying the plain JAX transformation, and recombining. Popinn's trainers use these internally, so you only meet the `filter_*` family if you write custom training code.

### Abstract classes as interfaces

Equinox formalizes an [abstract/final pattern](https://docs.kidger.site/equinox/pattern/): a class is either abstract or final. An abstract class declares abstract fields (`equinox.AbstractVar`) and methods (`@abc.abstractmethod`) that subclasses must supply. A final class is a subclass of an abstract class that supplies concrete versions of the abstract fields and methods. It is called "final" because it should not be subclassed further. All fields and the __init__ method live on the final class.

An abstract class may also provide concrete methods shared by every subclass. `popinn.AbstractModel` implements the `__call__` method and the derivative helper `D` this way, so every model inherits them without redefining them. The pattern's rule is that only abstract methods and attributes can be overridden; once something is concrete, no subclass changes it.

In Popinn, the main utility of this pattern is that abstract classes define the minimal structure a final class must have and standardize call conventions. This keeps the package modular: e.g. a model can be swapped for another with minimal changes to code. As long as the new model follows the call convention `model(*coords, aux) -> scalar` declared by `popinn.AbstractModel`, the downstream machinery is unchanged.

!!! Tip "Going Deeper"
    Reading through the [JAX documentation](https://docs.jax.dev/) and the [Equinox documentation](https://docs.kidger.site/equinox/) is not required to *use* Popinn, but it is highly recommended before writing custom components.
