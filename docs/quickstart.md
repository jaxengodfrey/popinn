# Quickstart

We recommend reading through each of the below examples in order:

- [PINN](examples/pinn.ipynb)
- [Parametrized PINN](examples/parametrized_pinn.ipynb)

<!--
To train a physics-informed network, you need three things:

1. A set of governing equations, e.g. a PDE and its boundary/initial conditions.
2. Training data, including a grid of *collocation points* and, depending on your problem, PDE parameters or functions evaluated at fixed sensor points (we call these auxiliary inputs).
3. An optimizer for training, like Adam or L-BFGS.

Each equation that must be satisfied is composed into a loss term, which is evaluated over your grid of collocation points and auxiliary inputs. For example:

See [quick start](quickstart.md) for examples. -->
