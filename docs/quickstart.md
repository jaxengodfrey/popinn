# Introduction

## Prerequisite: Physics-Informed Learning

Machine learning models are typically trained by minimizing a loss function. In supervised learning, the loss measures the error between the network's output and a target output, e.g. a ground-truth solution obtained from experiments or numerical solvers. In contrast, physics-informed learning minimizes a loss that encodes the governing physics of the system itself. Most commonly, the loss takes the form of the residual of a governing partial differential equation (PDE), along with its boundary and initial conditions, evaluated at a set of collocation and/or sensor points. Because the supervisory signal comes from the physics rather than from external labels, this is often described as a form of self-supervised learning: the network is trained to satisfy the governing equations directly, reducing or eliminating the need for ground-truth solution data.

<!-- This package implements common netw to physics-informed learning. Physics-Informed Neural Networks (PINNs) approximate the solution of a single PDE instance by parameterizing the solution field with a neural network and minimizing the PDE residual at collocation points. Parameterized Physics-Informed Neural Networks (P$^2$INNs) extend this idea to families of PDEs, learning a solution that generalizes across equation parameters (e.g. coefficients, source terms) so that new instances do not require retraining from scratch. Instead of approximating a single solution, Deep Operator Networks (DeepONets) learn a mapping from input functions (such as initial conditions or forcing terms) to output solution functions.  -->

## Introductory Examples

We recommend reading through each of the below examples in order:

1. [PINN](examples/pinn.ipynb)
2. [Parametrized PINN](examples/parametrized_pinn.ipynb)
3. [PI-DeepONet](examples/deeponet.ipynb)
<!--
To train a physics-informed network, you need three things:

1. A set of governing equations, e.g. a PDE and its boundary/initial conditions.
2. Training data, including a grid of *collocation points* and, depending on your problem, PDE parameters or functions evaluated at fixed sensor points (we call these auxiliary inputs).
3. An optimizer for training, like Adam or L-BFGS.

Each equation that must be satisfied is composed into a loss term, which is evaluated over your grid of collocation points and auxiliary inputs. For example:

See [quick start](quickstart.md) for examples. -->
