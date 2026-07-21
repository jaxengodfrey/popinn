Across nearly all scientific domains, differential equations (DEs) are essential for modeling complex processes, systems, and dynamics. Only a small fraction of DEs admit closed-form solutions; for the rest, numerical methods like finite differences, finite elements, and time-stepping integrators have long been the workhorse. These methods approximate the solution by discretizing the domain and propagating local update rules, and they are accurate and well understood. But that accuracy comes at a structural cost: fine meshes and stability constraints can make individual solves expensive or unstable, and every change to the parameters, initial conditions, or boundary conditions requires re-running the solver from scratch. In many-query settings like parameter sweeps, design optimization, inverse problems, and real-time prediction, this per-solve cost can become prohibitive.

An alternative is to construct a surrogate model: a function, typically a neural network, trained to approximate the solution directly. Once trained, a surrogate is cheap to evaluate anywhere in the domain, is continuously differentiable, and (for parametrized models) can cover an entire family of related problems in a single object. The question then becomes how to train it — and this is where physics-informed learning departs from ordinary supervised learning.

### Supervised training

Machine learning models like neural networks are typically trained by minimizing a loss function $\mathcal{L}$. In supervised training scenarios, the network is evaluated over a set of $N$ training points $\{\vec{x}_i\}$ to produce a corresponding set of outputs ${\{\vec{u}_i\}}$. Given the set of ground-truth solutions $\{\vec{y}_i\}$, the loss is then computed as some error metric $E$ over the network outputs and the labeled solution data

$$
\mathcal{L}_{\rm data} = E(\{\vec{u}_i\}, \{\vec{y}_i\}).
$$

A common choice of error metric is the mean squared error (MSE)

$$
E_{\rm MSE} = \frac{1}{N}\sum_{i = 1}^N ||\vec{y}_i - \vec{u}_i ||^2,
$$

but others include the mean absolute error (MAE), sum of squared errors (SSE), or sum of absolute errors (SAE).

### Self-Supervised training

In contrast, physics-informed learning minimizes a loss that encodes the governing physics of the system itself, like a differential equation and a set of physical constraints. This can reduce or eliminate the need for externally labeled data, which can be very useful for scenarios that are difficult to compute numerically or for which no analytic solutions exist.

For example, consider the following partial differential equation:

$$
\frac{\partial y(x,t)}{\partial t} = D[y(x,t);\gamma],
$$

with initial condition

$$
y(x, t = 0) = F(x),
$$

and boundary conditions

$$
y(x = 0, t) = G(t),
$$

$$
y(x = 1, t) = H(t),
$$

where $D$ is some operator that can include spatial derivatives of $y$, $\gamma$ is a fixed coefficient, $F$ is some function that depends on $x$, and $G$ and $H$ are some functions that depend on $t$. The physics-informed loss will consist of four different terms:

$$
\mathcal{L} = \mathcal{L}_{\rm PDE} + \mathcal{L}_{\rm IC} + \mathcal{L}_{\rm BC,left} + \mathcal{L}_{\rm BC, right}.
$$

The PDE residual

$$
\frac{\partial y}{\partial t} - D[y;\gamma] = 0,
$$

can be used to compute the PDE loss

$$
\mathcal{L}_{\rm PDE} = E\Big(\Big\{\frac{\partial u}{\partial t}\Big|_{\{x_i, t_i\}} - D[u;\gamma] \Big|_{\{x_i, t_i\}}\Big\}, 0\Big)
$$

given a set of $N$ points $\{x_i, t_i\}$, called **collocation points**, and a chosen metric $E$. The initial and boundary condition losses are computed in the same manner, though with their own set of $x$ and $t$ points.

Notice what the PDE loss requires: evaluating derivatives of the network's output $u$ with respect to its *inputs* $(x,t)$. To train the network with gradient-based optimizers like Adam or L-BFGS, the whole loss must also be differentiated with respect to the network's *parameters*. This is where Popinn's reliance on JAX becomes very useful: its automatic differentiation computes these derivatives exactly and essentially for free, no hand-derived gradients necessary.

The [next page](anatomy.md) walks through how to set-up and train a physics-informed problem with Popinn.
