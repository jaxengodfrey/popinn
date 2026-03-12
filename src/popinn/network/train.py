import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from ..physics.loss import LossWeights, total_loss
from .model import PINN
from .sampling import sample_collocation

def train(
    # Physics parameters
    theta: float = 1.0,
    nu: float = 1.0,
    gamma: float = 0.0,
    t_max: float = 0.5,
    # Network architecture
    hidden_dims: list[int] = None,
    # Training parameters
    num_epochs: int = 10_000,
    lr: float = 1e-3,
    lr_schedule: str = "cosine",  # "constant" or "cosine"
    n_interior: int = 1024,
    n_bc: int = 64,
    n_ic: int = 128,
    # Constraint mode
    use_hard: bool = True,
    weights: LossWeights = None,
    # Misc
    seed: int = 42,
    log_every: int = 500,
):
    """Train the PINN and return the trained model + training history.
    
    Args:
        theta: scaled mutation rate 4*N*mu
        gamma_init: scaled selection 2*N*s used for the initial condition
        gamma_evolve: scaled selection 2*N*s used in the PDE evolution
        t_max: final time in units of 2*N generations
    """
    if hidden_dims is None:
        hidden_dims = [64, 64, 64, 64]
    if weights is None:
        weights = LossWeights()

    key = jr.PRNGKey(seed)
    key, model_key = jr.split(key)

    # Initialize model
    model = PINN(model_key, hidden_dims=hidden_dims)

    # Optimizer with optional cosine decay
    if lr_schedule == "cosine":
        schedule = optax.cosine_decay_schedule(lr, num_epochs)
        optimizer = optax.adam(schedule)
    else:
        optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # JIT-compiled training step
    @eqx.filter_jit
    def step(model, opt_state, colloc_xt, x_ic, t_bc):
        (loss_val, loss_dict), grads = eqx.filter_value_and_grad(
            lambda m: total_loss(m, colloc_xt, x_ic, t_bc, gamma, weights, use_hard, theta = theta, nu = nu),
            has_aux=True
        )(model)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss_val, loss_dict

    # Training loop
    history = {"total": [], "pde": [], "ic": [], "bc_left": [], "bc_right": [], "non_neg": []}

    for epoch in range(num_epochs):
        key, sample_key = jr.split(key)
        colloc_xt, x_ic, t_bc = sample_collocation(
            sample_key, n_interior, n_bc, n_ic, t_max
        )

        model, opt_state, loss_val, loss_dict = step(
            model, opt_state, colloc_xt, x_ic, t_bc
        )

        history["total"].append(float(loss_val))
        history["pde"].append(float(loss_dict["pde"]))
        history["ic"].append(float(loss_dict["ic"]))
        history["bc_left"].append(float(loss_dict["bc_left"]))
        history["bc_right"].append(float(loss_dict["bc_right"]))
        history["non_neg"].append(float(loss_dict["non_neg"]))

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>6d} | Total: {loss_val:.2e} | "
                  f"PDE: {loss_dict['pde']:.2e} | "
                  f"BC_L: {loss_dict['bc_left']:.2e} | "
                  f"BC_R: {loss_dict['bc_right']:.2e} | "
                  f"NonNeg: {loss_dict['non_neg']:.2e}")

    return model, history