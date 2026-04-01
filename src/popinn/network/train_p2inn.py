import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from ..physics.loss import LossWeights, total_loss
from .models import P2INN
from .sampling import sample_collocation_and_param
import jax
from ..physics.solution import get_final_sols
import jax.numpy as jnp
import numpy as np

def train_p2inn_adam_lbfgs(
    model = None,
    # Physics parameters
    theta: float = 1.0,
    gamma_init: float = 0.0,
    gamma_range: tuple = (0.,10.),
    gamma_sign: float = -1.,
    n_gamma: int = 50,
    nu: float = 1.0,
    t_max: float = 0.5,
    # Network architecture
    # Training parameters
    num_epochs_adam: int = 10_000,
    num_epochs_lbfgs: int = 0,
    lr: float = 1e-3,
    lr_schedule: str = "cosine",  # "constant" or "cosine"
    n_interior: int = 100,
    # Constraint mode
    use_hard: bool = True,
    weights: LossWeights = None,
    # Misc
    seed: int = 42,
    log_every: int = 500,
):
    """Train the P2INN and return the trained model + training history.
    
    Supports two-phase training:
        1. Adam phase (num_epochs_adam steps): stochastic, good for exploring
           the loss landscape and getting into the right basin.
        2. L-BFGS phase (num_epochs_lbfgs steps): deterministic, uses fixed
           collocation points for fast convergence to a precise minimum.
    
    Set num_epochs_adam=0 to skip Adam, or num_epochs_lbfgs=0 to skip L-BFGS.
    
    Args:
        theta: scaled mutation rate 4*N*mu
        gamma_init: scaled selection 2*N*s used for the initial condition
        gamma_evolve: scaled selection 2*N*s used in the PDE evolution
        t_max: final time in units of 2*N generations
    """
    if weights is None:
        weights = LossWeights()

    key = jr.PRNGKey(seed)
    key, model_key = jr.split(key)

    # Initialize model
    if model is None:
        model = P2INN(model_key)

    # sol_gamma = np.linspace(*gamma_range, 100)
    # sol_gamma_jax = jnp.linspace(*gamma_range, 100)
    # sol_x, sols = get_final_sols(sol_gamma, gamma_init, tf = t_max)

    history = {"total": [], "pde": [], "ic": [], "bc_left": [], "bc_right": [], "non_neg": []}#, 'sol': []}

    def _log(epoch, loss_val, loss_dict, phase="Adam"):
        history["total"].append(float(loss_val))
        history["pde"].append(float(loss_dict["pde"]))
        history["ic"].append(float(loss_dict["ic"]))
        history["bc_left"].append(float(loss_dict["bc_left"]))
        history["bc_right"].append(float(loss_dict["bc_right"]))
        history["non_neg"].append(float(loss_dict["non_neg"]))
        # history["sol"].append(float(loss_dict["sol"]))

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"[{phase}] Epoch {epoch+1:>6d} | Total: {loss_val:.2e} | "
                  f"PDE: {loss_dict['pde']:.2e} | "
                  f"IC: {loss_dict['ic']:.2e} | "
                  f"BC_L: {loss_dict['bc_left']:.2e} | "
                  f"BC_R: {loss_dict['bc_right']:.2e} | "
                  f"NonNeg: {loss_dict['non_neg']:.2e} | ")
                #   f"Sol: {loss_dict['sol']:.2e}")

    # ---- Phase 1: Adam with stochastic collocation ----
    if num_epochs_adam > 0:
        print(f"Starting Adam phase ({num_epochs_adam} epochs)")

        if lr_schedule == "cosine":
            schedule = optax.cosine_decay_schedule(lr, num_epochs_adam)
            optimizer = optax.adam(schedule)
        else:
            optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def adam_step(model, opt_state, colloc_xt, x_ic, t_bc, gamma_evol):
            (loss_val, loss_dict), grads = eqx.filter_value_and_grad(
                lambda m: total_loss(m, colloc_xt, x_ic, t_bc, gamma_evol, gamma_init, weights, theta = theta, nu = nu),
                has_aux=True
            )(model)
            updates, opt_state_new = optimizer.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            model_new = eqx.apply_updates(model, updates)
            return model_new, opt_state_new, loss_val, loss_dict

        for epoch in range(num_epochs_adam):
            key, sample_key = jr.split(key)
            colloc_xt, x_ic, t_bc, gamma = sample_collocation_and_param(
                sample_key, gamma_range, n_gamma, n_interior, t_max
            )
            model, opt_state, loss_val, loss_dict = adam_step(
                model, opt_state, colloc_xt, x_ic, t_bc, gamma_sign * gamma, 
            )
            _log(epoch, loss_val, loss_dict, "Adam")


    # ---- Phase 2: L-BFGS with fixed collocation ----
    if num_epochs_lbfgs > 0:
        import jaxopt

        print(f"\nStarting L-BFGS phase ({num_epochs_lbfgs} max iterations)")

        # Use a fixed set of collocation points for deterministic gradients
        key, sample_key = jr.split(key)
        colloc_xt_fixed, x_ic_fixed, t_bc_fixed, gamma = sample_collocation_and_param(
            sample_key, gamma_range, n_gamma, n_interior, t_max
        )

        # Split model into trainable params and static structure
        params, static = eqx.partition(model, eqx.is_array)

        def lbfgs_objective(params):
            model_rebuilt = eqx.combine(params, static)
            loss_val, loss_dict = total_loss(
                model_rebuilt, colloc_xt_fixed, x_ic_fixed, t_bc_fixed,
                gamma, gamma_init, weights, theta = theta, nu = nu
            )
            return loss_val, loss_dict

        solver = jaxopt.LBFGS(
            fun=lbfgs_objective,
            maxiter=1,
            has_aux=True,
            tol=1e-9,
        )

        # Run the solver step-by-step so we can log history
        lbfgs_state = solver.init_state(params)

        for step in range(num_epochs_lbfgs):
            params, lbfgs_state = solver.update(
                params, lbfgs_state
            )

            # Log using the value and aux already computed in the state
            loss_val = lbfgs_state.value
            loss_dict = lbfgs_state.aux
            epoch_global = num_epochs_adam + step
            _log(epoch_global, loss_val, loss_dict, "L-BFGS")

            # Check convergence
            if lbfgs_state.error < 1e-9:
                print(f"[L-BFGS] Converged at step {step+1} "
                      f"(error={lbfgs_state.error:.2e})")
                break

        # Rebuild final model
        model = eqx.combine(params, static)

    return model, history