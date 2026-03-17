import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from ..physics.loss import LossWeights, total_loss
from .model import PINN
from .sampling import sample_collocation
import jax

def train(
    # Physics parameters
    theta: float = 1.0,
    nu: float = 1.0,
    gamma_evol: float = 0.0,
    gamma_init: float = 0.,
    t_max: float = 0.5,
    # Network architecture
    hidden_dims: list[int] = None,
    # Training parameters
    num_epochs: int = 10_000,
    lr: float = 1e-3,
    lr_schedule: str = "cosine",  # "constant" or "cosine"
    n_interior: int = 100,
    n_bc: int = 64,
    n_ic: int = 128,
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
        # optimizer = optax.adam(schedule)
    else:
        optimizer = optax.adam(lr)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # JIT-compiled training step
    @eqx.filter_jit
    def step(model, opt_state, colloc_xt, x_ic, t_bc):
        (loss_val, loss_dict), grads = eqx.filter_value_and_grad(
            lambda m: total_loss(m, colloc_xt, x_ic, t_bc, gamma_evol, gamma_init, weights, theta = theta, nu = nu),
            has_aux=True
        )(model)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss_val, loss_dict

    # Training loop
    history = {"total": [], "pde": [], "ic": [], "bc_left": [], "bc_right": [], "non_neg": []}
    coll = []
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
                  f"IC: {loss_dict['ic']:.2e} | "
                  f"PDE: {loss_dict['pde']:.2e} | "
                  f"BC_L: {loss_dict['bc_left']:.2e} | "
                  f"BC_R: {loss_dict['bc_right']:.2e} | "
                  f"NonNeg: {loss_dict['non_neg']:.2e}")

    return model, history



def train_adam_lbfgs(
    model = None,
    # Physics parameters
    theta: float = 1.0,
    gamma_init: float = 0.0,
    gamma_evol: float = 0.0,
    nu: float = 1.0,
    t_max: float = 0.5,
    # Network architecture
    hidden_dims: list[int] = None,
    # Training parameters
    num_epochs_adam: int = 10_000,
    num_epochs_lbfgs: int = 0,
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
    if hidden_dims is None:
        hidden_dims = [64, 64, 64, 64]
    if weights is None:
        weights = LossWeights()

    key = jr.PRNGKey(seed)
    key, model_key = jr.split(key)

    # Initialize model
    if model is None:
        model = PINN(model_key, hidden_dims=hidden_dims)

    history = {"total": [], "pde": [], "ic": [], "bc_left": [], "bc_right": [], "non_neg": []}

    def _log(epoch, loss_val, loss_dict, phase="Adam"):
        history["total"].append(float(loss_val))
        history["pde"].append(float(loss_dict["pde"]))
        history["ic"].append(float(loss_dict["ic"]))
        history["bc_left"].append(float(loss_dict["bc_left"]))
        history["bc_right"].append(float(loss_dict["bc_right"]))
        history["non_neg"].append(float(loss_dict["non_neg"]))

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"[{phase}] Epoch {epoch+1:>6d} | Total: {loss_val:.2e} | "
                  f"PDE: {loss_dict['pde']:.2e} | "
                  f"BC_L: {loss_dict['bc_left']:.2e} | "
                  f"BC_R: {loss_dict['bc_right']:.2e} | "
                  f"NonNeg: {loss_dict['non_neg']:.2e}")

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
        def adam_step(model, opt_state, colloc_xt, x_ic, t_bc):
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
            colloc_xt, x_ic, t_bc = sample_collocation(
                sample_key, n_interior, n_bc, n_ic, t_max
            )
            model, opt_state, loss_val, loss_dict = adam_step(
                model, opt_state, colloc_xt, x_ic, t_bc
            )
            _log(epoch, loss_val, loss_dict, "Adam")

    # ---- Phase 2: L-BFGS with fixed collocation ----
    if num_epochs_lbfgs > 0:
        import jaxopt

        print(f"\nStarting L-BFGS phase ({num_epochs_lbfgs} max iterations)")

        # Use a fixed set of collocation points for deterministic gradients
        key, sample_key = jr.split(key)
        colloc_xt_fixed, x_ic_fixed, t_bc_fixed = sample_collocation(
            sample_key, n_interior, n_bc, n_ic, t_max
        )

        # Split model into trainable params and static structure
        params, static = eqx.partition(model, eqx.is_array)

        def lbfgs_objective(params):
            model_rebuilt = eqx.combine(params, static)
            loss_val, loss_dict = total_loss(
                model_rebuilt, colloc_xt_fixed, x_ic_fixed, t_bc_fixed,
                gamma_evol, gamma_init, weights, theta = theta, nu = nu
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


def train_adam_lbfgs_optax(
    model = None,
    # Physics parameters
    theta: float = 1.0,
    gamma_init: float = 0.0,
    gamma_evol: float = 0.0,
    nu: float = 1.0,
    t_max: float = 0.5,
    # Network architecture
    hidden_dims: list[int] = None,
    # Training parameters
    num_epochs_adam: int = 10_000,
    num_epochs_lbfgs: int = 0,
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
    if hidden_dims is None:
        hidden_dims = [64, 64, 64, 64]
    if weights is None:
        weights = LossWeights()

    key = jr.PRNGKey(seed)
    key, model_key = jr.split(key)

    # Initialize model
    if model is None:
        model = PINN(model_key, hidden_dims=hidden_dims)
    

    history = {"total": [], "pde": [], "ic": [], "bc_left": [], "bc_right": [], "non_neg": []}

    def _log(epoch, loss_val, loss_dict, phase="Adam"):
        history["total"].append(float(loss_val))
        history["pde"].append(float(loss_dict["pde"]))
        history["ic"].append(float(loss_dict["ic"]))
        history["bc_left"].append(float(loss_dict["bc_left"]))
        history["bc_right"].append(float(loss_dict["bc_right"]))
        history["non_neg"].append(float(loss_dict["non_neg"]))

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"[{phase}] Epoch {epoch+1:>6d} | Total: {loss_val:.2e} | "
                  f"PDE: {loss_dict['pde']:.2e} | "
                  f"BC_L: {loss_dict['bc_left']:.2e} | "
                  f"BC_R: {loss_dict['bc_right']:.2e} | "
                  f"NonNeg: {loss_dict['non_neg']:.2e}")

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
        def adam_step(model, opt_state, colloc_xt, x_ic, t_bc):
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
            colloc_xt, x_ic, t_bc = sample_collocation(
                sample_key, n_interior, n_bc, n_ic, t_max
            )
            model, opt_state, loss_val, loss_dict = adam_step(
                model, opt_state, colloc_xt, x_ic, t_bc
            )
            _log(epoch, loss_val, loss_dict, "Adam")

    # ---- Phase 2: L-BFGS with fixed collocation ----
    if num_epochs_lbfgs > 0:
        print(f"\nStarting L-BFGS phase ({num_epochs_lbfgs} max iterations)")

        # Use a fixed set of collocation points for deterministic gradients
        key, sample_key = jr.split(key)
        colloc_xt_fixed, x_ic_fixed, t_bc_fixed = sample_collocation(
            sample_key, n_interior, n_bc, n_ic, t_max
        )

        # Split model into trainable params and static structure
        full_params, static = eqx.partition(model, eqx.is_array)

        # Flatten params into a single vector for L-BFGS
        params, unravel_fn = jax.flatten_util.ravel_pytree(full_params)

        def lbfgs_loss(params):
            """Scalar loss for L-BFGS (no aux)."""
            p = unravel_fn(params)
            model_rebuilt = eqx.combine(p, static)
            loss_val, _ = total_loss(
                model_rebuilt, colloc_xt_fixed, x_ic_fixed, t_bc_fixed, gamma_evol, gamma_init, weights, theta = theta, nu = nu
            )
            return loss_val
        
        def lbfgs_loss_detailed(params):
            """Full loss with component dict for logging."""
            p = unravel_fn(params)
            model_rebuilt = eqx.combine(p, static)
            return total_loss(
                model_rebuilt, colloc_xt_fixed, x_ic_fixed, t_bc_fixed, gamma_evol, gamma_init, weights, theta = theta, nu = nu
            )

        # def lbfgs_value_and_grad(params):
        #     model_rebuilt = eqx.combine(params, static)
        #     (loss_val, loss_dict), grads = eqx.filter_value_and_grad(
        #         lambda m: total_loss(
        #             m, colloc_xt_fixed, x_ic_fixed, t_bc_fixed, gamma_evol, gamma_init, weights, theta = theta, nu = nu
        #         ),
        #         has_aux=True
        #     )(model_rebuilt)
        #     # Partition grads to match params structure
        #     grad_params, _ = eqx.partition(grads, eqx.is_array)
        #     return loss_val, grad_params, loss_dict

        lbfgs_value_and_grad = eqx.filter_value_and_grad(lbfgs_loss)

        lbfgs = optax.lbfgs()
        opt_state = lbfgs.init(params)

        # L-BFGS needs value_and_grad passed to the update
        value, grad = lbfgs_value_and_grad(params)
        _, loss_dict = lbfgs_loss_detailed(params)
        _log(num_epochs_adam, value, loss_dict, "L-BFGS")


        @eqx.filter_jit
        def lbfgs_step(flat_params, grad, opt_state, value):
            updates, new_opt_state = lbfgs.update(
                grad, opt_state, flat_params,
                value=value, grad=grad,
                value_fn=lbfgs_loss,
            )
            new_params = optax.apply_updates(flat_params, updates)
            new_value, new_grad = lbfgs_value_and_grad(new_params)
            return new_params, new_grad, new_opt_state, new_value




        for step in range(1, num_epochs_lbfgs):
            updates, opt_state = lbfgs.update(
                grad, opt_state, params,
                value=value, grad=grad,
                value_fn=lbfgs_loss,
            )
            params, grad, opt_state, value = lbfgs_step(
                params, grad, opt_state, value
            )

            if (num_epochs_adam + step + 1) % log_every == 0 or step == 1:
                _, loss_dict = lbfgs_loss_detailed(params)
            _log(num_epochs_adam + step, value, loss_dict, "L-BFGS")

            # Check for convergence via gradient norm
            grad_norm = optax.global_norm(grad)
            if grad_norm < 1e-9:
                print(f"[L-BFGS] Converged at step {step} "
                      f"(grad_norm={grad_norm:.2e})")
                break

        rebuild_params = unravel_fn(params)

        # Rebuild final model
        model = eqx.combine(rebuild_params, static)

    return model, history