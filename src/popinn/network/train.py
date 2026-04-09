from __future__ import annotations

from typing import Callable, Union

import equinox as eqx
import jax.random as jr
import optax

from ..config import AdamConfig, Batch, LBFGSConfig

import jax


# ---------------------------------------------------------------------------
# Internal training phases
# ---------------------------------------------------------------------------

def _train_adam(
    model,
    sample_fn: Callable,
    loss_fn: Callable,
    cfg: AdamConfig,
    key,
):
    """Run the Adam optimisation phase.

    Args:
        model:     Equinox model to train.
        sample_fn: ``key -> Batch`` callable, resampled every epoch.
        loss_fn:   ``(model, Batch) -> (float, dict)`` callable.
        cfg:       Adam hyperparameters.
        key:       JAX PRNG key (consumed and returned updated).

    Returns:
        ``(model, key, history)``
    """
    if cfg.lr_schedule == "cosine":
        schedule = optax.cosine_decay_schedule(cfg.lr, cfg.num_epochs)
        optimizer = optax.adam(schedule)
    else:
        optimizer = optax.adam(cfg.lr)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, batch: Batch):
        (loss_val, loss_dict), grads = eqx.filter_value_and_grad(
            lambda m: loss_fn(m, batch), has_aux=True
        )(model)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss_val, loss_dict

    history: dict[str, list] = {}
    print(f"[Adam] Starting ({cfg.num_epochs} epochs)")

    for epoch in range(cfg.num_epochs):
        key, sample_key = jr.split(key)
        batch = sample_fn(sample_key)
        model, opt_state, loss_val, loss_dict = step(model, opt_state, batch)

        history.setdefault("total", []).append(float(loss_val))
        for k, v in loss_dict.items():
            history.setdefault(k, []).append(float(v))

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            parts = "  ".join(f"{k}: {v:.2e}" for k, v in loss_dict.items())
            print(f"[Adam] Epoch {epoch + 1:>6d} | total: {loss_val:.2e} | {parts}")

    return model, key, history


def _train_lbfgs(
    model,
    fixed_batch: Batch,
    loss_fn: Callable,
    cfg: LBFGSConfig,
):
    """Run the L-BFGS optimisation phase on a fixed batch.

    Args:
        model:       Equinox model to train.
        fixed_batch: Collocation batch sampled once before this phase begins.
        loss_fn:     ``(model, Batch) -> (float, dict)`` callable.
        cfg:         L-BFGS hyperparameters.

    Returns:
        ``(model, history)``
    """
    import jaxopt

    params, static = eqx.partition(model, eqx.is_array)

    def objective(params):
        model_rebuilt = eqx.combine(params, static)
        return loss_fn(model_rebuilt, fixed_batch)

    solver = jaxopt.LBFGS(
        fun=objective,
        maxiter=1,
        has_aux=True,
        tol=cfg.tol,
    )

    lbfgs_state = solver.init_state(params)
    history: dict[str, list] = {}
    print(f"[L-BFGS] Starting ({cfg.num_epochs} max iterations)")

    for step in range(cfg.num_epochs):
        params, lbfgs_state = solver.update(params, lbfgs_state)

        loss_val = lbfgs_state.value
        loss_dict = lbfgs_state.aux

        history.setdefault("total", []).append(float(loss_val))
        for k, v in loss_dict.items():
            history.setdefault(k, []).append(float(v))

        if (step + 1) % cfg.log_every == 0 or step == 0:
            parts = "  ".join(f"{k}: {v:.2e}" for k, v in loss_dict.items())
            print(f"[L-BFGS] Step {step + 1:>6d} | total: {loss_val:.2e} | {parts}")

        if lbfgs_state.error < cfg.tol:
            print(f"[L-BFGS] Converged at step {step + 1} "
                  f"(error={lbfgs_state.error:.2e})")
            break

    model = eqx.combine(params, static)
    return model, history


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_model(
    model,
    sample_fn: Callable,
    loss_fn: Callable,
    optimizers: Union[AdamConfig, LBFGSConfig, list],
    key,
):
    """Train a PINN with an arbitrary sequence of optimisers.

    Args:
        model:      An Equinox model (e.g. ``PINN`` or ``P2INN``).
        sample_fn:  ``key -> Batch`` callable.  Called once per Adam epoch;
                    called once before each L-BFGS phase to obtain a fixed batch.
        loss_fn:    ``(model, Batch) -> (float, dict)`` callable.
        optimizers: A single optimizer config or a list of them.  Phases are
                    executed in order, e.g. ``[AdamConfig(), LBFGSConfig()]``
                    runs Adam then L-BFGS.
        seed:       Random seed.

    Returns:
        ``(model, history)`` where ``history`` is a dict mapping loss component
        names to lists of values recorded across all phases.

    Examples::

        # Adam only
        model, history = train_model(model, sample_fn, loss_fn, AdamConfig())

        # Adam → L-BFGS
        model, history = train_model(
            model, sample_fn, loss_fn,
            [AdamConfig(num_epochs=5000), LBFGSConfig(num_epochs=2000)],
        )
    """
    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    # key = jr.PRNGKey(seed)
    history: dict[str, list] = {}

    for opt_cfg in optimizers:
        if isinstance(opt_cfg, AdamConfig):
            model, key, phase_history = _train_adam(
                model, sample_fn, loss_fn, opt_cfg, key
            )
        elif isinstance(opt_cfg, LBFGSConfig):
            key, sample_key = jr.split(key)
            fixed_batch = sample_fn(sample_key)
            model, phase_history = _train_lbfgs(
                model, fixed_batch, loss_fn, opt_cfg
            )
        else:
            raise TypeError(
                f"Unknown optimizer config type: {type(opt_cfg)}. "
                "Expected AdamConfig or LBFGSConfig."
            )

        for k, v in phase_history.items():
            history.setdefault(k, []).extend(v)

    return model, history
