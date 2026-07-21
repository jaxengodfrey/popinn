import dataclasses
from collections.abc import Callable, Mapping

import equinox as eqx
import jax.random as jr
import optax

from .models import AbstractModel

# ──────────────────────────────────────────────────────────────
# Optimizer configs
# ──────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class AdamConfig:
    """Configuration for an Adam optimization phase."""

    lr: float | optax.Schedule = 1e-3
    num_epochs: int = 1000
    log_every: int = 500
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8


AdamConfig.__init__.__doc__ = """Optimizer is `optax.adam`.

    Args:
        lr (float | optax.Schedule): Constant learning rate, or an optax
            schedule mapping step count to learning rate. Passed straight to
            optax.adam. When using a step-based schedule (e.g. cosine decay),
            its decay horizon should generally match `num_epochs`.
        num_epochs (int): Number of optimization steps in this phase.
        log_every (int): Print a log line every this many steps.
        b1 (float): Adam's first-moment decay rate.
        b2 (float): Adam's second-moment decay rate.
        eps (float): Adam's epsilon for numerical stability.

"""


@dataclasses.dataclass(frozen=True)
class LBFGSConfig:
    """Configuration for an L-BFGS optimization phase."""

    num_epochs: int = 1000
    log_every: int = 500
    tol: float = 1e-6
    history_size: int = 10
    optimizer_kwargs: Mapping = dataclasses.field(default_factory=dict)


LBFGSConfig.__init__.__doc__ = """Optimizer is `jaxopt.LBFGS`. Will be updated to the `optax` version in the future.

    Args:
        num_epochs (int): Maximum number of L-BFGS iterations.
        log_every (int): Print a log line every this many iterations.
        tol (float): Convergence tolerance; iteration stops when the solver's
            error drops below this.
        history_size (int): Number of past updates L-BFGS keeps to
            approximate the inverse Hessian.
        optimizer_kwargs (Mapping): Mapping for extra kwargs passed to the optimizer
"""


def warmup_cosine(
    peak_lr: float,
    num_epochs: int,
    init_lr: float = 1e-5,
    warmup_steps: int = 500,
) -> optax.Schedule:
    """Convenience wrapper for `optax.warmup_cosine_decay_schedule`.

    Args:
        peak_lr (float): peak learning rate reached at the end of warmup.
        num_epochs (int): total steps; the cosine decay spans this horizon.
        init_lr (float): learning rate at step 0.
        warmup_steps (int): number of steps spent ramping up to peak_lr.

    Returns:
        (optax.Schedule): a callable step -> learning rate.

    Raises:
        ValueError: if warmup_steps >= num_epochs (no steps left to decay
            over). optax's decay span is num_epochs - warmup_steps, which
            must be positive.
    """
    if warmup_steps >= num_epochs:
        raise ValueError(f"warmup_steps ({warmup_steps}) must be < num_epochs ({num_epochs}); the cosine decay spans the remaining steps.")
    return optax.warmup_cosine_decay_schedule(
        init_value=init_lr,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=num_epochs,
    )


# ──────────────────────────────────────────────────────────────
# Training phases
# ──────────────────────────────────────────────────────────────


def train_adam(
    model: AbstractModel,
    loss_fn: Callable,
    data: eqx.Module,
    cfg: AdamConfig,
    sample_fn: Callable | None = None,
    resample_every: int = 1,
    key: int | None = None,
):
    """Run the Adam optimization phase, optionally resampling the training data.

    Args:
        model (popinn.AbstractModel): The model to train.
        loss_fn (Callable): Callable with signature `loss_fn(model, data) -> (scalar, dict)`
            returning the total loss and a per-term breakdown.
        data (eqx.Module): The training-data container read by `loss_fn`
            that contains the coordinates and auxiliary inputs. When `sample_fn` is given,
            this is the initial batch.
        cfg (popinn.AdamConfig): Adam hyperparameters, including the learning rate
            or schedule (see AdamConfig).
        sample_fn (Callable | None): Optional resampler with signature
            `fn(key) -> equinox.Module`. When provided, a fresh batch is drawn every
            `resample_every` epochs (epoch 0 uses the passed `batch`). The
            sampler MUST return batches of constant shape, or each resample
            triggers jit compilation of the step. None keeps the batch
            fixed for the whole phase.
        resample_every (int): Resample interval in epochs; only used when
            `sample_fn` is given. Default 1 (resample every epoch).
        key (PRNGKeyArray | None): `jax.random.PRNGkey` passed to `sample_fn`. Only used
            when `sample_fn` is given; seed defaults to 0 when `sample_fn` is given with no `key`
            specified.

    Returns:
        (tuple[AbstractModel, dict]): The trained model and a history dict mapping
            'total' and each loss-component name to a list of per-epoch
            values.
    """
    optimizer = optax.adam(cfg.lr, b1=cfg.b1, b2=cfg.b2, eps=cfg.eps)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, batch):
        (loss_val, loss_dict), grads = eqx.filter_value_and_grad(lambda m: loss_fn(m, batch), has_aux=True)(model)
        updates, opt_state_new = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss_val, loss_dict

    if sample_fn is not None and key is None:
        key = jr.PRNGKey(0)

    history: dict[str, list] = {}
    print(f"[Adam] Starting ({cfg.num_epochs} epochs)")

    for epoch in range(cfg.num_epochs):
        # Resample the batch every `resample_every` epochs (epoch 0 keeps the
        # provided batch so its shapes set the compilation key for `step`).
        if sample_fn is not None and epoch > 0 and epoch % resample_every == 0:
            key, subkey = jr.split(key)
            data = sample_fn(subkey)

        model, opt_state, loss_val, loss_dict = step(model, opt_state, data)

        history.setdefault("total", []).append(float(loss_val))
        for k, v in loss_dict.items():
            history.setdefault(k, []).append(float(v))

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            parts = "  ".join(f"{k}: {v:.2e}" for k, v in loss_dict.items())
            print(f"[Adam] Epoch {epoch + 1:>6d} | total: {loss_val:.2e} | {parts}")

    return model, history


def train_lbfgs(
    model: AbstractModel,
    data: eqx.Module,
    loss_fn: Callable,
    cfg: LBFGSConfig,
):
    """Run the L-BFGS optimization phase on a fixed batch.

    L-BFGS accumulates a curvature history that assumes a fixed objective,
    so the training data cannot be resampled.

    Args:
        model (AbstractModel): The model to train.
        data (eqx.Module): The training data container read by `loss_fn`;
            held fixed across all iterations of this phase.
        loss_fn (Callable): Function with signature `loss_fn(model, batch) -> (scalar, dict)`
            returning the total loss and a per-term breakdown.
        cfg (LBFGSConfig): L-BFGS hyperparameters (max iterations,
            tolerance, history size, logging interval).

    Returns:
        (tuple[AbstractModel, dict]): The trained model and a history dict mapping
            'total' and each loss-component name to a list of per-iteration
            values.
    """
    import jaxopt

    reserved = {"fun", "maxiter", "has_aux", "tol", "history_size"}
    overlap = reserved & cfg.optimizer_kwargs.keys()
    if overlap:
        raise ValueError(f"solver_kwargs may not override {sorted(overlap)}; set tol and history_size via their own LBFGSConfig fields.")

    params, static = eqx.partition(model, eqx.is_array)

    def objective(params):
        model_rebuilt = eqx.combine(params, static)
        return loss_fn(model_rebuilt, data)

    solver = jaxopt.LBFGS(fun=objective, maxiter=1, has_aux=True, tol=cfg.tol, history_size=cfg.history_size, **cfg.optimizer_kwargs)

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
            print(f"[L-BFGS] Converged at step {step + 1} (error={lbfgs_state.error:.2e})")
            break

    model = eqx.combine(params, static)
    return model, history


def train_model(
    model,
    data,
    loss_fn: Callable,
    optimizers: AdamConfig | LBFGSConfig | list,
    sample_fn: Callable | None = None,
    resample_every: int = 1,
    seed: int = 0,
):
    """Train a model with an arbitrary sequence of optimization phases.

    Phases run in order; each is dispatched to `train_adam` or `train_lbfgs` by
    its config type. If a resampler is supplied, Adam phases resample every
    `resample_every` epochs, while each L-BFGS phase draws a single fresh
    batch at its start and holds it fixed.

    Args:
        model (AbstractModel): The model to train.
        data (eqx.Module): The training data container read by `loss_fn`;
            held fixed across all iterations of this phase.
        loss_fn (Callable): Function with signature `loss_fn(model, data) -> (scalar, dict)`
            returning the total loss and a per-term breakdown.
        optimizers (AdamConfig | LBFGSConfig | list): A single optimizer
            config or a list of them. They executed in order, e.g.
            `[AdamConfig(), LBFGSConfig()]` runs Adam then L-BFGS.
        sample_fn (Callable | None): Optional resampler with signature
            `fn(key) -> equinox.Module`. Enables data resampling for Adam
            phases and resamples a single batch at the start of L-BFGS. None keeps `data`
            fixed throughout.
        resample_every (int): Resample interval in epochs for Adam phases;
            only used when `sample_fn` is given. Default 1.
        seed (int): Seed for the PRNG used for resampling.

    Returns:
        (tuple[eqx.Module, dict]): The trained model and a history dict mapping
            loss-component names to lists of values concatenated across all
            phases.

    Examples:

    ```python
        # Adam only, constant learning rate
        model, history = train_model(model, data, loss_fn, AdamConfig(lr=1e-3))

        # Adam (warmup + cosine schedule) then L-BFGS
        n = 5000
        model, history = train_model(
            model, data, loss_fn,
            [AdamConfig(lr=warmup_cosine(1e-3, n), num_epochs=n),
             LBFGSConfig(num_epochs=2000)],
        )

        # Any optax schedule works directly
        sched = optax.exponential_decay(1e-3, transition_steps=1000, decay_rate=0.9)
        model, history = train_model(model, data, loss_fn, AdamConfig(lr=sched))

        # Adam with collocation points resampled every 100 epochs
        model, history = train_model(
            model, data, loss_fn, AdamConfig(num_epochs=5000),
            sample_fn=sampler, resample_every=100, seed=0,
        )
    ```
    """
    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    key = jr.PRNGKey(seed)
    history: dict[str, list] = {}

    for opt_cfg in optimizers:
        if isinstance(opt_cfg, AdamConfig):
            key, subkey = jr.split(key)
            model, phase_history = train_adam(
                model,
                loss_fn,
                data,
                opt_cfg,
                sample_fn=sample_fn,
                resample_every=resample_every,
                key=subkey,
            )
        elif isinstance(opt_cfg, LBFGSConfig):
            # L-BFGS needs a fixed objective: draw one fresh batch for the
            # phase (if resampling is enabled), then hold it constant.
            phase_batch = data
            if sample_fn is not None:
                key, subkey = jr.split(key)
                phase_batch = sample_fn(subkey)
            model, phase_history = train_lbfgs(model, phase_batch, loss_fn, opt_cfg)
        else:
            raise TypeError(f"Unknown optimizer config type: {type(opt_cfg)}. Expected AdamConfig or LBFGSConfig.")

        for k, v in phase_history.items():
            history.setdefault(k, []).extend(v)

    return model, history
