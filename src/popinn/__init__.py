# Config / data structures
from .config import (
    AdamConfig,
    Batch,
    LBFGSConfig,
    LossWeights,
    PhysicsConfig,
    SamplingConfig,
)

# Models
from .network.models import PINN, P2INN, evaluate_model

# Sampling
from .network.sampling import make_pinn_sampler, make_p2inn_sampler

# Training
from .network.train import train_model

# Physics — built-in PDE
from .physics.pde import pde_residual, log_pde_residual

# Physics — equilibrium distribution
from .physics.phi import g_equilibrium

# Physics — loss
from .physics.loss import make_loss, mse, mae

# Utilities
from .utils.io import save_model, load_model
from .utils.plotting import plot_training_history