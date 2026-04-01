import equinox as eqx
from ..network.models import PINN
import jax.random as jr

def save_model(model, path: str, metadata: dict = None):
    """Save a trained PINN to disk.
    
    Args:
        model: trained PINN (Equinox module)
        path: file path (e.g. "models/neutral_eq.eqx")
        metadata: optional dict of training params to save alongside
                  (theta, gamma_init, gamma_evolve, t_max, hidden_dims, etc.)
    """
    import json, pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)
    if metadata is not None:
        meta_path = path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {meta_path}")
    print(f"Saved model to {path}")


def load_model(path: str, hidden_dims: list[int] = None, key=None):
    """Load a trained PINN from disk.
    
    Must provide the same hidden_dims used during training so that
    the model skeleton matches the saved weights.
    
    Args:
        path: file path used in save_model
        hidden_dims: must match the architecture used during training
        key: random key for skeleton init (values are overwritten)
    
    Returns:
        model: loaded PINN
        metadata: dict if a .meta.json file exists, else None
    """
    import json, pathlib
    if hidden_dims is None:
        hidden_dims = [64, 64, 64, 64]
    if key is None:
        key = jr.PRNGKey(0)
    skeleton = PINN(key, hidden_dims=hidden_dims)
    model = eqx.tree_deserialise_leaves(path, skeleton)

    meta_path = path + ".meta.json"
    metadata = None
    if pathlib.Path(meta_path).exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    print(f"Loaded model from {path}")
    return model, metadata