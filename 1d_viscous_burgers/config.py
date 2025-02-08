from dataclasses import dataclass
import json

@dataclass
class Config:
    # default values. DO NOT TOUCH
    name: str = 'FlexiPropagator'
    latent_dim: int = 2
    input_dim: int = 128
    batch_size: int = 128
    lr: float = 3e-4
    num_epochs: int = 200
    n_samples_train: int = 8_00_000

    num_time_steps: int = 500

    tau_left_fraction: float = 0.35
    tau_right_fraction: float = 0.85

    gamma: float = 3.25
    beta: float = 1e-4

    val_every: float = 0.25
    plot_train_every: float = 0.01

    save_dir: str = 'checkpoints'

def load_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return Config(**config)