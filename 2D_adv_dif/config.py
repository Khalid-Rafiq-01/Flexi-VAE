from dataclasses import dataclass
import json

@dataclass
class Config:
    # default values. DO NOT TOUCH
    name: str = 'FlexiPropagator_2D'
    latent_dim: int = 3
    batch_size: int = 64
    lr: float = 3e-4
    num_epochs: int = 25
    num_time_steps: int = 500

    gamma: float = 3.25
    beta: float = 1e-3

    val_every: float = 0.25
    plot_train_every: float = 0.01

    save_dir: str = 'checkpoints'

def load_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return Config(**config)
