import torch
from dataclasses import dataclass, asdict
from data import IntervalSplit
from config import Config


def save_model(path, model, tau_interval_split, alpha_interval_split, config):
    torch.save({
        'model_state_dict': model.state_dict(),
        'alpha_interval_split': asdict(alpha_interval_split),
        'tau_interval_split': asdict(tau_interval_split),
        'config': asdict(config),
    }, path)


def load_model(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    alpha_interval_split = IntervalSplit(**checkpoint['alpha_interval_split'])
    tau_interval_split = IntervalSplit(**checkpoint['tau_interval_split'])
    config = Config(**checkpoint['config'])
    return model, alpha_interval_split, tau_interval_split, config
