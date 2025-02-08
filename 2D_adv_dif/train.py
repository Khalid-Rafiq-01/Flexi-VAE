import os
import uuid
import torch
import wandb
import argparse
import logging
import datetime
import numpy as np
from data import *
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from config import Config, load_config

from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict

from model_io import load_model, save_model

from data import load_from_path, prepare_adv_diff_dataset, AdvectionDiffussionDataset, get_train_val_test_folds, IntervalSplit, exact_solution

# We we define all our model here:
from new_model import Encoder, Decoder, Propagator_concat as Propagator, Model, loss_function

import warnings
warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue")


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')


def get_model(latent_dim):
    encoder = Encoder(latent_dim)
    decoder  = Decoder(latent_dim)  # Decoder for x(t)
    propagator = Propagator(latent_dim) # z(t) --> z(t+tau)
    model = Model(encoder, decoder, propagator)
    return model

def get_data_loader(dataset, batch_size):
    data = list(zip(dataset.X, dataset.X_tau, dataset.t_values, dataset.tau_values, dataset.alpha_values))
    data = data[: len(data) - len(data) % batch_size]
    return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)

    
def plot_prediction(x, x_tau, x_hat, x_hat_tau, tau, alpha):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))  # 2 rows, 2 columns

    # Plot each field
    axes[0, 0].imshow(x.cpu().squeeze().numpy(), cmap="jet")
    axes[0, 0].set_title("x (Original)", fontsize=12)

    axes[0, 1].imshow(x_tau.cpu().squeeze().numpy(), cmap="jet")
    axes[0, 1].set_title("x_tau (Ground Truth)", fontsize=12)

    axes[1, 0].imshow(x_hat.cpu().squeeze().detach().numpy(), cmap="jet")
    axes[1, 0].set_title("x_hat (Reconstruction)", fontsize=12)

    axes[1, 1].imshow(x_hat_tau.cpu().squeeze().detach().numpy(), cmap="jet")
    axes[1, 1].set_title("x_hat_tau (Predicted)", fontsize=12)

    # Add a common title for the figure
    fig.suptitle(f"Tau: {tau.item()}, Re: {alpha.item():.2f}", fontsize=12)

    # Remove axes for clean visualization
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    return fig, axes


def validate(config: Config, model, val_loader, step):
    model.eval()
    losses = []
    for batch in val_loader:
        x, x_tau, t, tau, alpha = batch
        x, x_tau, t, tau, alpha = x.cuda().float().unsqueeze(1), x_tau.cuda().float().unsqueeze(1), t.cuda().float().unsqueeze(1), tau.cuda().float().unsqueeze(1), alpha.cuda().float().unsqueeze(1)
        x_hat, x_hat_tau, mean, log_var, z_tau, _ = model(x, tau, alpha)
        reconstruction_loss, reconstruction_loss_tau, KLD = loss_function(x, x_tau, x_hat, x_hat_tau, mean, log_var)
        loss = reconstruction_loss + config.gamma * reconstruction_loss_tau + config.beta * KLD
        losses.append(loss.item())

    # plot the last sample
    fig, ax = plot_prediction(x[0], x_tau[0], x_hat[0], x_hat_tau[0], tau[0], alpha[0])
    wandb.log({'plot_val': fig}, step=step)
    plt.close(fig)
    model.train()
    return np.mean(losses)

    
def train(config: Config):
    os.makedirs(config.save_dir, exist_ok=True)
    # model id name + timestamp + random uuid
    model_id = f'{config.name}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}_{str(uuid.uuid4()).split("-")[0]}'
    save_path = os.path.join(config.save_dir, model_id)
    conf = asdict(config)
    conf['save_path'] = save_path

    wandb.init(project='FlexiPropagator_2D', config=conf)

    
    model = get_model(config.latent_dim)
    optimizer = Adam(model.parameters(), lr=config.lr)

    logger.info('Model and optimizer created')
    logger.info('Loading data')
    # tau_range = (int(config.tau_left_fraction * config.num_time_steps), int(config.tau_right_fraction * config.num_time_steps))
    # dataset_train, dataset_val, Re_interval_split, tau_interval_split = get_train_val_test_folds((1000, 3000),
    #                                                                                          tau_range,
    #                                                                                       n_samples_train=config.n_samples_train)
    
    dataset_train, dataset_val, alpha_interval_split, tau_interval_split = load_from_path("data")
    logger.info('Data loaded')
    train_loader = get_data_loader(dataset_train, config.batch_size)
    val_loader = get_data_loader(dataset_val, config.batch_size)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, epochs=config.num_epochs, steps_per_epoch=len(train_loader))

    model = model.cuda()
    model.train()

    total_steps = len(train_loader) * config.num_epochs

    pbar = tqdm(range(total_steps), total=total_steps, desc='Training')

    step = 0
    val_every_int = int(config.val_every * len(train_loader))
    plot_train_every_int = int(config.plot_train_every * len(train_loader))
    best_val_loss = float('inf')

    logger.info('Starting training')
    for epoch in range(config.num_epochs):
        wandb.log({'epoch': epoch}, step=step)
        for batch in train_loader:
            x, x_tau, t, tau, alpha = batch
            x, x_tau, t, tau, alpha = x.cuda().float().unsqueeze(1), x_tau.cuda().float().unsqueeze(1), t.cuda().float().unsqueeze(1), tau.cuda().float().unsqueeze(1), alpha.cuda().float().unsqueeze(1)
            optimizer.zero_grad()
            x_hat, x_hat_tau, mean, log_var, z_tau, _ = model(x, tau, alpha)
            reconstruction_loss, reconstruction_loss_tau, KLD = loss_function(x, x_tau, x_hat, x_hat_tau, mean, log_var)
            loss = reconstruction_loss + config.gamma * reconstruction_loss_tau + config.beta * KLD
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())
            if step % 100 == 0:
                wandb.log({'loss': loss.item(), 'reconstruction_loss': reconstruction_loss.item(), 'reconstruction_loss_tau': reconstruction_loss_tau.item(), 'KLD': KLD.item(), 'lr': scheduler.get_last_lr()[0]}, step=step)


            with torch.no_grad():
                # if step % plot_train_every_int == 0:
                #     # plot train 
                #     fig, ax = plot_prediction(x[0], x_tau[0], x_hat[0], x_hat_tau[0], tau[0], re[0])
                #     wandb.log({'plot': fig}, step=step)
                #     # plt.close(fig)


                if step % val_every_int == 0:
                    val_loss = validate(config, model, val_loader, step=step)
                    wandb.log({'val_loss': val_loss}, step=step)

                    # save latest
                    # torch.save(model.state_dict(), 'model_latest.pt')
                    save_model(save_path + '_latest.pt', model, tau_interval_split, alpha_interval_split, config)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_model(save_path + '_best.pt', model, tau_interval_split, alpha_interval_split, config)
                        
                    model.train()
            step += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    #parser.add_argument('--n_samples_train', type=int, default=25000)
    parser.add_argument('--gamma', type=float, default=2.5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--val_every', type=float, default=0.25)
    parser.add_argument('--plot_train_every', type=float, default=0.25)
    

    parser.add_argument('--config', type=str, required=False)
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        conf = dict(vars(args))
        conf.pop('config')
        config = Config(**conf)

    train(config)
