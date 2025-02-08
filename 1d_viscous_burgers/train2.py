#from model_v2 import loss_function, Encoder, Decoder, Propagator_concat_one_step as Propagator, Model_One_Step as Model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.optim as optim

from torch.optim import Adam
import torch
import wandb
from data import load_from_path, prepare_Re_dataset, get_train_val_test_folds, IntervalSplit
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import numpy as np
from torch.optim import Adam
from model_v2 import Encoder, Decoder, Propagator_concat as Propagator, Model, loss_function
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.optim as optim
import torch
import wandb
import argparse
import logging
import uuid
import datetime
import os
from config import Config, load_config
from model_io import load_model, save_model


from data import *


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')


def get_model(input_dim, latent_dim):
    # Instantiate encoder, decoder, and model
    encoder = Encoder(input_dim, latent_dim)
    decoder  = Decoder(latent_dim, input_dim)  # Decoder for x(t)
    propagator = Propagator(latent_dim) # z(t) --> z(t+tau)
    model = Model(encoder, decoder, propagator)
    return model

def get_data_loader(dataset, batch_size):
    data = list(zip(dataset.X, dataset.X_tau, dataset.t_values, dataset.tau_values, dataset.Re_values))
    data = data[: len(data) - len(data) % batch_size]
    return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)


def validate(config: Config, model, val_loader, step):
    model.eval()
    losses = []
    for batch in val_loader:
        batch: torch.Tensor
        x, x_tau, t, tau, re = batch
        x, x_tau, t, tau, re = x.cuda().float().unsqueeze(1), x_tau.cuda().float().unsqueeze(1), t.cuda().float().unsqueeze(1), tau.cuda().float().unsqueeze(1), re.cuda().float().unsqueeze(1)
        x_hat, x_hat_tau, mean, log_var, z_tau = model(x, tau, re)
        reconstruction_loss, reconstruction_loss_tau, KLD = loss_function(x, x_tau, x_hat, x_hat_tau, mean, log_var)
        loss = reconstruction_loss + config.gamma * reconstruction_loss_tau + config.beta * KLD
        losses.append(loss.item())

    # plot the last sample
    fig, ax = plot_prediction(x[0], x_tau[0], x_hat[0], x_hat_tau[0], tau[0], re[0])
    wandb.log({'plot_val': fig}, step=step)
    # plt.close(fig)
    model.train()
    return np.mean(losses)


def train(config: Config):
    os.makedirs(config.save_dir, exist_ok=True)
    # model id name + timestamp + random uuid
    model_id = f'{config.name}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}_{str(uuid.uuid4()).split("-")[0]}'
    save_path = os.path.join(config.save_dir, model_id)
    conf = asdict(config)
    conf['save_path'] = save_path

    wandb.init(project='FlexiPropagator', config=conf)

    
    model = get_model(config.input_dim, config.latent_dim)
    optimizer = Adam(model.parameters(), lr=config.lr)

    logger.info('Model and optimizer created')
    logger.info('Loading data')
    tau_range = (int(config.tau_left_fraction * config.num_time_steps), int(config.tau_right_fraction * config.num_time_steps))
    # dataset_train, dataset_val, Re_interval_split, tau_interval_split = get_train_val_test_folds((1000, 3000),
    #                                                                                          tau_range,
    #                                                                                       n_samples_train=config.n_samples_train)
    
    dataset_train, dataset_val, Re_interval_split, tau_interval_split = load_from_path("data")
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
            batch: torch.Tensor
            x, x_tau, t, tau, re = batch
            x, x_tau, t, tau, re = x.cuda().float().unsqueeze(1), x_tau.cuda().float().unsqueeze(1), t.cuda().float().unsqueeze(1), tau.cuda().float().unsqueeze(1), re.cuda().float().unsqueeze(1)
            optimizer.zero_grad()
            x_hat, x_hat_tau, mean, log_var, z_tau = model(x, tau, re)
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
                if step % plot_train_every_int == 0:
                    # plot train 
                    fig, ax = plot_prediction(x[0], x_tau[0], x_hat[0], x_hat_tau[0], tau[0], re[0])
                    wandb.log({'plot': fig}, step=step)
                    # plt.close(fig)


                if step % val_every_int == 0:
                    val_loss = validate(config, model, val_loader, step=step)
                    wandb.log({'val_loss': val_loss}, step=step)

                    # save latest
                    # torch.save(model.state_dict(), 'model_latest.pt')
                    save_model(save_path + '_latest.pt', model, tau_interval_split, Re_interval_split, config)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # torch.save(model.state_dict(), 'model_best.pt')
                        save_model(save_path + '_best.pt', model, tau_interval_split, Re_interval_split, config)

                    # plot inter extra polation
                    # fig, axs = plot_inter_extra_polation(model, Re_interval_split, tau_interval_split)
                    # wandb.log({'Inter/extra-polation': fig}, step=step)
                    model.train()
            step += 1


def plot_prediction(x, x_tau, x_hat, x_hat_tau, tau, re):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(x.cpu().squeeze().numpy(), label='x', linewidth = 3)
    ax.plot(x_tau.cpu().squeeze().numpy(), label='x_tau', linewidth = 3)
    ax.plot(x_hat.cpu().squeeze().detach().numpy(), label='x_hat', linewidth = 3)
    ax.plot(x_hat_tau.cpu().squeeze().detach().numpy(), label='x_hat_tau', linewidth = 3)
    ax.set_title(f'Tau: {tau.item()}, Re: {re.item()}', fontsize=18)
    ax.legend()

    return fig, ax




@torch.no_grad()
def plot_random_prediction(model, X, X_tau, Tau, Re):
    # generate random index for X
    i = np.random.randint(0, len(X)-1)
    # pick the i-th sample. 1, 1, 128, cuda, dtype 32
    x = torch.tensor(X[i]).cuda().float()[None, None, :]
    x_tau = torch.tensor(X_tau[i]).cuda().float()[None, None, :]
    tau = torch.tensor(Tau[i]).cuda().float()[None, None]
    re = torch.tensor(Re[i]).cuda().float()[None, None]
    x_hat, x_hat_tau, _, _, _ = model(x, tau, re)

    plt.figure(figsize = (11, 7))
    plt.plot(x.cpu().squeeze().numpy(), label='x', linewidth = 3)
    plt.plot(x_tau.cpu().squeeze().numpy(), label='x_tau', linewidth = 3)
    plt.plot(x_hat.cpu().squeeze().detach().numpy(), label='x_hat', linewidth = 3)
    plt.plot(x_hat_tau.cpu().squeeze().detach().numpy(), label='x_hat_tau', linewidth = 3)
    plt.title(f'Tau: {tau.item()}, Re: {Re[i]}', fontsize=18)
    plt.legend()
    plt.show()

def plot_inter_extra_polation(model, Re_interval_split, tau_interval_split):
    # select a sample from each interval range and plot in a wandb table

    # all possibilities i.e. the following cartesian product:
    # Re[interpolation, extrapolation_left, extrapolation_right] x tau[interpolation, extrapolation_left, extrapolation_right]

    interval_types = ['interpolation', 'extrapolation_left', 'extrapolation_right']
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for k, Re_interval_type in enumerate(interval_types):
        for j, tau_interval_type in enumerate(interval_types):
            dataset_inter_re = prepare_Re_dataset(Re_range=getattr(Re_interval_split, Re_interval_type),
                                                tau_range=getattr(tau_interval_split, tau_interval_type), n_samples=10000)
            X, X_tau, Tau, Re = dataset_inter_re.X, dataset_inter_re.X_tau, dataset_inter_re.tau_values, dataset_inter_re.Re_values
            i = 0
            x = torch.tensor(X[i]).cuda().float()[None, None, :]
            x_tau = torch.tensor(X_tau[i]).cuda().float()[None, None, :]
            tau = torch.tensor(Tau[i]).cuda().float()[None, None]
            re = torch.tensor(Re[i]).cuda().float()[None, None]
            x_hat, x_hat_tau, _, _, _ = model(x, tau, re)
            ax = axs[k, j]

            ax.plot(x.cpu().squeeze().numpy(), label='x', linewidth = 3)
            ax.plot(x_tau.cpu().squeeze().numpy(), label='x_tau', linewidth = 3)
            ax.plot(x_hat.cpu().squeeze().detach().numpy(), label='x_hat', linewidth = 3)
            ax.plot(x_hat_tau.cpu().squeeze().detach().numpy(), label='x_hat_tau', linewidth = 3)
            ax.set_title(f'Tau: {tau.item()}, Re: {Re[i]}', fontsize=18)
            ax.legend()
    return fig, axs



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--n_samples_train', type=int, default=4_000)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1e-4)
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

