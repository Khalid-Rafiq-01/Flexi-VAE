#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn

import math
import torch


# In[2]:


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


# In[3]:


class Norm(nn.Module):
  def __init__(self, num_channels, num_groups=4):
    super(Norm, self).__init__()
    self.norm = nn.GroupNorm(num_groups, num_channels)

  def forward(self, x):
    return self.norm(x.permute(0,2,1)).permute(0,2,1)


# In[4]:

class Norm_new(nn.Module):
    def __init__(self, num_channels, num_groups=4):
        super(Norm_new, self).__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        if x.dim() == 2:
            # Reshape to (batch_size, num_channels, 1)
            x = x.unsqueeze(-1)
            x = self.norm(x)
            # Reshape back to (batch_size, num_channels)
            x = x.squeeze(-1)
        else:
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=2, feats=[512, 256, 128, 64, 32]):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self._net = nn.Sequential(
            nn.Linear(input_dim, feats[0]),
            nn.GELU(),
            Norm_new(feats[0]),
            nn.Linear(feats[0], feats[1]),
            nn.GELU(),
            Norm_new(feats[1]),
            nn.Linear(feats[1], feats[2]),
            nn.GELU(),
            Norm_new(feats[2]),
            nn.Linear(feats[2], feats[3]),
            nn.GELU(),
            Norm_new(feats[3]),
            nn.Linear(feats[3], feats[4]),
            nn.GELU(),
            Norm_new(feats[4]),
            nn.Linear(feats[4], 2 * latent_dim)
        )

    def forward(self, x):
      Z = self._net(x)
      mean, log_var = torch.split(Z, self.latent_dim, dim=-1)
      return mean, log_var


# In[5]:


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, feats=[32, 64, 128, 256, 512]):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self._net = nn.Sequential(
            nn.Linear(latent_dim, feats[0]),
            nn.GELU(),
            Norm_new(feats[0]),
            nn.Linear(feats[0], feats[1]),
            nn.GELU(),
            Norm_new(feats[1]),
            nn.Linear(feats[1], feats[2]),
            nn.GELU(),
            Norm_new(feats[2]),
            nn.Linear(feats[2], feats[3]),
            nn.GELU(),
            Norm_new(feats[3]),
            nn.Linear(feats[3], feats[4]),
            nn.GELU(),
            Norm_new(feats[4]),
            nn.Linear(feats[4], output_dim),
            nn.Tanh()
        )

    def forward(self, x):
      y = self._net(x)
      return y


# In[6]:


class Propagator(nn.Module): #taken in (z(t), tau) and outputs z(t+tau)  [2, 5, 10, 2]
  def __init__(self, latent_dim, feats=[16, 32], max_tau=10000, encoding_dim=64):
    
    """
    Input : (z(t), tau)
    Output: z(t+tau)
    """
    self.max_tau = max_tau
    super(Propagator, self).__init__()
    self.register_buffer('encodings',  positionalencoding1d(encoding_dim, max_tau))  # shape: max_tau, 64

    self.projector = nn.Sequential(
        nn.Linear(latent_dim, encoding_dim),
        nn.ReLU(),
        Norm(encoding_dim),
        nn.Linear(encoding_dim, encoding_dim),
    )

    self._net = nn.Sequential(
            nn.Linear(encoding_dim, feats[0]),
            nn.ReLU(),
            Norm(feats[0]),
            nn.Linear(feats[0], feats[1]),
            nn.ReLU(),
            Norm(feats[1]),
            nn.Linear(feats[1], latent_dim),
            )


  def forward(self, z, tau):
    zproj = self.projector(z)
    enc = self.encodings[tau.long()]
    # z: 2
    # enc: 64
    # [z1, z2, enc1, enc2, ..., enc64]
    z = zproj + enc

    z_tau = self._net(z)
    return z_tau


# Doing this for the embedding for Re
class Propagator_encoding(nn.Module): #taken in (z(t), tau) and outputs z(t+tau)  [2, 5, 10, 2]
  def __init__(self, latent_dim, feats=[16, 32], max_tau=10000, encoding_dim=64, max_re = 5000):
    
    """
    Input : (z(t), tau, re)
    Output: z(t+tau)
    """
    self.max_tau = max_tau
    self.max_re = max_re
    super(Propagator_encoding, self).__init__()
    self.register_buffer('tau_encodings',  positionalencoding1d(encoding_dim, max_tau))  # shape: max_tau, 64
    self.register_buffer('re_encodings',  positionalencoding1d(encoding_dim, max_re))  # shape: max_re, 64
    
    self.projector = nn.Sequential(
        nn.Linear(latent_dim, encoding_dim),
        nn.ReLU(),
        Norm(encoding_dim),
        nn.Linear(encoding_dim, encoding_dim),
    )

    self._net = nn.Sequential(
            nn.Linear(encoding_dim, feats[0]),
            nn.ReLU(),
            Norm(feats[0]),
            nn.Linear(feats[0], feats[1]),
            nn.ReLU(),
            Norm(feats[1]),
            nn.Linear(feats[1], latent_dim),
            )


  def forward(self, z, tau, re):
    zproj = self.projector(z)
    tau_enc = self.tau_encodings[tau.long()]
    re_enc = self.re_encodings[re.long()]
    # z: 2
    # enc: 64
    # [z1, z2, enc1, enc2, ..., enc64]
    z = zproj + tau_enc + re_enc
    #print("shape after enc addition: ", z.shape)
    z_tau = self._net(z)
    #print("shape z_tau: ", z_tau.shape)
    return z_tau



class Propagator_concat(nn.Module): #taken in (z(t), tau) and outputs z(t+tau)  [2, 5, 10, 2]
  def __init__(self, latent_dim, feats = [16, 32]):
    
    """
    Input : (z(t), tau, re)
    Output: z(t+tau)
    """
    super(Propagator_concat, self).__init__()
    
    self._net = nn.Sequential(
            nn.Linear(latent_dim + 2, feats[0]),
            nn.ReLU(),
            nn.Linear(feats[0], feats[1]),
            nn.ReLU(),
            nn.Linear(feats[1], latent_dim),
            )
    
  def forward(self, z, tau, re):
    zproj = z.squeeze(1)
    z_ = torch.cat((zproj, tau, re), dim = 1)   
    z_tau = self._net(z_)
    z_tau = z_tau[:, None, :]
    
    return z_tau


class Model(nn.Module):
    def __init__(self, encoder, decoder, propagator):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder # decoder for x(t)
        self.propagator = propagator  # used to time march z(t) to z(t+tau)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def forward(self, x, tau, re):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        # Update small fcnn to get z(t+tau) from z(t)
        z_tau = self.propagator(z, tau, re)

        # Reconstruction
        x_hat = self.decoder(z)  # Reconstruction of x(t)
        x_hat_tau = self.decoder(z_tau)

        return x_hat, x_hat_tau, mean, log_var, z_tau
    

# Define loss function
def loss_function(x, x_tau, x_hat, x_hat_tau, mean, log_var):
    reconstruction_loss1 = nn.MSELoss()(x, x_hat) 
    reconstruction_loss2 = nn.MSELoss()(x_tau, x_hat_tau)
    
    KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=2))
    return reconstruction_loss1, reconstruction_loss2, KLD


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params








