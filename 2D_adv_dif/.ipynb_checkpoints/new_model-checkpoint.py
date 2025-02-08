#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn

import math
import torch

import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from dataclasses import dataclass, asdict
import json
from torch.utils.data import DataLoader


# Normalization Layer for Conv2D
class Norm(nn.Module):
    def __init__(self, num_channels, num_groups=4):
        super(Norm, self).__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        return self.norm(x)

# Encoder using Conv2D
class Encoder(nn.Module):
    def __init__(self, latent_dim=3):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            # Input: (batch_size, 1, 256, 256)
            nn.Conv2d(1, 32, kernel_size=2, stride=2, padding=0),  # (batch_size, 64, 128, 128)
            nn.GELU(),
            Norm(32),
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),  # (batch_size, 128, 64, 64)
            nn.GELU(),
            Norm(64),
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0),  # (batch_size, 256, 32, 32)
            nn.GELU(),
            Norm(128),
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),  # (batch_size, 512, 16, 16)
            nn.GELU(),
            Norm(256),
            nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0),  # (batch_size, 512, 8, 8)
            nn.GELU(),
            Norm(512),
        )
        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mean, log_var
        
        
        
class Decoder(nn.Module):
    def __init__(self, latent_dim=3):
        super(Decoder, self).__init__()
        # Fully connected layer to transform the latent vector back to the shape (batch_size, 512, 8, 8)
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        self.deconv_layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.GELU(),
            Norm(256),


            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.GELU(),
            Norm(128),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.GELU(),
            Norm(64),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.GELU(),
            Norm(32),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, z):
        # Transform the latent vector to match the shape of the feature maps
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)  # Reshape to (batch_size, 512, 4, 4)
        x = self.deconv_layers(x)
        return x
        
        
class Propagator_concat(nn.Module): 
    """
    Takes in (z(t), tau, alpha) and outputs z(t+tau)
    """
    def __init__(self, latent_dim, feats=[16, 32, 64]):
        """
        Initialize the propagator network.
        Input : (z(t), tau)
        Output: z(t+tau)
        """
        super(Propagator_concat, self).__init__()

        self._net = nn.Sequential(
            nn.Linear(latent_dim + 2, feats[0]),  # 1 is for tau; more params will increase this
            nn.GELU(),
            nn.Linear(feats[0], feats[1]),
            nn.GELU(),
            nn.Linear(feats[1], feats[2]),
            nn.GELU(),
            nn.Linear(feats[2], latent_dim),
        )

    def forward(self, z, tau, alpha):
        """
        Forward pass of the propagator.
        Concatenates latent vector z with tau and processes through the network.
        """
        zproj = z.squeeze(1)  # Adjust z dimensions if necessary
        z_ = torch.cat((zproj, tau, alpha), dim=1)  # Concatenate z and tau along the last dimension
        z_tau = self._net(z_)
        return z_tau, z_
        
        
        
        
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

    def forward(self, x, tau, alpha):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        # Update small fcnn to get z(t+tau) from z(t)
        z_tau, z_ = self.propagator(z, tau, alpha)

        # Reconstruction
        x_hat = self.decoder(z)  # Reconstruction of x(t)
        x_hat_tau = self.decoder(z_tau)

        return x_hat, x_hat_tau, mean, log_var, z_tau, z_
        
def loss_function(x, x_tau, x_hat, x_hat_tau, mean, log_var):
    """
    Compute the VAE loss components.
    :param x: Original input
    :param x_tau: Future input (ground truth)
    :param x_hat: Reconstructed x(t)
    :param x_hat_tau: Predicted x(t+tau)
    :param mean: Mean of the latent distribution
    :param log_var: Log variance of the latent distribution
    :return: reconstruction_loss1, reconstruction_loss2, KLD
    """
    reconstruction_loss1 = nn.MSELoss()(x, x_hat)  # Reconstruction loss for x(t)
    reconstruction_loss2 = nn.MSELoss()(x_tau, x_hat_tau)  # Prediction loss for x(t+tau)
    
    # Kullback-Leibler Divergence
    KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1))  # Updated dim
    
    return reconstruction_loss1, reconstruction_loss2, KLD
