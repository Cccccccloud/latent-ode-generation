import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchdiffeq import odeint


class TimeSeriesCellDataset(Dataset):
    def __init__(self, z_path, t_path, seq_len=10):
        z_data = np.load(z_path)  
        t_data = np.load(t_path)  
        
        self.N = z_data.shape[0]
        self.z_flat = z_data.reshape(self.N, -1) 
        self.t_data = t_data
        self.seq_len = seq_len

    def __len__(self):
        return self.N - self.seq_len

    def __getitem__(self, idx):
        z_seq = self.z_flat[idx : idx + self.seq_len]
        t_seq = self.t_data[idx : idx + self.seq_len]
        return torch.tensor(z_seq, dtype=torch.float32), torch.tensor(t_seq, dtype=torch.float32)


class ODEFunc(nn.Module):
    def __init__(self, latent_dim=128):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Tanh(),
            nn.Linear(256, latent_dim)
        )
        
    def forward(self, t, z):
        return self.net(z)

# Projector + RNN + ODE + GP
class DynamicCellEvolver(nn.Module):
    def __init__(self, input_dim=8088, latent_dim=128, rnn_hidden=256):
        super(DynamicCellEvolver, self).__init__()
        
        self.projector = nn.Linear(input_dim, latent_dim)
        
        self.rnn = nn.GRU(latent_dim, rnn_hidden, batch_first=True)
        self.hidden_to_z0 = nn.Linear(rnn_hidden, latent_dim)
        
        self.ode_func = ODEFunc(latent_dim)
        self.reconstructor = nn.Linear(latent_dim, input_dim)
        
        self.gp_logvar_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, z_seq, t_seq, obs_len=3, temperature=1.0):
 
        z_latent = self.projector(z_seq)
        z_obs = z_latent[:, :obs_len, :]
        _, h_n = self.rnn(z_obs)
        z0 = self.hidden_to_z0(h_n[-1])
        
        pred_len = z_seq.size(1) - obs_len + 1
        t_points = torch.linspace(0.0, 1.0, steps=pred_len).to(z_seq.device)
        
        z_pred_latent_mean = odeint(self.ode_func, z0, t_points, method='rk4')
        z_pred_latent_mean = z_pred_latent_mean.permute(1, 0, 2) 
        
        # Calculate the variance of the trajectory
        logvar = self.gp_logvar_net(z_pred_latent_mean)
        std = torch.exp(0.5 * logvar)
        
        # Temperature Scaling
        if self.training:
            eps = torch.randn_like(std)
            z_pred_latent_sample = z_pred_latent_mean + eps * std
        else:
            eps = torch.randn_like(std)
            z_pred_latent_sample = z_pred_latent_mean + eps * (std * temperature)
        
        z_pred = self.reconstructor(z_pred_latent_sample)
        
        return z_pred, logvar