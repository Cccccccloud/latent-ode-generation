
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np


from latent_ode_rnn import TimeSeriesCellDataset, DynamicCellEvolver

def train():

    z_path = 'tabsyn/vae/ckpt/neftel_smartseq/train_z_sorted.npy'
    t_path = 'tabsyn/vae/ckpt/neftel_smartseq/train_t_sorted.npy'
    save_path = 'tabsyn/vae/ckpt/neftel_smartseq/ode_rnn_rk4_model.pt'
    
    seq_len = 10
    obs_len = 3
    pred_len = seq_len - obs_len + 1
    
    batch_size = 128
    epochs = 100
    learning_rate = 1e-3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = TimeSeriesCellDataset(z_path, t_path, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Total sequences: {len(dataset)}, Batches per epoch: {len(dataloader)}")

    model = DynamicCellEvolver(input_dim=8088, latent_dim=128, rnn_hidden=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_z, batch_t in pbar:
            batch_z = batch_z.to(device)
            batch_t = batch_t.to(device)
            
            optimizer.zero_grad()
            
            # forward propagation
            z_pred, logvar = model(batch_z, batch_t) 
            z_target = batch_z[:, obs_len-1:, :]

            z_pred_num = z_pred[:, :, :-4]
            z_target_num = z_target[:, :, :-4]
            
            z_pred_cat = z_pred[:, :, -4:]
            z_target_cat = z_target[:, :, -4:]
            
            # Loss of genetic characteristics
            recon_loss_num = criterion(z_pred_num, z_target_num)
            
            # Loss of category features
            recon_loss_cat = criterion(z_pred_cat, z_target_cat)
            
            recon_loss = recon_loss_num + 50.0 * recon_loss_cat
            
            # GP KL divergence
            gp_kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - logvar.exp(), dim=-1))
            
            loss = recon_loss + 0.01 * gp_kl_loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Total': f"{loss.item():.4f}", 'Num': f"{recon_loss_num.item():.4f}", 'Cat': f"{recon_loss_cat.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model with loss: {best_loss:.4f}")

    print("Training Complete.")

if __name__ == "__main__":
    train()