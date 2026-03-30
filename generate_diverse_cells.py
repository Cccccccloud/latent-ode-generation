import torch
import numpy as np
import json
import os
import gc

from latent_ode_rnn import DynamicCellEvolver
from tabsyn.vae.model import Decoder_model
from utils_train import preprocess

def generate_diverse():
    dataname = 'neftel_smartseq'
    data_dir = f'data/{dataname}' 
    ckpt_dir = f'tabsyn/vae/ckpt/{dataname}'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("1. Preparing Models...")
    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)
    _, _, categories, d_numerical = preprocess(data_dir, task_type=info['task_type'])
    
    decoder = Decoder_model(2, d_numerical, categories, 4, n_head=1, factor=32).to(device)
    decoder.load_state_dict(torch.load(f'{ckpt_dir}/decoder.pt', map_location=device))
    decoder.eval()
    
    ode_model = DynamicCellEvolver(input_dim=8088, latent_dim=128, rnn_hidden=256).to(device)
    ode_model.load_state_dict(torch.load(f'{ckpt_dir}/ode_rnn_rk4_model.pt', map_location=device))
    ode_model.eval()
    
    print("2. Generating DIVERSE Continuous Cell Trajectories...")
    z_real = np.load(f'{ckpt_dir}/train_z_sorted.npy') 
    
    obs_len = 3
    n_trajectories = 1000
    steps_per_traj = 10
    
    np.random.seed(42)
    start_indices = np.random.choice(len(z_real) - obs_len, n_trajectories, replace=False)
    
    z_fake_list = list()
    with torch.no_grad():
        from torchdiffeq import odeint
        for idx in start_indices:
            z_seed = torch.tensor(z_real[idx : idx+obs_len].reshape(1, obs_len, -1), dtype=torch.float32).to(device)
            
            z_latent = ode_model.projector(z_seed)
            _, h_n = ode_model.rnn(z_latent)
            z0 = ode_model.hidden_to_z0(h_n[-1]) 
            
            t_points = torch.linspace(0, 0.2, steps=steps_per_traj).to(device)
            z_pred_latent_mean = odeint(ode_model.ode_func, z0, t_points, method='rk4').permute(1, 0, 2)

            logvar = ode_model.gp_logvar_net(z_pred_latent_mean)
            std = torch.exp(0.5 * logvar)
        
            temperature = 0.5 
            z_sample = z_pred_latent_mean + torch.randn_like(std) * (std * temperature)
            
            z_generated_flat = ode_model.reconstructor(z_sample) 
            z_fake_list.append(z_generated_flat)
            
        z_generated_flat_all = torch.cat(z_fake_list, dim=0).squeeze(1)
        z_fake_3d = z_generated_flat_all.view(n_trajectories * steps_per_traj, -1, 4) 
        
    print(f"Generated Diverse Latent Z shape: {z_fake_3d.shape}")
    
    print("3. Decoding Latent Space (Micro-Batching)...")
    n_samples = z_fake_3d.shape[0]
    batch_size = 10
    x_fake_num_list = list()
    x_fake_cat_list = list()
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i+batch_size, n_samples)
            z_batch = z_fake_3d[i:end_idx]
            batch_num, batch_cat = decoder(z_batch)
            x_fake_num_list.append(batch_num.cpu().numpy())
            
            batch_cat_processed = list()
            if batch_cat is not None:
                for cat_tensor in batch_cat:
                    if cat_tensor is not None:

                        probs = torch.softmax(cat_tensor, dim=-1)
                        flat_probs = probs.view(-1, probs.shape[-1])
                        cat_classes = torch.multinomial(flat_probs,1).view(*cat_tensor.shape[:-1]).cpu().numpy()
                        
                        batch_cat_processed.append(cat_classes)
            
            if len(batch_cat_processed) > 0:
                batch_cat_stacked = np.stack(batch_cat_processed, axis=1)
                x_fake_cat_list.append(batch_cat_stacked)
                
            del z_batch, batch_num, batch_cat, batch_cat_processed
            gc.collect()

    x_fake_num = np.concatenate(x_fake_num_list, axis=0)
    
    if len(x_fake_cat_list) > 0:
        x_fake_cat = np.concatenate(x_fake_cat_list, axis=0)
    else:
        x_fake_cat = np.zeros((n_samples, 0))
            
    print("4. Saving DIVERSE generated data to disk...")
    save_dir = f'tabsyn/vae/ckpt/{dataname}'
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(f'{save_dir}/generated_fake_num.npy', x_fake_num)
    np.save(f'{save_dir}/generated_fake_cat.npy', x_fake_cat)
    
    print(f"Numerical Data Saved: {x_fake_num.shape}")
    print(f"Categorical Data Saved: {x_fake_cat.shape}")
    print("Data generation completed.")

if __name__ == "__main__":
    generate_diverse()