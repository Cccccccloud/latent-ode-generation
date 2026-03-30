
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel
import umap
import warnings
warnings.filterwarnings('ignore') 


def eval_gene_trend(real_3d, ode_3d, tabsyn_3d, gene_idx, gene_name="Marker Gene"):
    """Calculate the population average evolutionary trend error of a specific core gene over time (MAE)"""
    T = real_3d.shape[1]
    time_steps = np.arange(T)
    
    real_trend = np.mean(real_3d[:, :, gene_idx], axis=0)
    ode_trend = np.mean(ode_3d[:, :, gene_idx], axis=0)
    tabsyn_trend = np.mean(tabsyn_3d[:, :, gene_idx], axis=0)
    
    ode_mae = np.mean(np.abs(real_trend - ode_trend))
    tabsyn_mae = np.mean(np.abs(real_trend - tabsyn_trend))
    
    plt.figure(figsize=(8, 5))
    plt.plot(time_steps, real_trend, label='Real Data', marker='o', linewidth=2, color='black')
    plt.plot(time_steps, ode_trend, label=f'Latent ODE (MAE: {ode_mae:.2f})', marker='^', linewidth=2, color='green')
    plt.plot(time_steps, tabsyn_trend, label=f'TabSyn (MAE: {tabsyn_mae:.2f})', marker='x', linewidth=2, color='red', linestyle='--')
    
    plt.title(f"{gene_name} Expression Trend over Pseudotime")
    plt.xlabel("Pseudotime Steps (t)")
    plt.ylabel("Mean Expression Level")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Gene_Trend_{gene_name}.png", dpi=300)
    plt.close()
    
    return ode_mae, tabsyn_mae


def eval_interpolation_mse(real_3d, ode_3d, tabsyn_3d, start_step=4, end_step=7):
    """Calculate the mean square error of predictions for specific intermediate developmental stages"""
    real_mid = real_3d[:, start_step:end_step, :]
    ode_mid = ode_3d[:, start_step:end_step, :]
    tabsyn_mid = tabsyn_3d[:, start_step:end_step, :]
    
    ode_mse = np.mean((real_mid - ode_mid) ** 2)
    tabsyn_mse = np.mean((real_mid - tabsyn_mid) ** 2)
    
    return ode_mse, tabsyn_mse


def calc_mmd(x, y, gamma=1.0):
    """Calculate Maximum Mean Discrepancy (MMD)"""
    xx = rbf_kernel(x, x, gamma)
    yy = rbf_kernel(y, y, gamma)
    xy = rbf_kernel(x, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

def eval_static_divergence(real_3d, ode_3d, tabsyn_3d):
    """Calculate Wasserstein Distance MMD"""
    real_flat = real_3d.reshape(-1, real_3d.shape[-1])
    ode_flat = ode_3d.reshape(-1, ode_3d.shape[-1])
    tabsyn_flat = tabsyn_3d.reshape(-1, tabsyn_3d.shape[-1])
    
    # Wasserstein Distance
    ode_wd = np.mean([wasserstein_distance(real_flat[:, i], ode_flat[:, i]) for i in range(real_flat.shape[1])])
    tabsyn_wd = np.mean([wasserstein_distance(real_flat[:, i], tabsyn_flat[:, i]) for i in range(real_flat.shape[1])])
    
    # MMD
    idx = np.random.choice(real_flat.shape[0], min(2000, real_flat.shape[0]), replace=False)
    ode_mmd = calc_mmd(real_flat[idx], ode_flat[idx])
    tabsyn_mmd = calc_mmd(real_flat[idx], tabsyn_flat[idx])
    
    return (ode_wd, ode_mmd), (tabsyn_wd, tabsyn_mmd)

def plot_state_coherence_umap(real_3d, ode_3d, tabsyn_3d):
    B, T, F = real_3d.shape
    
    real_flat = real_3d.reshape(-1, F)
    ode_flat = ode_3d.reshape(-1, F)
    tabsyn_flat = tabsyn_3d.reshape(-1, F)
    
    time_labels = np.tile(np.arange(T), B)
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    
    real_emb = reducer.fit_transform(real_flat)
    ode_emb = reducer.transform(ode_flat)
    tabsyn_emb = reducer.transform(tabsyn_flat)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cm = 'Spectral_r' 
    
    sc1 = axes[0].scatter(real_emb[:, 0], real_emb[:, 1], c=time_labels, cmap=cm, s=5, alpha=0.7)
    axes[0].set_title("Real Data UMAP\n(Smooth Pseudotime Gradient)")
    
    axes[1].scatter(ode_emb[:, 0], ode_emb[:, 1], c=time_labels, cmap=cm, s=5, alpha=0.7)
    axes[1].set_title("Latent ODE UMAP\n(Coherent Continuous Flow)")
    
    axes[2].scatter(tabsyn_emb[:, 0], tabsyn_emb[:, 1], c=time_labels, cmap=cm, s=5, alpha=0.7)
    axes[2].set_title("TabSyn UMAP\n(Disjointed Temporal States)")
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        
    cbar = fig.colorbar(sc1, ax=axes.ravel().tolist(), label='Pseudotime Step (t)', shrink=0.8)
    plt.savefig("UMAP_Pseudotime_Comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    
    
    df_real = pd.read_csv("data/neftel_smartseq/neftel_smartseq.csv")
    df_ode = pd.read_csv("eval/ODE_Generated_Cells_0.5.csv")
    df_tabsyn = pd.read_csv("eval/gnn-tabsyn-result2.csv")
    
    df_real_num = df_real.select_dtypes(include=[np.number])
    F = df_real_num.shape[1]
    print(f"The number of purely numerical genes actually involved in the calculation is F={F}")
    
    real_data_2d = df_real_num.values.astype(np.float32)
    ode_data_2d = df_ode.select_dtypes(include=[np.number]).iloc[:, :F].values.astype(np.float32)
    tabsyn_data_2d = df_tabsyn.select_dtypes(include=[np.number]).iloc[:, :F].values.astype(np.float32)
    
    T = 10  
    
    B_real = real_data_2d.shape[0] // T
    B_ode = ode_data_2d.shape[0] // T
    B_tabsyn = tabsyn_data_2d.shape[0] // T
    
    print(f"Number of detected trajectories: Real B={B_real}, ODE B={B_ode}, TabSyn B={B_tabsyn}")
    
    real_3d = real_data_2d[:B_real * T, :].reshape(B_real, T, F)
    ode_3d = ode_data_2d[:B_ode * T, :].reshape(B_ode, T, F)
    tabsyn_3d = tabsyn_data_2d[:B_tabsyn * T, :].reshape(B_tabsyn, T, F)
    
    min_B = min(B_real, B_ode, B_tabsyn)
    if min_B < max(B_real, B_ode, B_tabsyn):
        print(f" The number of tracks is inconsistent, align (truncate) to the minimum number of tracks: B={min_B}")
        real_3d = real_3d[:min_B, :, :]
        ode_3d = ode_3d[:min_B, :, :]
        tabsyn_3d = tabsyn_3d[:min_B, :, :]
    

    print("\n[1/4] Calculating evolutionary trends for specific genes...")
    
    real_time_trends = np.mean(real_3d, axis=0)
    trend_variances = np.var(real_time_trends, axis=0) 
    
    top_2_idx = np.argsort(trend_variances)[-2:]
    dynamic_gene_1, dynamic_gene_2 = top_2_idx[1], top_2_idx[0]
    
    o_mae_1, t_mae_1 = eval_gene_trend(real_3d, ode_3d, tabsyn_3d, gene_idx=dynamic_gene_1, gene_name=f"Dynamic_Gene_{dynamic_gene_1}")
    o_mae_2, t_mae_2 = eval_gene_trend(real_3d, ode_3d, tabsyn_3d, gene_idx=dynamic_gene_2, gene_name=f"Dynamic_Gene_{dynamic_gene_2}")
    
    print("\n[2/4] Calculating the trajectory intermediate state prediction error...")
    o_mse, t_mse = eval_interpolation_mse(real_3d, ode_3d, tabsyn_3d, start_step=4, end_step=7)
    
    print("\n[3/4] Calculating the global static distribution error...")
    (o_wd, o_mmd), (t_wd, t_mmd) = eval_static_divergence(real_3d, ode_3d, tabsyn_3d)
    
    print("\n[4/4] Plot UMAP...")
    plot_state_coherence_umap(real_3d, ode_3d, tabsyn_3d)
    
    print("\n" + "="*80)
    print(f"{'Metric':<35} | {'Latent ODE (Ours)':<20} | {'TabSyn (Baseline)':<20}")
    print("-" * 80)
    print(f"{'Gene Trend (Gene_1) MAE (↓)':<35} | {o_mae_1:<20.4f} | {t_mae_1:<20.4f}")
    print(f"{'Gene Trend (Gene_2) MAE (↓)':<35} | {o_mae_2:<20.4f} | {t_mae_2:<20.4f}")
    print(f"{'Mid-Trajectory MSE (↓)':<35} | {o_mse:<20.4f} | {t_mse:<20.4f}")
    print(f"{'Static Wasserstein Dist (↓)':<35} | {o_wd:<20.4f} | {t_wd:<20.4f}")
    print(f"{'Static MMD (↓)':<35} | {o_mmd:<20.4f} | {t_mmd:<20.4f}")
    print("="*80)
