import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import os

def calc_temporal_smoothness(data_3d):
    """Trajectory smoothness (L2 distance between average neighboring time steps, smaller is smoother)"""
    step_diffs = data_3d[:, 1:, :] - data_3d[:, :-1, :] 
    distances = np.linalg.norm(step_diffs, axis=-1) 
    return np.mean(distances)

def calc_lag1_autocorrelation(data_3d):
    """Lag-1 autocorrelation (kinetic memory, closer to 1 the better))"""
    B, T, F = data_3d.shape
    correlations = []
    for b in range(B):
        traj = data_3d[b]
        for t in range(T - 1):
            x_t, x_next = traj[t], traj[t+1]
            corr = np.corrcoef(x_t, x_next)[0, 1] 
            if not np.isnan(corr):
                correlations.append(corr)
    return np.mean(correlations)

def calc_pseudotime_spearman(data_3d, pca_model):
    """Proposed temporal ordering fidelity (whether or not a biologically mature direction is learned, the closer to 1 the better)"""
    B, T, F = data_3d.shape
    time_steps = np.arange(T)
    spearman_scores = []
    for b in range(B):
        traj = data_3d[b]
        pc1 = pca_model.transform(traj)[:, 0] 
        corr, _ = spearmanr(time_steps, pc1)
        if not np.isnan(corr):
            spearman_scores.append(abs(corr)) 
    return np.mean(spearman_scores)


def load_and_reshape(file_path, num_features, seq_len=10, is_csv=True, max_rows=2000):
    """Automatically truncates and deforms 2D static tables into 3D trajectory structures."""
    print(f"Read file: {file_path}")
    if is_csv:
        df = pd.read_csv(file_path, nrows=max_rows)
        numeric_df = df.select_dtypes(include=[np.number])
        data_2d = numeric_df.iloc[:, :num_features].values.astype(np.float32)
    else:
        data_2d = np.load(file_path)[:max_rows, :num_features].astype(np.float32)
        
    n_traj = data_2d.shape[0] // seq_len
    data_2d = data_2d[:n_traj * seq_len, :]
    
    data_3d = data_2d.reshape(n_traj, seq_len, num_features)
    print(f"   => Successfully converted to 3D trajectory,")
    return data_3d


def run_dynamic_evaluation():
    print("\n" + "Dynamic Trajectory Evaluation...".center(80) + "\n")
    
    PATH_REAL = 'data/neftel_smartseq/X_num_train.npy'  
    PATH_TABSYN = 'eval/gnn-tabsyn-result2.csv'     
    PATH_ODE = 'eval/ODE_Generated_Cells_0.5.csv'     
    
    try:
        temp_real = np.load(PATH_REAL)
        REAL_DIM = temp_real.shape[1]
        print(f"The real gene value feature dimension was detected as: {REAL_DIM} ")
        
        real_data = load_and_reshape(PATH_REAL, num_features=REAL_DIM, is_csv=False)
        tabsyn_data = load_and_reshape(PATH_TABSYN, num_features=REAL_DIM, is_csv=True)
        ode_data = load_and_reshape(PATH_ODE, num_features=REAL_DIM, is_csv=True)
    except Exception as e:
        print(f"Error: {e}")
        return

    pca = PCA(n_components=1)
    pca.fit(real_data.reshape(-1, real_data.shape[-1]))

    models = {
        "Real Data (Oracle)": real_data,
        "TabSyn (Baseline)": tabsyn_data,
        "Latent ODE (Ours)": ode_data
    }
    
    results = {}
    print("\nCalculating dynamic indicators...\n")
    for name, data in models.items():
        smoothness = calc_temporal_smoothness(data)
        acf = calc_lag1_autocorrelation(data)
        spearman = calc_pseudotime_spearman(data, pca)
        results[name] = (smoothness, acf, spearman)
        
    print("="*85)
    print(f"{'Model / Metric':<25} | {'Smoothness (↓)':<15} | {'Lag-1 ACF (↑)':<15} | {'Spearman Order (↑)':<15}")
    print("-" * 85)
    
    for name, (smooth, acf, spear) in results.items():
        if "Ours" in name:
            print(f"\033[92m{name:<25} | {smooth:<15.4f} | {acf:<15.4f} | {spear:<15.4f}\033[0m")
        elif "Baseline" in name:
            print(f"\033[91m{name:<25} | {smooth:<15.4f} | {acf:<15.4f} | {spear:<15.4f}\033[0m")
        else:
            print(f"{name:<25} | {smooth:<15.4f} | {acf:<15.4f} | {spear:<15.4f}")
    
    print("="*85)

if __name__ == "__main__":
    run_dynamic_evaluation()