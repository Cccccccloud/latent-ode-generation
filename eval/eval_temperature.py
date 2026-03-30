import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


DATANAME = 'neftel_smartseq'
TEMP_FILES = {
    0.5: 'eval/ODE_Generated_Cells_0.5.csv',
    1.0: 'eval/ODE_Generated_Cells_1.0.csv',
    2.0: 'eval/ODE_Generated_Cells_2.0.csv'
}

# Core evaluation functions
def eval_interpolation_mse(real_3d, pred_3d, start_step=4, end_step=7):
    min_b = min(real_3d.shape[0], pred_3d.shape[0])
    return np.mean((real_3d[:min_b, start_step:end_step, :] - pred_3d[:min_b, start_step:end_step, :]) ** 2)

def load_and_reshape(file_path, num_features, seq_len=10, max_rows=2000):
    df = pd.read_csv(file_path, nrows=max_rows)
    data_2d = df.select_dtypes(include=[np.number]).iloc[:, :num_features].values.astype(np.float32)
    n_traj = data_2d.shape[0] // seq_len
    return data_2d[:n_traj * seq_len, :].reshape(n_traj, seq_len, num_features)

def run_temperature_analysis():
    print("Automated analysis of GP layer temperature sensitivity...")

    real_train = pd.read_csv(f'data/{DATANAME}/train.csv')
    real_test = pd.read_csv(f'data/{DATANAME}/test.csv')
    
    with open(f'data/{DATANAME}/info.json', 'r') as f:
        info = json.load(f)
        
    target_idx = info['target_col_idx']
    target_idx = target_idx[0] if isinstance(target_idx, list) else target_idx
    target_col = real_train.columns[target_idx]
    
    cat_cols =[real_train.columns[i] for i in info.get('cat_col_idx',[]) if real_train.columns[i] != target_col]
    num_cols = [real_train.columns[i] for i in info.get('num_col_idx',[]) if real_train.columns[i] != target_col]

    le = LabelEncoder()
    le.fit(pd.concat([real_train[target_col], real_test[target_col].astype(str).fillna('missing')]))
    y_real_test = le.transform(real_test[target_col].astype(str).fillna('missing'))
    
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if cat_cols:
        oe.fit(pd.concat([real_train[cat_cols].astype(str).fillna('missing'), real_test[cat_cols].astype(str).fillna('missing')]))
    
    def get_X_y(df):
        for col in cat_cols + [target_col]: df[col] = df[col].astype(str).fillna('missing')
        for col in num_cols: df[col] = df[col].fillna(0)
        X = np.hstack([oe.transform(df[cat_cols]), df[num_cols].values]) if cat_cols else df[num_cols].values
        y = le.transform(df[target_col])
        return X, y

    X_real_test, _ = get_X_y(real_test)
    
    real_3d = np.load(f'data/{DATANAME}/X_num_train.npy')
    REAL_DIM = real_3d.shape[1]
    real_3d = load_and_reshape(f'data/{DATANAME}/train.csv', REAL_DIM)

    results_f1 = []
    results_mse = []
    temps = sorted(list(TEMP_FILES.keys()))
    
    for t in temps:
        path = TEMP_FILES[t]
        print(f"\n▶ Evaluating temperature T={t} (File: {path})")
        df_gen = pd.read_csv(path)
        
        # F1
        X_gen, y_gen = get_X_y(df_gen)
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf.fit(X_gen, y_gen)
        f1 = f1_score(y_real_test, clf.predict(X_real_test), average='macro')
        results_f1.append(f1)
        
        # MSE
        gen_3d = load_and_reshape(path, REAL_DIM)
        mse = eval_interpolation_mse(real_3d, gen_3d)
        results_mse.append(mse)
        
        print(f"  => Macro F1: {f1:.4f} | MSE: {mse:.4f}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    ax1.set_xlabel('GP Inference Temperature ($\mathcal{T}$)', fontsize=12)
    ax1.set_ylabel('Interpolation MSE (↓)', color='tab:blue', fontsize=12)
    line1 = ax1.plot(temps, results_mse, marker='o', color='tab:blue', linewidth=2, label='Interpolation MSE')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Downstream Macro F1-score (↑)', color='tab:red', fontsize=12)
    line2 = ax2.plot(temps, results_f1, marker='^', color='tab:red', linewidth=2, linestyle='--', label='Macro F1-score')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', bbox_to_anchor=(1.45, 0.5), fontsize=10)
    
    plt.title("Sensitivity Analysis of GP Temperature ($\mathcal{T}$)", fontsize=14)
    plt.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1.15, 1])
    plt.savefig("Real_Temperature_Sensitivity.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    run_temperature_analysis()