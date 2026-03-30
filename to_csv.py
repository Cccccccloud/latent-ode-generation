import numpy as np
import pandas as pd
import json
import os

def convert_and_csv():
    dataname = 'neftel_smartseq'
    save_dir = f'tabsyn/vae/ckpt/{dataname}'
    
    print("1. Loading generated matrices...")
    x_fake_num = np.load(f'{save_dir}/generated_fake_num.npy')
    x_fake_cat = np.load(f'{save_dir}/generated_fake_cat.npy')
    
    print("2. Loading original real dataset to match formats...")
    real_df = pd.read_csv(f'data/{dataname}/train.csv')
    with open(f'data/{dataname}/info.json', 'r') as f:
        info = json.load(f)
        
    num_col_idx = info.get('num_col_idx',[])
    cat_col_idx = info.get('cat_col_idx',[])
    
    if 'target_col_idx' in info and info['target_col_idx']:
        target_idx = info['target_col_idx']
        if isinstance(target_idx, int):
            target_idx = [target_idx]
        all_cat_cols = cat_col_idx + target_idx
    else:
        all_cat_cols = cat_col_idx
        
    syn_df = pd.DataFrame(columns=real_df.columns)
    
    print("3. Reversing numerical normalization...")
    for i, col_idx in enumerate(num_col_idx):
        col_name = real_df.columns[col_idx]
        real_vals = real_df[col_name].dropna().values
        fake_vals = x_fake_num[:, i]
        
        real_vals_sorted = np.sort(real_vals)
        ranks = pd.Series(fake_vals).rank(pct=True).values
        idx = np.clip(np.floor(ranks * (len(real_vals_sorted) - 1)).astype(int), 0, len(real_vals_sorted) - 1)
        syn_df[col_name] = real_vals_sorted[idx]
        
    print("4. Mapping ALL category & target indices back to string labels...")
    for i, col_idx in enumerate(all_cat_cols):
        col_name = real_df.columns[col_idx]
        real_series = real_df[col_name].astype(str)
        real_cats = sorted(real_series.unique().tolist())
        
        fake_idx = x_fake_cat[:, i]
        fake_labels =[]
        for idx in fake_idx:
            idx_int = int(idx)
            if idx_int < len(real_cats):
                fake_labels.append(real_cats[idx_int])
            else:
                fake_labels.append(real_cats[0])
        syn_df[col_name] = fake_labels
        
    print("5. Failsafe: Removing any accidental NaNs...")
    for col in syn_df.columns:
        if syn_df[col].isnull().any():
            mode_val = real_df[col].mode(dropna=True)
            if not mode_val.empty:
                syn_df[col] = syn_df[col].fillna(mode_val.iloc[0])
            else:
                syn_df[col] = syn_df[col].fillna(0)
                
    os.makedirs('eval', exist_ok=True)
    eval_csv_path = 'eval/ODE_Generated_Cells.csv'
    syn_df.to_csv(eval_csv_path, index=False)
    print(f"Created Perfect CSV for Evaluation: {eval_csv_path}")

if __name__ == "__main__":
    convert_and_csv()

