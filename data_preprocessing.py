import pandas as pd
import scanpy as sc
import scipy.io
import os
import json

print("1. Loading dataset...")
meta_df = pd.read_csv('Cells2.csv')
genes = pd.read_csv('Genes.txt', header=None, names=['gene_name'])
adata = sc.read_mtx('Exp_data_TPM.mtx')

# Transpose matrix if necessary
if adata.n_vars == len(meta_df) and adata.n_obs == len(genes):
    adata = adata.T
    print("   Transposed matrix: Genes x Cells -> Cells x Genes")

adata.obs_names = meta_df['cell_name'].values
adata.var_names = genes['gene_name'].values

adata.obs = meta_df.set_index('cell_name')

print(f"   Read completed! There are currently {adata.n_obs} cells and {adata.n_vars} genes.")

print("2. Extracting hypervariable genes (HVGs)...")
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable].copy()

print(f"   Gene screening completed! {adata.n_vars} core genes retained.")

print("3. Merging data...")
gene_df = pd.DataFrame(
    adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
    columns=adata.var_names,
    index=adata.obs_names
)

final_meta_df = adata.obs.copy()

cols_to_drop = []

# Identify unique ID columns to drop
for col in final_meta_df.columns:
    if final_meta_df[col].nunique() == len(final_meta_df) and final_meta_df[col].dtype == 'object':
        cols_to_drop.append(col)

if cols_to_drop:
    final_meta_df = final_meta_df.drop(columns=cols_to_drop)
    print(f"   Unused ID columns automatically deleted: {cols_to_drop}")

# Horizontally concatenate Metadata and Gene Matrix
final_csv = pd.concat([final_meta_df, gene_df], axis=1)

print("4. Creating TabSyn data directories and saving CSV...")
dataset_name = "neftel_smartseq"

# Create TabSyn formatted directory and save CSV
os.makedirs(f'./data/{dataset_name}', exist_ok=True)
save_path = f'./data/{dataset_name}/{dataset_name}.csv'
final_csv.to_csv(save_path, index=False)

print(f"   Success! Final CSV with {final_csv.shape[1]} columns saved to: {save_path}")

print("5. Generating TabSyn JSON configuration...")
num_col_idx = []
cat_col_idx = []

# Automatically classify columns into numerical and categorical
for i, col in enumerate(final_csv.columns):
    if pd.api.types.is_numeric_dtype(final_csv[col]):
        num_col_idx.append(i)
    else:
        cat_col_idx.append(i)

# Build the info dictionary as required by TabSyn
info_dict = {
    "name": dataset_name,
    "task_type": "binclass",  # Keep default for general generation
    "header": "infer",
    "column_names": None,
    "num_col_idx": num_col_idx,
    "cat_col_idx": cat_col_idx,
    "target_col_idx": [],     # Leave empty for unconditional generation
    "file_type": "csv",
    "data_path": f"data/{dataset_name}/{dataset_name}.csv",
    "test_path": None
}

# Create Info directory and save the json file
os.makedirs('./Info', exist_ok=True)
json_path = f'./Info/{dataset_name}.json'

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(info_dict, f, indent=4)

print(f"   Configuration successfully written to: {json_path}")
print(f"   - Number of numerical columns: {len(num_col_idx)}")
print(f"   - Number of categorical columns: {len(cat_col_idx)}")
