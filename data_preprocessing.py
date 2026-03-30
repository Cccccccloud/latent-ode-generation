import pandas as pd
import scanpy as sc
import scipy.io
import os

meta_df = pd.read_csv('Cells2.csv')
genes = pd.read_csv('Genes.txt', header=None, names=['gene_name'])
adata = sc.read_mtx('Exp_data_TPM.mtx')

if adata.n_vars == len(meta_df) and adata.n_obs == len(genes):
    adata = adata.T
    print("Genes x Cells -> Cells x Genes")

adata.obs_names = meta_df['cell_name'].values
adata.var_names = genes['gene_name'].values

adata.obs = meta_df.set_index('cell_name')

print(f"Read completed! There are currently {adata.n_obs} cells and {adata.n_vars} genes.")

print("2. Extracting hypervariable genes (HVGs)...")
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable].copy()

print(f"Gene screening completed! {adata.n_vars} core genes retained.")

print("3. Merging data...")

gene_df = pd.DataFrame(
    adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
    columns=adata.var_names,
    index=adata.obs_names
)

final_meta_df = adata.obs.copy()

cols_to_drop =[]

for col in final_meta_df.columns:
    if final_meta_df[col].nunique() == len(final_meta_df) and final_meta_df[col].dtype == 'object':
        cols_to_drop.append(col)

if cols_to_drop:
    final_meta_df = final_meta_df.drop(columns=cols_to_drop)
    print(f"Unused ID columns have been automatically deleted: {cols_to_drop}")

# 水平拼接 Metadata 和 基因矩阵
final_csv = pd.concat([final_meta_df, gene_df], axis=1)

# 创建 TabSyn 格式的目录并保存
os.makedirs('./data/neftel_smartseq', exist_ok=True)
save_path = './data/neftel_smartseq/neftel_smartseq.csv'
final_csv.to_csv(save_path, index=False)

print(f"🎉 成功！最终包含 {final_csv.shape[1]} 列的 CSV 文件已保存至: {save_path}")