
# Dynamic Single-Cell Trajectory Generation:Augmenting TabSyn VAE with Latent ODE-RNN and Gaussian Processes

This repository contains the official implementation of our Latent ODE RNN model for single-cell/tabular data generation, along with a comprehensive comparison against the [TabSyn](https://github.com/amazon-science/tabsyn) baseline.

## Repository Structure

Our customized code files are integrated with the TabSyn framework:

* **`latent_ode_rnn.py`**: Core architecture of our Latent ODE RNN model.
* **`train_dynamics.py`**: Script to train our dynamics model.
* **`generate_diverse_cells.py`**: Generates synthetic cell data.
* **`to_csv.py`**: Synthesizes and converts final output to CSV.
* *Note: All other files in this repository are based on the original TabSyn framework.*

---

## Step-by-Step Guide

### 1. Data Preprocessing
First, run the data preprocessing scripts to prepare the dataset. The processed data will be saved in the `data/` directory.

```bash
python data_preprocessing.py
python process_dataset.py
```

### 2. Baseline: TabSyn Pipeline 
Train and generate data using the TabSyn baseline. The generated data will be named `TabSyn_Generated_Cells.csv`.

```bash
# Step 2.1: Train the VAE first
python main.py --dataname neftel_smartseq --method vae --mode train

# Step 2.2: Train the diffusion model (after VAE is trained)
python main.py --dataname neftel_smartseq --method tabsyn --mode train

# Step 2.3: Generate synthetic data
python main.py --dataname neftel_smartseq --method tabsyn --mode sample --save_path [PATH_TO_SAVE]/TabSyn_Generated_Cells.csv
```

### 3. Our Model: Latent ODE Pipeline
Train our custom Latent ODE RNN model and generate the corresponding data. The final output will be `ODE_Generated_Cells_Fixed.csv`.

```bash
# Step 3.1: Train our Model
python train_dynamics.py

# Step 3.2: Generate data
python generate_diverse_cells.py

# Step 3.3: Combine data and export to CSV
python to_csv.py
```

### 4. Evaluation & Comparison 
We provide several evaluation scripts to compare the performance of TabSyn and our ODE-based model. 
*(Note: Replace `--path` with the path to either `ODE_Generated_Cells_Fixed.csv` or `TabSyn_Generated_Cells.csv` to evaluate the respective models.)*

```bash
# 1. Density Evaluation
python eval/eval_density.py --dataname neftel_smartseq --model tabsyn --path eval/ODE_Generated_Cells_0.5.csv

# 2. Dynamics Evaluation
python eval/eval_dynamics.py

# 3. Advanced Dynamics Evaluation
python eval/eval_advanced_dynamics.py 

# 4. Downstream Task Evaluation
python eval/eval_downstream_task.py

# 5. Sensitivity Analysis
python eval/eval_temperature.py
```

---

## 📖 References

If you find this repository useful, please consider citing the original TabSyn paper:

```bibtex
@inproceedings{tabsyn2024,
  title={Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space},
  author={Zhang, Hengrui and Zhang, Jiani and Srinivasan, Balasubramaniam and Shen, Zhengyuan and Qin, Xiao and Faloutsos, Christos and Rangwala, Huzefa and Karypis, George},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```


