import numpy as np
import pandas as pd
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def run_mle_evaluation():
    dataname = 'neftel_smartseq'
    print("1. Loading Data for Downstream ML Evaluation...")
    
    real_train = pd.read_csv(f'data/{dataname}/train.csv')
    real_test = pd.read_csv(f'data/{dataname}/test.csv')
    ode_train = pd.read_csv('eval/ODE_Generated_Cells_0.5.csv')
    
    tabsyn_path = 'eval/gnn-tabsyn-result2.csv'
    if os.path.exists(tabsyn_path):
        tabsyn_train = pd.read_csv(tabsyn_path)
    else:
        print("TabSyn Result not found, ODE data placeholder will be used...")
        tabsyn_train = ode_train.copy()

    with open(f'data/{dataname}/info.json', 'r') as f:
        info = json.load(f)

    target_idx = info['target_col_idx']
    if isinstance(target_idx, list): 
        target_idx = target_idx[0]
    target_col = real_train.columns[target_idx]

    cat_cols =[real_train.columns[i] for i in info.get('cat_col_idx',[])]
    num_cols = [real_train.columns[i] for i in info.get('num_col_idx',[])]

    if target_col in cat_cols: cat_cols.remove(target_col)
    if target_col in num_cols: num_cols.remove(target_col)

    print(f"Predicted Target: {target_col}")
    
    print("\n2. Safely Encoding Features and Labels...")
    for df in[real_train, real_test, ode_train, tabsyn_train]:
        for col in cat_cols + [target_col]:
            df[col] = df[col].astype(str).fillna('missing')
        for col in num_cols:
            df[col] = df[col].fillna(0)

    le = LabelEncoder()
    le.fit(pd.concat([real_train[target_col], real_test[target_col]]))

    y_real_train = le.transform(real_train[target_col])
    y_real_test = le.transform(real_test[target_col])
    y_ode_train = le.transform(ode_train[target_col])
    y_tabsyn_train = le.transform(tabsyn_train[target_col])

    if cat_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        oe.fit(pd.concat([real_train[cat_cols], real_test[cat_cols]]))
        def process_x(df):
            return np.hstack([oe.transform(df[cat_cols]), df[num_cols].values])
    else:
        def process_x(df):
            return df[num_cols].values

    X_real_train = process_x(real_train)
    X_real_test = process_x(real_test)
    X_ode_train = process_x(ode_train)
    X_tabsyn_train = process_x(tabsyn_train)

    def evaluate(X_train, y_train, name):
        print(f"  Training Random Forest on {name}...")
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_real_test)
        
        acc = accuracy_score(y_real_test, preds)
        f1 = f1_score(y_real_test, preds, average='macro')
        return acc, f1

    print("\n3. Running Downstream Classification (MLE)")
    
    # 真实数据上限 (Upper Bound)
    real_acc, real_f1 = evaluate(X_real_train, y_real_train, "Real Train Data (Upper Bound)")
    # TabSyn
    tab_acc, tab_f1 = evaluate(X_tabsyn_train, y_tabsyn_train, "TabSyn Generated Data")
    # ODE 模型
    ode_acc, ode_f1 = evaluate(X_ode_train, y_ode_train, "Your ODE Generated Data")

    print("\n" + "="*55)
    print(f" Downstream cell type classification tasks (Test Set Results)")
    print("="*55)
    
    df_res = pd.DataFrame({
        'Model Source':['Real Data (Upper Bound)', 'TabSyn (Baseline)', 'Your ODE (Ours)'],
        'Macro F1-Score':[real_f1, tab_f1, ode_f1],
        'Accuracy':[real_acc, tab_acc, ode_acc]
    })
    
    print(df_res.to_string(index=False))
    print("="*55)

if __name__ == "__main__":
    run_mle_evaluation()