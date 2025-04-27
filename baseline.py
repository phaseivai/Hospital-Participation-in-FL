import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
# from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split  # for train/test split
from utils import apply_spline_transformation, apply_transformation_noh
from tqdm import tqdm
from utils import baseline_ettala, baseline_noh

'''This file runs the centralized training for all the data'''

def run_baseline_training(datasets, dataset_names, n_repeats = 100, seed=1234, baseline = 'ettala'):
    
    num_datasets = len(datasets)
    
    auc_matrices_baseline = [] # Stores [1 x dataset_names]

    # for reproducibility
    np.random.seed(seed)

        
    if baseline == 'ettala':
        baseline_function = baseline_ettala
        baseline_transformation = apply_spline_transformation
    elif baseline == 'noh':
        baseline_function = baseline_noh
        baseline_transformation = apply_transformation_noh
    else:
        raise ValueError(f"No baseline {baseline} found") 
    print(f"model according to {baseline}")
    
    datasets = baseline_transformation(datasets)
    
    # Initialize an 1 x m matrix for AUC values
    auc_matrix = np.zeros(num_datasets)
    
    for dataset_idx, dataset_name in enumerate(dataset_names):
        # train_df, test_df = train_test_split(datasets[dataset_name], test_size=0.2, stratify=datasets[dataset_name]['sig_cancer'], random_state=random_state)
        datasets[dataset_name]['pred'] = datasets[dataset_name].apply(baseline_function, axis=1)
        auc = roc_auc_score(datasets[dataset_name]['sig_cancer'], datasets[dataset_name]['pred'])
        
        auc_matrix[dataset_idx] = auc
        
    auc_matrices_baseline.append(auc_matrix)
    
        # Calculate the average AUC matrix across all repetitions
    final_auc_matrix_baseline = np.mean(auc_matrices_baseline, axis=0)
    
    return final_auc_matrix_baseline

def main():
    # Parameters           
    seed = 1234               # Fixed seed for reproducibility
    n_repeats = 1             # Number of monte carlo simulations
    
    # dataset from pickle
    with open('data/data.pkl', 'rb') as file:
        silos = pickle.load(file)
        
    client_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)
        
    final_auc_matrix_baseline = run_baseline_training(silos, client_names,
            n_repeats=n_repeats, seed=seed, baseline = 'noh')
    
     # After all experiments, print the AUC matrix
    print("\n### Final AUC Matrix ###")
    print(pd.DataFrame(np.array(final_auc_matrix_baseline).reshape(1,-1), columns=client_names))


if __name__ == "__main__":
    main()
