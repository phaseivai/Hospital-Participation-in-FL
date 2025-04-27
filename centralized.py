import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split  # for train/test split
from utils import apply_spline_transformation, apply_transformation_noh
from utils import baseline_ettala, baseline_noh
from tqdm import tqdm

'''This file runs the centralized training for all the data'''

def run_centralized_training(silos, dataset_names, n_repeats = 100, seed=1234, baseline='ettala'):
    
    num_datasets = len(silos)

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
    
    silos = baseline_transformation(silos)
    print(f"model according to {baseline}")
    
    # Initialize a list to hold all AUC matrices
    auc_matrices_cl = [] # Stores [1 x dataset_names]

    for repeat in tqdm(range(n_repeats)):
        # Generate a random state from a reproducible sequence
        random_state = np.random.randint(0, 10000000)

        test_features = {}
        test_labels = {}
        train_features = {}
        train_labels = {}
        
        # Initialize an 1 x m matrix for AUC values
        auc_matrix = np.zeros(num_datasets)
        
        for key, value in silos.items():
            labels = value.iloc[:, 0]
            features = value.iloc[:, 1:]

            features = features.values
            labels = labels.values
            
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=random_state)

            test_features[key] = X_test
            test_labels[key] = y_test

            train_features[key] = X_train
            train_labels[key] = y_train
    
        combined_train_features = [item for sublist in train_features.values() for item in sublist]
        combined_train_labels = [item for sublist in train_labels.values() for item in sublist]

        model = LogisticRegression(solver='lbfgs', max_iter=10000, warm_start=True)
        model.fit(combined_train_features, combined_train_labels)
        
        for i, key in enumerate(dataset_names):
            y_pred_proba = model.predict_proba(test_features[key])[:, 1]
            auc = roc_auc_score(test_labels[key], y_pred_proba)
            auc_matrix[i] = auc
            
        auc_matrices_cl.append(auc_matrix)
        
    # Calculate the average AUC matrix across all repetitions
    final_auc_matrix_cl = np.mean(auc_matrices_cl, axis=0)
    
    return final_auc_matrix_cl

def main():
    # Parameters           
    seed = 1234               # Fixed seed for reproducibility
    n_repeats = 1             # Number of monte carlo simulations
    baseline = "ettala"
    
    # dataset from pickle
    with open('data/data.pkl', 'rb') as file:
        silos = pickle.load(file)

        
    client_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)
        
    final_auc_matrix_cl = run_centralized_training(silos, client_names,
                                                   n_repeats=n_repeats, seed=seed, baseline=baseline)
    
     # After all experiments, print the AUC matrix
    print("\n### Final AUC Matrix ###")
    print(pd.DataFrame(np.array(final_auc_matrix_cl).reshape(1,-1), columns=client_names))


if __name__ == "__main__":
    main()
