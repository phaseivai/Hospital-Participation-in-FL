import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split  # for train/test split
from utils import apply_spline_transformation, apply_transformation_noh
from utils import baseline_ettala, baseline_noh
from tqdm import tqdm
import pandas as pd
import argparse
import os


def run_local_training(silos, dataset_names, n_repeats = 100, seed=1234, baseline = "ettala"):
    
            
    np.random.seed(seed)  # For reproducibility    # Number of times to repeat the experiment
    
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

    # Number of datasets (m x m matrix)
    num_datasets = len(silos)

    # Dictionary of dataset names for referencing
    dataset_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)
    # dataset_pseud = ['es', 'kr', 'de1', 'ch2', 'ch1', 'us2', 'us1', 'nl', 'it', 'de2', 'gb']

    # Initialize a list to hold all AUC matrices
    auc_matrices = []

    # print("\n Running single silo training...")  

    for repeat in tqdm(range(n_repeats)):
        
        random_state = np.random.randint(0, 10000000)
        
        # Create dictionaries for storing train/test features and labels
        test_features = {}
        test_labels = {}
        train_features = {}
        train_labels = {}
        
        # Initialize an m x m matrix for AUC values
        auc_matrix = np.zeros((num_datasets, num_datasets))
        
        for key, value in silos.items():
            
            labels = value.iloc[:, 0]  
            features = value.iloc[:, 1:] 

            features = features.values
            labels = labels.values
            
            # Split dataset into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=random_state)

            test_features[key] = X_test
            test_labels[key] = y_test

            train_features[key] = X_train
            train_labels[key] = y_train

        # Train models on each dataset and populate AUC matrix
        for i, key in enumerate(dataset_names):

            model = LogisticRegression(solver='lbfgs', max_iter=10000)
            model.fit(train_features[key], train_labels[key])

            # Test on each dataset
            for j, key1 in enumerate(dataset_names):
                
                y_pred_proba = model.predict_proba(test_features[key1])[:, 1]
                auc = roc_auc_score(test_labels[key1], y_pred_proba)
                
                # Store the AUC value in the matrix
                auc_matrix[i, j] = auc
            
        
        auc_matrices.append(auc_matrix)
        
        
    # print("\n Finished")
        
    # Calculate the average AUC matrix across all repetitions
    final_auc_matrix = np.mean(auc_matrices, axis=0)

    return(final_auc_matrix)


def main():
    ## Parameters
    seed = 1234               # Fixed seed for reproducibility
    num_rounds = 1
    n_repeats = 2
    baseline = "noh"
    
    # dataset from pickle
    with open('data/data.pkl', 'rb') as file:
        silos = pickle.load(file)
        
        
    client_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)
        
    final_auc_matrix_local = run_local_training(silos, client_names, n_repeats=n_repeats, 
                                                seed=seed, baseline = baseline)
    
     # After all experiments, print the AUC matrix
    print("\n### Final AUC Matrix ###")
    print(pd.DataFrame(np.array(final_auc_matrix_local), columns=client_names))


if __name__ == "__main__":
    main()