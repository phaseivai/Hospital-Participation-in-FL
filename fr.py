import numpy as np
import pandas as pd
import pickle
from utils import apply_spline_transformation, apply_transformation_noh
from utils import baseline_ettala, baseline_noh
from utils import Client, Server
from tqdm import tqdm

'''This file runs emulates free-riding'''

def run_lso_training(silos, client_names, num_rounds = 1, n_repeats = 1, seed = 1, baseline="ettala"):

    num_rounds = num_rounds

    # For reproducibility 
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
    auc_matrices_lso = [] # Stores [dataset_names x dataset_names]

    for repeat in tqdm(range(n_repeats)):
        
        random_state = np.random.randint(0, 10000000)
        
        # Create an empty matrix to store AUC scores (m x m matrix, where m = number of clients)
        num_clients = len(silos)
        auc_matrix_lso = np.zeros((num_clients, num_clients))

        for exclude_idx, exclude_client_name in enumerate(client_names):

            # Initialize the server
            server = Server()
            
            # Excluding one client from the training process
            participating_clients = [
                Client(df=silos[name], name=name, random_state=random_state) 
                for idx, name in enumerate(client_names) 
                if idx != exclude_idx
            ]

            # Train the federated model with remaining clients
            for round_num in range(num_rounds):
                
                client_updates = []
                for client in participating_clients:
                    coef, intercept, num_samples = client.train()
                    if coef is not None and intercept is not None:
                        # Collect client updates: (coef, intercept, number of samples)
                        client_updates.append((coef, intercept, num_samples))
                
                # Aggregate the updates
                global_coef, global_intercept = server.aggregate(client_updates)
                
                # Send the global parameters to each client
                for client in participating_clients:
                    client.set_params(global_coef, global_intercept)
            
            # Evaluate on the holdout sets of all participating clients
            for _, client in enumerate(participating_clients):
                _, _, auc = client.evaluate()
                
                # Store AUC in the matrix
                auc_matrix_lso[exclude_idx, client_names.index(client.name)] = auc
                # print(f"Storing AUC in matrix at [{exclude_idx}, {client_names.index(client.name)}]: {auc_matrix_lso[exclude_idx, client_names.index(client.name)]:.4f}")

            # Evaluate on the client that was excluded from training
            excluded_client = Client(df=silos[exclude_client_name], name=exclude_client_name)
            excluded_client.set_params(global_coef, global_intercept)
            
            # Use the entire dataset of the excluded client for evaluation
            _, _, auc = excluded_client.evaluate(use_full_dataset=True)
            
            auc_matrix_lso[exclude_idx, exclude_idx] = auc
            # print(f"Storing AUC in matrix at [{exclude_idx}, {exclude_idx}]: {auc_matrix[exclude_idx, exclude_idx]:.4f}")
            
            auc_matrices_lso.append(auc_matrix_lso)
        
    # Calculate the average AUC matrix across all repetitions
    final_auc_matrix_lso = np.mean(auc_matrices_lso, axis=0)
    
    return final_auc_matrix_lso

def main():
    # Parameters           
    seed = 1234               # Fixed seed for reproducibility
    n_repeats = 1
    num_rounds = 1
    baseline = "noh"
    
    # dataset from pickle
    with open('data/data.pkl', 'rb') as file:
        silos = pickle.load(file)
        
        
    client_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)
        
    final_auc_matrix_lso = run_lso_training(silos, client_names, num_rounds=num_rounds, 
                                            n_repeats=n_repeats, seed=seed, baseline=baseline)
    
    # print the AUC matrix, only for illustrative purposes
    print("\n### Final AUC Matrix ###")
    print(pd.DataFrame(final_auc_matrix_lso, columns=client_names, index=client_names))

if __name__ == "__main__":
    main()