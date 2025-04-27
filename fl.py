import numpy as np
import pandas as pd
import pickle
from utils import Client, Server
from tqdm import tqdm
from utils import apply_spline_transformation, apply_transformation_noh
from utils import baseline_ettala, baseline_noh

'''This file runs the federated training'''

def run_federated_training(datasets, client_names, num_rounds = 1, n_repeats = 1, seed=1234, baseline='ettala'):
    
    num_clients = len(datasets)
    
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
    
    # Initialize a list to hold all AUC matrices
    auc_matrices_fl = [] # Stores [1 x dataset_names]

    # Repeat the experiment
    for repeat in tqdm(range(n_repeats)):
        random_state = np.random.randint(0, 10000000)
        
        ## Initialize an 1 x m matrix for AUC values
        auc_matrix_fl = np.zeros(num_clients)

        # Initialize Server
        server = Server()
        clients = []

        # Create and initialize clients from the dictionary
        for client_name, client_data in datasets.items():
            client = Client(df=client_data, name=client_name, random_state=random_state)
            clients.append(client)

        for round_num in range(num_rounds):
            
            client_updates = []
            for client in clients:
                coef, intercept, num_samples = client.train()
                if coef is not None and intercept is not None:
                    # Collect client updates: (coef, intercept, number of samples)
                    client_updates.append((coef, intercept, num_samples))
            
            # Aggregate the updates
            global_coef, global_intercept = server.aggregate(client_updates)
            
            # Send the global parameters to each client
            for client in clients:
                client.set_params(global_coef, global_intercept)
            
            
            # for client in clients:
            #     _, _, auc = client.evaluate(use_full_dataset=False)
            # print(f"Client {client.name}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}, AUC = {auc:.4f}")

        # Evaluate the final global model
        for _, client in enumerate(clients):
            _, _, auc = client.evaluate(use_full_dataset=False)
            auc_matrix_fl[client_names.index(client.name)] = auc
        
        auc_matrices_fl.append(auc_matrix_fl)
        

    # Calculate the average AUC matrix across all repetitions
    final_auc_matrix_fl = np.mean(auc_matrices_fl, axis=0)
    
    return(final_auc_matrix_fl)


def main():
    # Parameters           
    seed = 1234               # Fixed seed for reproducibility
    n_repeats = 1             # Number of monte carlo simulations
    num_rounds = 1
    baseline = "noh"
    
    # dataset from pickle
    with open('data/data.pkl', 'rb') as file:
        silos = pickle.load(file)
        
        
    client_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)
    
    final_auc_matrix_fl = run_federated_training(silos, client_names, num_rounds=num_rounds,
                                                 n_repeats= n_repeats, seed=seed, baseline=baseline)
    
    # After all experiments, print the AUC matrix
    print("\n### Final AUC Matrix ###")
    print(pd.DataFrame(np.array(final_auc_matrix_fl).reshape(1,-1), columns=client_names))

if __name__ == "__main__":
    main()