from sklearn.model_selection import train_test_split
from utils import apply_spline_transformation, apply_transformation_noh
from utils import baseline_ettala, baseline_noh
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
from utils import Client, Server
from tqdm import tqdm
import os
import random

'''This file runs fl and free-riding with subsampling'''

def run_lso_sub_sampling(silos, num_rounds=1, n_repeats=1, seed=1234, 
                         directory = 'results', baseline = 'ettala'):
    
    if baseline == 'ettala':
        baseline_function = baseline_ettala
        baseline_transformation = apply_spline_transformation
    elif baseline == 'noh':
        baseline_function = baseline_noh
        baseline_transformation = apply_transformation_noh
    else:
        raise ValueError(f"No baseline {baseline} found") 
    
    silos = baseline_transformation(silos)
    
    np.random.seed(seed)
    random.seed(seed)
    num_datasets = len(silos)
    n_display = 50
    mandatory_visit_interval = 100  # Force sampling every dataset count once every 100 repetitions

    auc_results_total_lso = defaultdict(list)  # Stores {dataset_count: [av_auc1, av_auc2, av_auc3], ...]}
    test_data_sizes_lso = defaultdict(list) # Stores sizes of test splits
    auc_results_total = defaultdict(list)
    test_data_sizes = defaultdict(list)
    
    # Initialize variance tracking
    r_variance = {r: 1.0 for r in range(1, num_datasets)}  # Start with uniform weights
    # r_variance = {r: 1.0 for r in range(4, num_datasets-3)}
    # r_variance[2] = 0.8
    # r_variance[3] = 0.8
    # r_variance[num_datasets-3] = 0.8
    # r_variance[num_datasets-2] = 0.8
    
    # Track when each dataset count was last visited
    last_visit = {r: 0 for r in range(1, num_datasets)}
 

    for repeat in tqdm(range(n_repeats)):
        # Initialize new random state for each repeat
        random_state = np.random.randint(0, 10000000)
        
        # Check if any dataset count hasn't been visited within the interval
        if repeat - min(last_visit.values()) >= mandatory_visit_interval:
            # Force a visit to the least recently visited dataset count
            dataset_count = min(last_visit, key=last_visit.get)
            # print(f"Mandatory visit to dataset count {dataset_count} at repetition {repeat + 1}")
        else:
            # Decide the dataset count based on variance
            weights = [r_variance[r] for r in range(2, num_datasets-1)]
            dataset_count = random.choices(range(2, num_datasets-1), weights)[0]
        
        # Update last visit for the selected dataset count
        last_visit[dataset_count] = repeat
        
        # Shuffle the datasets and select the first `dataset_count` datasets
        shuffled_datasets = list(silos.items())
        random.shuffle(shuffled_datasets)
        participating_datasets = dict(shuffled_datasets[:dataset_count])
        excluded_datasets = dict(shuffled_datasets[dataset_count:])
            
        server = Server()
        clients = []
    
        for client_name, client_data in participating_datasets.items():
            client = Client(df=client_data, name=client_name, random_state=random_state)
            clients.append(client)
            
        excluded_clients = []
        for client_name, client_data in excluded_datasets.items():
            client = Client(df=client_data, name=client_name, random_state=random_state)
            excluded_clients.append(client)
            
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
                
            # Also Send the global parameters to excluded clients
            for client in excluded_clients:
                client.set_params(global_coef, global_intercept)

        # Evaluate the final global model
        round_aucs_lso = []
        test_sizes_lso = []
        for _, client in enumerate(excluded_clients):
            _, _, auc = client.evaluate(use_full_dataset=True)
            round_aucs_lso.append(auc)
            test_sizes_lso.append(len(client.df))
            
        # Store individual AUCs for this dataset count
        auc_results_total_lso[dataset_count].extend(round_aucs_lso)
        # Store individual sizes for this dataset count
        test_data_sizes_lso[dataset_count].extend(test_sizes_lso)
            
        test_sizes = []
        round_aucs = []
        for _, client in enumerate(clients):
            _, _, auc = client.evaluate(use_full_dataset=False)
            round_aucs.append(auc)
            test_sizes.append(len(client.y_test))
        
        # Store individual AUCs for this dataset count
        auc_results_total[dataset_count].extend(round_aucs)
        # Store individual sizes for this dataset count
        test_data_sizes[dataset_count].extend(test_sizes)
            
    
        # # Update variance for this dataset count
        # current_aucs = np.array(auc_results_total[dataset_count])
        # if len(current_aucs) > 1:  # Variance is undefined for a single value
        #     r_variance[dataset_count] = np.var(current_aucs)
            
        # # Display variance rankings every n_display repetitions
        # if (repeat + 1) % n_display == 0:
        #     variance_by_count = {
        #         count: np.var(auc_results_total[count]) for count in auc_results_total
        #     }
        #     ranked_variance = sorted(variance_by_count.items(), key=lambda x: x[1], reverse=True)
        #     print(last_visit)
        #     # print(max(last_visit.values()))
        #     print(f"\nVariance Rankings After {repeat + 1} Repetitions:")
        #     for count, var in ranked_variance:
        #         print(f"  Dataset Count {count}: Variance = {var:.4f}")
    
    # save free-riding results
    average_results_lso = []
    for dataset_count, aucs in auc_results_total_lso.items():
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        # Compute weighted mean
        weights = np.array(test_data_sizes_lso[dataset_count])
        weighted_mean_auc = np.average(aucs, weights=weights)
        
        average_results_lso.append({
            'num_datasets': dataset_count,
            'auc': mean_auc,
            'std': std_auc,
            'weighted_mean_auc': weighted_mean_auc
            })

    # Convert to Pandas DataFrame
    df_results_lso = pd.DataFrame(average_results_lso)
    df_results_lso = df_results_lso.sort_values(by="num_datasets", ascending=True)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.exists(directory + '/subsiloing/fr'):
        os.makedirs(directory + '/subsiloing/fr')
        
    df_results_lso.to_csv(f"{directory}/subsiloing/fr/fr_average_{baseline}.csv", index=False)
    
    # save FL results
    average_results = []
    for dataset_count, aucs in auc_results_total.items():
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        # Compute weighted mean
        weights = np.array(test_data_sizes[dataset_count])
        weighted_mean_auc = np.average(aucs, weights=weights)
        
        average_results.append({
            'num_datasets': dataset_count,
            'auc': mean_auc,
            'std': std_auc,
            'weighted_mean_auc': weighted_mean_auc
            })

    # Convert to Pandas DataFrame
    df_results = pd.DataFrame(average_results)
    df_results = df_results.sort_values(by="num_datasets", ascending=True)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.exists(directory + '/subsiloing/fl'):
        os.makedirs(directory + '/subsiloing/fl')
        
    df_results.to_csv(f"{directory}/subsiloing/fl/fl_average_{baseline}.csv", index=False)
        
    return df_results_lso, df_results

def main():
    # Parameters
    seed = 1234               # Fixed seed for reproducibility
    num_rounds = 1
    n_repeats = 10000           # Number of Monte Carlo rounds
    baseline='ettala'
    
    # dataset from pickle
    with open('data/data.pkl', 'rb') as file:
        silos = pickle.load(file)
        
    average_results_lso, average_results_fl = run_lso_sub_sampling(silos, num_rounds=num_rounds, 
                                                 n_repeats=n_repeats,seed=seed, baseline=baseline)
    
    # Print the AUC matrix. For demonstration only
    print("\n### subsiloing with leave silo out training ###")
    # average_results_local.pop('average')
    print(pd.DataFrame(average_results_lso))
    print(pd.DataFrame(average_results_fl))

if __name__ == "__main__":
    main()