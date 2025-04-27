import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from utils import apply_spline_transformation, apply_transformation_noh
from utils import baseline_ettala, baseline_noh
from tqdm import tqdm
import os

from collections import defaultdict

'''This file trains local models with subsampling'''

def run_local_sub_sampling(datasets, n_repeats= 1, seed = 1234, 
                           directory = 'results', baseline = 'ettala'):
    
    np.random.seed(seed)

    auc_results_total = defaultdict(list) # Stores {dataset_name: [{sample_size: auc}, ...]}
    
    if baseline == 'ettala':
        baseline_function = baseline_ettala
        baseline_transformation = apply_spline_transformation
    elif baseline == 'noh':
        baseline_function = baseline_noh
        baseline_transformation = apply_transformation_noh
    else:
        raise ValueError(f"No baseline {baseline} found") 
    
    datasets = baseline_transformation(datasets)

    for repeat in tqdm(range(n_repeats)):
        random_state = np.random.randint(0, 10000000)

        for dataset_name, dataset in datasets.items():
            
            auc_for_number_of_silos = {}
            
            labels = dataset.iloc[:, 0]
            features = dataset.iloc[:, 1:]

            # Perform stratified sampling to get test split
            stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
            for train_idx, test_idx in stratified_split.split(features, labels):
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

            auc_results_local = {}

            model = LogisticRegression(solver='lbfgs', max_iter=10000)
            model.fit(X_train, y_train)

            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            
            total_silos = np.arange(1,len(datasets)+1)
            for current_silo_count in total_silos:
                auc_for_number_of_silos[current_silo_count] = auc
             
            auc_results_total[dataset_name].append(auc_for_number_of_silos)

    average_results = {}

    for dataset_name, auc_list in auc_results_total.items():
        auc_df = pd.DataFrame(auc_list).mean(axis=0)  
        average_results[dataset_name] = auc_df

    overall_averages = pd.DataFrame(average_results).mean(axis=1)
    overall_std = pd.DataFrame(average_results).std(axis=1)

    average_results['average'] = pd.concat([overall_averages, overall_std], axis=1)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.exists(directory + '/subsiloing/local'):
        os.makedirs(directory + '/subsiloing/local')

    for dataset_name,value in average_results.items():
        # value = pd.DataFrame(value)
        value = value.reset_index()
        if dataset_name == 'average':
            value.columns=['num_datasets', 'auc', 'std']
        else:
            continue
            # value.columns=['num_datasets', 'auc']
        value.to_csv(f"{directory}/subsiloing/local/local_{dataset_name}_{baseline}.csv", index=False)
        
    return average_results


def main():
    # Parameters
    n_repeats = 1             # Number of Monte-Carlo simulation rounds
    seed = 1234               # Fixed seed for reproducibility
    baseline='ettala'
    
    # dataset from pickle
    with open('data/data.pkl', 'rb') as file:
        silos = pickle.load(file)
        
    client_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)
        
    average_results_local = run_local_sub_sampling(silos, n_repeats=n_repeats, 
                                                   seed=seed, directory = 'results', baseline=baseline)
    
    # Print the AUC matrix for demonstration only
    print("\n### AUC Matrix for subsampling whe local training ###")
    average_results_local.pop('average')
    print(pd.DataFrame(average_results_local, columns=client_names))

if __name__ == "__main__":
    main()