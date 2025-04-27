import numpy as np
import pandas as pd
import pickle
from utils import apply_spline_transformation, apply_transformation_noh
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
from collections import defaultdict
from utils import baseline_ettala, baseline_noh

'''This file trains baseline models (from the Otto Ettala et al. “Individualised non-contrast 
MRI-based risk estimation and shared decision-making in men with a suspicion of prostate cancer: 
Protocol for multicentre randomised controlled trial (multi-IMPROD V. 2.0)") with subsampling'''

def run_baseline_sub_sampling(silos, n_repeats=1, seed=1234, 
                              directory = 'results', baseline = 'ettala'):

    np.random.seed(seed)

    auc_results_total = defaultdict(list) # Stores {dataset_name: [{n_silos: auc}, ...]}

    if baseline == 'ettala':
        baseline_function = baseline_ettala
        baseline_transformation = apply_spline_transformation
    elif baseline == 'noh':
        baseline_function = baseline_noh
        baseline_transformation = apply_transformation_noh
    else:
        raise ValueError(f"No baseline {baseline} found") 
    
    silos = baseline_transformation(silos)

    # for repeat in tqdm(range(n_repeats)):
    random_state = np.random.randint(0, 10000000)
    for dataset_name, dataset in silos.items():
        
        auc_for_number_of_silos = {} 
            
        # train_df, test_df = train_test_split(dataset, test_size=0.2, stratify=dataset['sig_cancer'], random_state=random_state)
        # test_df['pred'] = dataset.apply(baseline_ettala, axis=1)
        silos[dataset_name]['pred'] = silos[dataset_name].apply(baseline_function, axis=1)
        auc = roc_auc_score(silos[dataset_name]['sig_cancer'], silos[dataset_name]['pred'])
        
        total_silos = np.arange(1,len(silos)+1)
        for current_silo_count in total_silos:
            auc_for_number_of_silos[current_silo_count] = auc
            
        auc_results_total[dataset_name].append(auc_for_number_of_silos)
                

    # Averaging results and creating a DataFrame for plotting
    average_results = {}
    for dataset_name, auc_list in auc_results_total.items():
        auc_df = pd.DataFrame(auc_list).mean(axis=0)  
        average_results[dataset_name] = auc_df

    # Compute overall averages across all datasets
    overall_averages = pd.DataFrame(average_results).mean(axis=1)
    overall_std = pd.DataFrame(average_results).std(axis=1)

    # Combine the dataset-specific averages with the overall average
    average_results['average'] = pd.concat([overall_averages, overall_std], axis=1)

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.exists(directory + '/subsiloing/baseline'):
        os.makedirs(directory + '/subsiloing/baseline')

    for dataset_name,value in average_results.items():
        value = value.reset_index()
        if dataset_name == 'average':
            value.columns=['num_datasets', 'auc', 'std']
        else:
            continue
            # value.columns=['num_datasets', 'auc']
        value.to_csv(f"{directory}/subsiloing/baseline/baseline_{dataset_name}_{baseline}.csv", index=False)
        
    return average_results
    
def main():
    # Parameters
    seed = 1234               # Fixed seed for reproducibility
    n_repeats = 1
    baseline='ettala'
    
    # dataset from pickle
    with open('data/data.pkl', 'rb') as file:
        silos = pickle.load(file)
        
        
    client_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)
        
    average_results_baseline = run_baseline_sub_sampling(silos, n_repeats=n_repeats, 
                                                         seed=seed, baseline=baseline)
    

    # Print the AUC matrix. For demonstration only
    print("\n### AUC Matrix for subsampling with baseline training ###")
    # average_results_baseline.pop('average')
    average_results_baseline['average'] = average_results_baseline['average'].reset_index()
    average_results_baseline['average'].columns=['num_datasets', 'auc', 'std']
    print(pd.DataFrame(average_results_baseline['average']))

if __name__ == "__main__":
    main()