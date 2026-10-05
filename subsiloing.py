# from subsiloing_fl_old import run_fl_sub_sampling
from subsiloing_fr import run_lso_sub_sampling
from subsiloing_local import run_local_sub_sampling
from subsiloing_baseline import run_baseline_sub_sampling
import pickle
from utils import apply_spline_transformation
import argparse
import os
import pandas as pd

'''This file trains various models with subsampling to construct the Figure 5'''

parser = argparse.ArgumentParser(description="A script with argparse defaults.")
parser.add_argument("--data", type=str, default="data", help="Data folder")
parser.add_argument("--output", type=str, default="results", help="Folder to store the output")
parser.add_argument("--monte_carlo", type=int, default=10000, help="Number of Monte Carlo Simulations")
parser.add_argument("--federated_rounds", type=int, default=1, help="Number of Federated Rounds")
parser.add_argument("--baseline", type=str, default="noh", help="Transformation of variables to use")

# Parse arguments
args = parser.parse_args()

# create output directory
if not os.path.exists(args.output):
        os.makedirs(args.output)


 # Parameters
seed = 1234               # Fixed seed for reproducibility
num_rounds = args.federated_rounds
n_repeats = args.monte_carlo
baseline = args.baseline

with open(args.data+'/data.pkl', 'rb') as file:
        silos = pickle.load(file)
        
    
# client_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)

print("\n Running local subsampling...")
average_results_local = run_local_sub_sampling(silos, n_repeats=1, 
                                               seed=seed, directory = args.output, baseline=baseline)
print("\n Running federated learning subsampling...")
# average_results_fl = run_fl_sub_sampling(silos, num_rounds=num_rounds, n_repeats=15000*n_repeats, 
                                        #  seed=seed, directory = args.output, baseline=baseline)
print("\n Running leave-silo-out subsampling...")
average_results_lso, average_results_fl = run_lso_sub_sampling(silos, num_rounds=num_rounds, n_repeats=n_repeats, 
                                           seed=seed, directory = args.output, baseline=baseline)

print("\n Running baseline subsampling...")
average_results_baseline = run_baseline_sub_sampling(silos, n_repeats=1, 
                                                     seed=seed, directory = args.output, baseline=baseline)

print("\n Finished")