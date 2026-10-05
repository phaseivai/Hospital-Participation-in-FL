import numpy as np
import pickle
from centralized import run_centralized_training
from fl import run_federated_training
from fr import run_lso_training
from local import run_local_training
from baseline import run_baseline_training
import pandas as pd
import argparse
import os

'''This file produces the results for Figure 3'''

def calc_color(color):
    return 100*(color-40)/(100-40)


# Function to generate LaTeX matrix code with row and column labels
def generate_latex_matrix(auc_matrix, dataset_names, client_pseud, dataset_sizes):
    latex_code = ''
    latex_code += "\\definecolor{lowColor}{HTML}{E6E6FA}\n"
    latex_code += "\\definecolor{highColor}{HTML}{009E73} \n"
    # Start the LaTeX code for the matrix
    latex_code += "\\matrix[matrix of nodes, nodes={font=\\scriptsize}] (m) {\n"
    
    # Add AUC values row by row with two additional columns (Average and Weighted Average)
    for i, row in enumerate(auc_matrix):
        avg_auc = np.mean(row)  # Average AUC
        weighted_avg_auc = np.average(row, weights=dataset_sizes)  # Weighted average AUC

        # Prepare each row's values for each AUC
        row_latex = ""
        for j, auc in enumerate(row):
            # Scale AUC to 0-100 for color
            auc_scaled = auc * 100
            
            # Construct the node with custom coloring
            row_latex += f"\\node[name=m-{i+1}-{j+1}, fill=highColor!{calc_color(auc_scaled):.1f}!lowColor] {{{auc_scaled:.1f}}}; & "
        
        # Add Average and Weighted Average as the last two columns
        row_latex += f"\\node[name=m-{i+1}-{len(row)+1}, fill=highColor!{calc_color(avg_auc*100):.1f}!lowColor] {{{avg_auc*100:.1f}}}; & "
        row_latex += f"\\node[name=m-{i+1}-{len(row)+2}, fill=highColor!{calc_color(weighted_avg_auc*100):.1f}!lowColor] {{{weighted_avg_auc*100:.1f}}}; \\\\\n"

        latex_code += "    " + row_latex
    
    latex_code += "  };\n"  # Close the matrix

    # Add labels for rows: Centralized and Federated training
    latex_code += f"  \\node[anchor=east] at ([xshift=-0.16cm]m-1-1.west) {{\\small{{BL}}}};\n"
    latex_code += f"  \\node[anchor=east] at ([xshift=-0.16cm]m-2-1.west) {{\\small{{LOC}}}};\n"
    latex_code += f"  \\node[anchor=east] at ([xshift=-0.16cm]m-3-1.west) {{\\small{{CEN}}}};\n"
    latex_code += f"  \\node[anchor=east] at ([xshift=-0.16cm]m-4-1.west) {{\\small{{FL}}}};\n"
    latex_code += f"  \\node[anchor=east] at ([xshift=-0.16cm]m-5-1.west) {{\\small{{LSO}}}};\n"
    
    
    # Add column labels using the custom node format
    for j, dataset_name in enumerate(client_pseud):
        latex_code += f"  \\node[anchor=south] at ([yshift= 0cm]m-1-{j+1}.north) {{\\small{{{dataset_name}}}}};\n"
    
    # Add labels for the two extra columns: Average and Weighted Average
    latex_code += f"  \\node[anchor=south] at ([yshift= 0cm]m-1-{len(dataset_names)+1}.north) {{\\small{{AV-H}}}};\n"
    latex_code += f"  \\node[anchor=south] at ([yshift= 0cm]m-1-{len(dataset_names)+2}.north) {{\\small{{AV-W}}}};\n"
    
    
    return latex_code


parser = argparse.ArgumentParser(description="A script with argparse defaults.")
parser.add_argument("--data", type=str, default="data", help="Data folder")
parser.add_argument("--output", type=str, default="test", help="Folder to store the output")
parser.add_argument("--monte_carlo", type=int, default=10000, help="Number of Monte Carlo Simulations")
parser.add_argument("--federated_rounds", type=int, default=1, help="Number of Federated Rounds")
parser.add_argument("--baseline", type=str, default="ettala", help="Transformation of variables to use")
# Parse arguments
args = parser.parse_args()

# create output directory
if not os.path.exists(args.output):
        os.makedirs(args.output)

# Parameters           
seed = 1234               # Fixed seed for reproducibility
n_repeats = args.monte_carlo        # Monte Carlo repetitions
num_rounds = args.federated_rounds      #Number of Federated Rounds
baseline=args.baseline

with open(args.data+'/data.pkl', 'rb') as file:
        silos = pickle.load(file)

    
num_clients = len(silos)
client_names = sorted(silos.keys(), key=lambda k: len(silos[k]), reverse=True)


print("\n Running baseline training...")    
final_auc_matrix_baseline = run_baseline_training(silos, client_names,
                                                  n_repeats=n_repeats, seed=seed, baseline=baseline)
print("\n Running local training...")    
final_auc_matrix_local = run_local_training(silos, client_names, n_repeats=n_repeats, 
                                                seed=seed, baseline = baseline)
final_auc_matrix_local = np.diagonal(final_auc_matrix_local)
print("\n Running centralized training...")    
final_auc_matrix_cl = run_centralized_training(silos, client_names,
                                               n_repeats=n_repeats, seed=seed,baseline=baseline)
print("\n Running federated training...")
final_auc_matrix_fl = run_federated_training(silos, client_names,
                                             n_repeats=n_repeats, seed=seed, baseline=baseline)
print("\n Running leave silo out training...")
final_auc_matrix_lso = run_lso_training(silos, client_names,
                                        n_repeats=n_repeats, seed=seed, baseline=baseline)
print("\n Finished")
final_auc_matrix_lso = np.diagonal(final_auc_matrix_lso)


# Stack the full matrix
final_auc_matrix = np.vstack([final_auc_matrix_baseline.reshape(1, -1), 
                              final_auc_matrix_local.reshape(1, -1), 
                              final_auc_matrix_cl.reshape(1, -1), 
                              final_auc_matrix_fl.reshape(1, -1), 
                              final_auc_matrix_lso.reshape(1, -1)])


# Generate the LaTeX code for the AUC matrix with dataset labels
latex_auc_matrix = generate_latex_matrix(final_auc_matrix, client_names, client_names, dataset_sizes = [len(silos[key]) for key in client_names])

f = open(args.output + "/matrix_msl_"+args.baseline+".txt", "w")
f.write(latex_auc_matrix)
f.close()

pd.DataFrame(final_auc_matrix).to_csv(args.output + "/matrix_msl_"+args.baseline+ ".csv", index=False, header=False)