import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # for train/test split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
import joblib

''' Taken from the Otto Ettala et al. “Individualised non-contrast MRI-based risk estimation and shared decision-making in
men with a suspicion of prostate cancer: Protocol for multicentre randomised controlled trial (multi-IMPROD V. 2.0)" '''
def get_spline(x, i):
    t = [3.80, 6.60, 9.40, 18.47]
    value = np.power(max(x - t[i], 0), 3) - np.power(max(x - t[2], 0), 3)*(t[3] - t[i])/(t[3] - t[2]) \
        + np.power(max(x - t[3], 0), 3)*(t[2] - t[i])/(t[3] - t[2]) 
    return value

''' Taken from the Otto Ettala et al. “Individualised non-contrast MRI-based risk estimation and shared decision-making in
men with a suspicion of prostate cancer: Protocol for multicentre randomised controlled trial (multi-IMPROD V. 2.0)" '''
def apply_spline_transformation(silos):
    silos = silos.copy() 
    for key, value in silos.items():
        value['PV'] = value.apply(lambda row: row['PV']/0.7 if row['5ARI'] == 1 else row['PV'], axis=1)
        value['PSA'] = value.apply(lambda row: row['PSA']*2 if row['5ARI'] == 1 else row['PSA'], axis=1)
        value = value.drop(['5ARI'], axis=1)
        
        value['PSA_SPLINE_2'] = value['PSA'].apply(get_spline, i=0)
        value['PSA_SPLINE_3'] = value['PSA'].apply(get_spline, i=1)

        value['PIRADS'] = value['PIRADS'].map({0: '0', 1: '0', 2: '0', 3: '3', 4: '4', 5: '5'})
        all_categories = ['0', '3', '4', '5']
        value = pd.get_dummies(value, columns=['PIRADS'], prefix='PIRADS').reindex(
            columns=value.columns.tolist()[:] + ['PIRADS_' + cat for cat in all_categories], fill_value=0)
        value = value.drop(['PIRADS','PIRADS_0'], axis=1)
        value = value.astype({'sig_cancer': int, 'PIRADS_3':float, 'PIRADS_4':float, 'PIRADS_5':float})
        silos[key] = value
    return silos


''' Taken from the Otto Ettala et al. “Individualised non-contrast MRI-based risk estimation and shared decision-making in
men with a suspicion of prostate cancer: Protocol for multicentre randomised controlled trial (multi-IMPROD V. 2.0)" '''
def baseline_ettala(row):
    # return max(row['column1'] - row['column2'], 0)
    pred = -6.97314184 + 0.064172722*row['age'] - 0.008141264*row['PV'] - 0.182694534*row['PSA'] + 0.006136442*row['PSA_SPLINE_2'] \
    - 0.013049396*row['PSA_SPLINE_3'] \
    + 1.37637197*row['PIRADS_3'] \
    + 2.50939431*row['PIRADS_4'] \
    + 4.07331563 *row['PIRADS_5']

    pred = np.exp(pred)/(1 + np.exp(pred))
    return pred

''' Taken from the Tae Il Noh et al. “A Predictive Model Based on Bi-parametric Magnetic Resonance Imaging
and Clinical Parameters for Clinically Significant Prostate Cancer
in the Korean Population" '''
def apply_transformation_noh(silos):
    silos = silos.copy() 
    for key, value in silos.items():
        value['PIRADS_group'] = value['PIRADS'].map({0: 0, 1: 0, 2: 0, 3: 2, 4: 3, 5: 4})
        value['psad'] = value['PSA']/value['PV']
        # Define bins and corresponding labels
        bins = [0, 0.06, 0.09, 0.14, 0.19, 0.24, np.inf]  # Define interval edges
        labels = [1, 2, 3, 4, 5, 6]  # Labels for each bin
        value['psad_group'] = pd.cut(value['psad'], bins=bins, labels=labels, include_lowest=True)
        value = value.drop(['5ARI','PIRADS','PV','PSA','psad'], axis=1)
        silos[key] = value
    return silos


''' Taken from the Tae Il Noh et al. “A Predictive Model Based on Bi-parametric Magnetic Resonance Imaging
and Clinical Parameters for Clinically Significant Prostate Cancer
in the Korean Population" '''

def baseline_noh(row):
    # return max(row['column1'] - row['column2'], 0)
    pred = -9.5189 + 0.0511*row['age'] \
    + 1.1834*row['PIRADS_group']\
    + 0.4847*row['psad_group']\

    pred = np.exp(pred)/(1 + np.exp(pred))
    return pred

'''This computes off-diagonal mean difference between Single-silo matrix and LSO matrix'''
def compute_off_diag_mean(matrix1,matrix2):
    n = matrix1.shape[0]
    off_diagonal_mask = ~np.eye(n, dtype=bool)  
    off_diag_diff = (matrix1 - matrix2)[off_diagonal_mask]
    mean_off_diag_diff = np.mean(off_diag_diff)
    
    return mean_off_diag_diff


# Custom Logistic Regression for FL
class CustomLogisticRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None

    def fit(self, X, y):
        model = LogisticRegression(solver='lbfgs', max_iter=10000)
        model.fit(X, y)
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        self.n_iter_ = model.n_iter_

    def predict_proba(self, X):
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model parameters not set.")
        z = np.dot(X, self.coef_.T) + self.intercept_
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        prob = self.predict_proba(X)
        return (prob >= 0.5).astype(int)
    
    def save(self, filename):
        joblib.dump({'coef_': self.coef_, 'intercept_': self.intercept_, 'n_iter_': self.n_iter_}, filename)

    @classmethod
    def load(cls, filename):
        params = joblib.load(filename)
        model = cls()
        model.coef_ = params['coef_']
        model.intercept_ = params['intercept_']
        model.n_iter_ = params['n_iter_']
        return model

# Custom Server for FL
class Server:
    def aggregate(self, client_updates):
        # Get total number of samples from all clients
        total_samples = sum(client_update[2] for client_update in client_updates)

        # Perform weighted averaging of coefficients and intercepts
        global_coef = np.sum([coef * num_samples for coef, intercept, num_samples in client_updates], axis=0) / total_samples
        global_intercept = np.sum([intercept * num_samples for coef, intercept, num_samples in client_updates], axis=0) / total_samples

        return global_coef, global_intercept

# Cient for FL
class Client:
    def __init__(self, df, name, random_state=1234):
        self.df = df
        self.name = name
        self.random_state = random_state
        self.model = CustomLogisticRegression()
        self.X_test = None
        self.y_test = None

    def train(self, initial_sample_size=None):
        X_train, X_test, y_train, y_test = self.prepare_data(self.df, self.random_state, initial_sample_size, is_train=True)
        
        if len(np.unique(y_train)) < 2:
            print(f"Skipping training for client {self.name} due to insufficient class diversity.")
            return None, None, None

        self.X_test = X_test  # Store test data
        self.y_test = y_test  # Store test labels

        # Fit model and return number of samples used for training
        self.model.fit(X_train, y_train)
        num_samples = X_train.shape[0]
        return self.model.coef_, self.model.intercept_, num_samples

    def prepare_data(self, df, random_state=1234, initial_sample_size=None, is_train=False):
        # Extract features and labels
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values

        if is_train:
            # Perform stratified test split
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state, stratify=y
            )

            if initial_sample_size is not None:
                train_fraction = initial_sample_size / len(X_train_full)
                
                # If train_fraction < 1, perform a secondary split
                if train_fraction < 1:
                    X_train, _, y_train, _ = train_test_split(
                        X_train_full, y_train_full, train_size=train_fraction, random_state=random_state, stratify=y_train_full
                    )
                else:
                    X_train, y_train = X_train_full, y_train_full
            else:
                X_train = X_train_full
                y_train = y_train_full
            
            return X_train, X_test, y_train, y_test
        else:
            # Return entire dataset for evaluation (test)
            return X, y

    def set_params(self, coef, intercept):
        self.model.coef_ = coef
        self.model.intercept_ = intercept

    def evaluate(self, use_full_dataset=False):
        if use_full_dataset:
            # Evaluate on the entire dataset
            X, y = self.prepare_data(self.df, is_train=False)
        else:
            # Use stored test data for evaluation
            if self.X_test is None or self.y_test is None:
                raise ValueError("Test data not available for evaluation.")
            X, y = self.X_test, self.y_test

        y_pred_proba = self.model.predict_proba(X)
        y_pred = self.model.predict(X)
        loss = log_loss(y, y_pred_proba)
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)
        return loss, accuracy, auc