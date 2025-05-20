from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import os
import csv

BATCH_SIZE = 8192  # this batch size because it showed the most balanced data from the batch size experiments
LAYER_CONFIGS = [4, 8, 16, 32]  
NUM_RUNS = 3 
NEURONS_PER_LAYER = 100

def load_and_preprocess_data(file_path):
    # load the data
    data = pd.read_csv(file_path)
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # split 80:20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def create_mlp_model(num_layers):
    """
    Create MLPClassifier with configurable depth
    
    Args:
        num_layers: Number of hidden layers
    """
    # a tuple of layer sizes with the specified number of layers
    hidden_layer_sizes = tuple([NEURONS_PER_LAYER] * num_layers)
    
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 regularization parameter
        batch_size=BATCH_SIZE,
        learning_rate_init=0.001,  # tf default
        beta_1=0.9,  # tf default, Adam parameter
        beta_2=0.999,  # tf default, Adam parameter
        max_iter=100,
        tol=1e-8,  # small tolerance to ensure 100 iterations
        early_stopping=False,  # just in case
        n_iter_no_change=10,
        random_state=42
    )

def run_experiment(file_path, num_layers):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    
    model = create_mlp_model(num_layers=num_layers)
    print(f"Creating model with {num_layers} hidden layers...")
    
    # to measure training time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # to measure inference time
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    test_accuracy = model.score(X_test, y_test)
    
    results = {
        'num_layers': num_layers,
        'training_time': training_time,
        'inference_time': inference_time,
        'test_accuracy': test_accuracy,
        'loss': model.loss_  
    }
    
    return results

def run_multiple_tests(data_files, output_file="sklearn_layer_results.csv"):
    # if the directory doesn't already exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # results file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Dataset', 'Num_Layers', 'Run', 'Rows', 'Features',
            'Training Time (s)', 'Inference Time (s)', 
            'Test Accuracy', 'Final Loss'
        ])
    
    # each dataset tested with each layer config
    for file_path in data_files:
        dataset_name = os.path.basename(file_path)
        print(f"\nRunning tests for {dataset_name}...")
        
        data = pd.read_csv(file_path)
        num_rows, num_cols = data.shape
        num_features = num_cols - 1 
        
        for num_layers in LAYER_CONFIGS:
            print(f"Testing with {num_layers} layers...")
            
            for run in range(1, NUM_RUNS + 1):
                print(f"  Run {run}/{NUM_RUNS}...")
                results = run_experiment(file_path, num_layers)
                
                # append before continuing
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        dataset_name, num_layers, run, num_rows, num_features,
                        f"{results['training_time']:.4f}",
                        f"{results['inference_time']:.4f}",
                        f"{results['test_accuracy']:.4f}",
                        f"{results['loss']:.4f}"
                    ])
                
                # print results as well
                print(f"    Layers: {num_layers}")
                print(f"    Training Time: {results['training_time']:.2f} seconds")
                print(f"    Inference Time: {results['inference_time']:.2f} seconds")
                print(f"    Test Accuracy: {results['test_accuracy']:.4f}")
    
    print(f"\nAll tests completed. Results saved to {output_file}")

if __name__ == "__main__":
    data_files = [
        "datasets/data_500.csv",
        "datasets/data_1000.csv",
        "datasets/data_5000.csv",
        "datasets/data_10000.csv",
        "datasets/data_50000.csv",
        "datasets/data_100000.csv"
    ]
    
    run_multiple_tests(
        data_files=data_files,
        output_file=f"results/sklearn_cpu_layer_experiment_batchsize{BATCH_SIZE}.csv"
    )
