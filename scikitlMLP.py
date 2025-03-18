from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
BATCH_SIZE = 32768
def load_and_preprocess_data(file_path):
    """
    Load data from CSV and preprocess it.
    """

    data = pd.read_csv(file_path)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def create_mlp_model():
    """
    Create MLPClassifier with identical architecture to TensorFlow model
    """
    return MLPClassifier(
        hidden_layer_sizes=(100, 100),
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

def run_experiment(file_path):
    """
    Run the complete experiment and return metrics
    """
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    
    model = create_mlp_model()
    
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
        'training_time': training_time,
        'inference_time': inference_time,
        'test_accuracy': test_accuracy,
        'loss': model.loss_  
    }
    
    return results

def run_multiple_tests(data_files, num_runs=3, output_file="sklearn_results.csv"):
    """
    Run multiple tests for each dataset and save results to CSV.
    
    Args:
        data_files: List of data file paths to test
        num_runs: Number of repeat runs for each dataset
        output_file: CSV file to save results
    """
    import csv
    import os
    import platform
    
    # results file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Dataset', 'Run', 'Rows', 'Features',
            'Training Time (s)', 'Inference Time (s)', 
            'Test Accuracy', 'Final Loss'
        ])
    
    # each dataset gets 3 runs, defined at the end of the file
    for file_path in data_files:
        dataset_name = os.path.basename(file_path)
        print(f"\nRunning tests for {dataset_name}...")
        
        data = pd.read_csv(file_path)
        num_rows, num_cols = data.shape
        num_features = num_cols - 1 
        
        for run in range(1, num_runs + 1):
            print(f"  Run {run}/{num_runs}...")
            results = run_experiment(file_path)
            
            # append before continuing
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    dataset_name, run, num_rows, num_features,
                    f"{results['training_time']:.4f}",
                    f"{results['inference_time']:.4f}",
                    f"{results['test_accuracy']:.4f}",
                    f"{results['loss']:.4f}"
                ])
            
            # print results as well
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
        num_runs=3,  
        output_file=f"results/sklearn_cpu_results_batchsize{BATCH_SIZE}.csv"
    )
