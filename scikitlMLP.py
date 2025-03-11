from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time

def load_and_preprocess_data(file_path):
    """
    Load data from CSV and preprocess it.
    """
    # Load the data
    data = pd.read_csv(file_path)
    
    # Assuming the last column is the target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data
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
        batch_size=200,
        learning_rate_init=0.001,  # Same as TF default
        beta_1=0.9,  # Adam parameter, same as TF default
        beta_2=0.999,  # Adam parameter, same as TF default
        max_iter=100,
        tol=1e-8,  # Very small tolerance to ensure all 100 iterations run
        early_stopping=False,  # Disable early stopping
        n_iter_no_change=10,  # Not used when early_stopping=False
        random_state=42
    )

def run_experiment(file_path):
    """
    Run the complete experiment and return metrics
    """
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    
    # Create model
    model = create_mlp_model()
    
    # Measure training time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Measure inference time
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    # Calculate accuracy
    test_accuracy = model.score(X_test, y_test)
    
    results = {
        'training_time': training_time,
        'inference_time': inference_time,
        'test_accuracy': test_accuracy,
        'loss': model.loss_  # Final loss value from training
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
    
    # Print CPU info
    print(f"CPU Info: {platform.processor()}")
    try:
        import psutil
        print(f"Logical CPUs: {psutil.cpu_count(logical=True)}")
        print(f"Physical CPUs: {psutil.cpu_count(logical=False)}")
    except ImportError:
        print("psutil not available, install with pip install psutil for CPU details")
    
    # Create results file with headers
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Dataset', 'Run', 'Rows', 'Features',
            'Training Time (s)', 'Inference Time (s)', 
            'Test Accuracy', 'Final Loss'
        ])
    
    # Run tests for each dataset
    for file_path in data_files:
        dataset_name = os.path.basename(file_path)
        print(f"\nRunning tests for {dataset_name}...")
        
        # Get dataset dimensions
        data = pd.read_csv(file_path)
        num_rows, num_cols = data.shape
        num_features = num_cols - 1  # Assuming last column is target
        
        # Run multiple times
        for run in range(1, num_runs + 1):
            print(f"  Run {run}/{num_runs}...")
            results = run_experiment(file_path)
            
            # Append to CSV
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    dataset_name, run, num_rows, num_features,
                    f"{results['training_time']:.4f}",
                    f"{results['inference_time']:.4f}",
                    f"{results['test_accuracy']:.4f}",
                    f"{results['loss']:.4f}"
                ])
            
            # Print results
            print(f"    Training Time: {results['training_time']:.2f} seconds")
            print(f"    Inference Time: {results['inference_time']:.2f} seconds")
            print(f"    Test Accuracy: {results['test_accuracy']:.4f}")
    
    print(f"\nAll tests completed. Results saved to {output_file}")

if __name__ == "__main__":
    # List all data files to test
    data_files = [
        "data_500.csv",
        "data_1000.csv",
        "data_5000.csv",
        "data_10000.csv",
        "data_50000.csv"
    ]
    
    # Run tests - modify parameters as needed
    run_multiple_tests(
        data_files=data_files, 
        num_runs=3,  # Number of runs per dataset 
        output_file="sklearn_cpu_results.csv"
    )
