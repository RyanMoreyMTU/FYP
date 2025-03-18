import tensorflow as tf
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import csv

# Define constants
BATCH_SIZE = 8192  # Using the batch size where GPU starts showing advantage
LAYER_CONFIGS = [4, 8, 16, 32]  # Layer configurations to test
NUM_RUNS = 3  # Number of runs per configuration
NEURONS_PER_LAYER = 100  # Number of neurons per layer

def load_and_preprocess_data(file_path):
    """
    Load data from CSV and preprocess it.
    """
    # Load the data
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

def create_mlp_model(input_dim, num_layers):
    """
    Create a configurable MLP model with variable depth
    
    Args:
        input_dim: Dimension of input features
        num_layers: Number of hidden layers
    """
    # configure to use Adam optimizer with same defaults as sklearn
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(tf.keras.layers.Dense(NEURONS_PER_LAYER, activation='relu', input_dim=input_dim))
    
    # Hidden layers
    for _ in range(num_layers - 1):  # -1 because we already added the first layer
        model.add(tf.keras.layers.Dense(NEURONS_PER_LAYER, activation='relu'))
    
    # Output layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def run_experiment(file_path, num_layers):
    """
    Run the complete experiment with specified number of layers and return metrics
    """
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    model = create_mlp_model(X_train.shape[1], num_layers=num_layers)
    
    # Print model summary for verification
    print(f"Model with {num_layers} hidden layers:")
    model.summary()
    
    # to measure training time
    start_time = time.time()
    
    # train model without validation split to match sklearn
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    
    training_time = time.time() - start_time
    
    # Measure inference time and accuracy
    start_time = time.time()
    y_pred = model.predict(X_test, verbose=0)
    inference_time = time.time() - start_time
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    results = {
        'num_layers': num_layers,
        'training_time': training_time,
        'inference_time': inference_time,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss
    }
    
    return results

def run_multiple_tests(data_files, output_file="tensorflow_layer_results.csv"):
    """
    Run multiple tests for each dataset and layer configuration, and save results to CSV.
    
    Args:
        data_files: List of data file paths to test
        output_file: CSV file to save results
    """
    # Make results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # GPU Availability Check
    gpus = tf.config.list_physical_devices('GPU')
    gpu_available = len(gpus) > 0
    print(f"GPU Available: {gpu_available}")
    if gpu_available:
        print(f"GPU Devices: {gpus}")
    
    # results csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Dataset', 'Num_Layers', 'Run', 'Rows', 'Features',
            'Training Time (s)', 'Inference Time (s)', 
            'Test Accuracy', 'Test Loss'
        ])
    
    # Test each dataset with different layer configurations
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
                        f"{results['test_loss']:.4f}"
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
    
    # For GPU
    run_multiple_tests(
        data_files=data_files,
        output_file=f"results/tensorflow_gpu_layer_experiment_batchsize{BATCH_SIZE}.csv"
    )