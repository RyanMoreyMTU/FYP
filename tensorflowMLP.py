#ignore
# eval "$(/root/anaconda3/bin/conda shell.bash hook)"
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# tf.config.set_visible_devices([], 'GPU')
BATCH_SIZE=256
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

def create_mlp_model(input_dim):
    """
    Create a simple MLP model with identical architecture to sklearn's MLPClassifier
    """
    # configure to use Adam optimizer with same defaults as sklearn
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def run_experiment(file_path):
    """
    Run the complete experiment and return metrics
    """
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    model = create_mlp_model(X_train.shape[1])
    
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
        'training_time': training_time,
        'inference_time': inference_time,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss
    }
    
    return results

def run_multiple_tests(data_files, num_runs=3, output_file="tensorflow_results.csv"):
    """
    Run multiple tests for each dataset and save results to CSV.
    
    Args:
        data_files: List of data file paths to test
        num_runs: Number of repeat runs for each dataset
        output_file: CSV file to save results
    """
    import csv
    import os
    
    # GPU Availibility Check
    gpus = tf.config.list_physical_devices('GPU')
    gpu_available = len(gpus) > 0
    print(f"GPU Available: {gpu_available}")
    if gpu_available:
        print(f"GPU Devices: {gpus}")
    
    # results csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Dataset', 'Run', 'Rows', 'Features',
            'Training Time (s)', 'Inference Time (s)', 
            'Test Accuracy', 'Test Loss'
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
                    f"{results['test_loss']:.4f}"
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
        output_file=f"results/tensorflow_cpu_results_batchsize{BATCH_SIZE}.csv"
    )
