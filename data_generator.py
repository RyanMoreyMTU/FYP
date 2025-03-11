from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import os

def generate_datasets(sizes, n_features=20, n_informative=10, n_redundant=5, 
                      output_dir="datasets"):
    os.makedirs(output_dir, exist_ok=True)
    
    for size in sizes:
        print(f"Generating dataset with {size} samples...")
        X, y = make_classification(
            n_samples=size,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=2,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        output_file = os.path.join(output_dir, f"data_{size}.csv")
        df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")

if __name__ == "__main__":
    # Define dataset sizes to generate
    sizes = [500, 1000, 5000, 10000, 20000, 50000, 100000]
    
    generate_datasets(
        sizes=sizes,
        n_features=20,  # Total features
        n_informative=10,  # Informative features
        n_redundant=5,  # Redundant features
        output_dir="datasets"
    )
    
    print("All datasets generated successfully.")
