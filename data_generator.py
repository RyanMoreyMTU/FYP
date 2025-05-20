<<<<<<< HEAD
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import os

def generate_datasets(sizes, n_features=20, n_informative=10, n_redundant=5, 
                      output_dir="datasets"):
    # directory created if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # generation of datasets
    for size in sizes:
        print(f"Generating dataset with {size} samples...")
        
        # data being generated
        X, y = make_classification(
            n_samples=size,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=2,
            random_state=42
        )
        
        # dataframe creation
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # saved to csv
        output_file = os.path.join(output_dir, f"data_{size}.csv")
        df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")

if __name__ == "__main__":
    # dataset size list
    sizes = [500, 1000, 5000, 10000, 50000, 100000]
    
    # calling the dataset generation function
    generate_datasets(
        sizes=sizes,
        n_features=20,  # total amount of features 
        n_informative=10,  
        n_redundant=5,
        output_dir="datasets"
    )
    
    print("All datasets generated successfully.")
=======
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

    
    print("All datasets generated successfully.")
>>>>>>> f5f58aa2972d4ab9ab60053a3bac16d4ca39e983
