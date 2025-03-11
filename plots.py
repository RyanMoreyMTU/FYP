import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

tf_files = glob.glob('tensorflow_gpu_results_batch*.csv')
sk_files = glob.glob('sklearn_cpu_results_batch*.csv')

tf_dfs = []
for file in tf_files:
    df = pd.read_csv(file)
    if 'Batch Size' not in df.columns:
        batch_size = int(file.split('batch')[1].split('.')[0])
        df['Batch Size'] = batch_size
    tf_dfs.append(df)

sk_dfs = []
for file in sk_files:
    df = pd.read_csv(file)
    if 'Batch Size' not in df.columns:
        batch_size = int(file.split('batch')[1].split('.')[0])
        df['Batch Size'] = batch_size
    sk_dfs.append(df)

tf_results = pd.concat(tf_dfs) if tf_dfs else pd.DataFrame()
sk_results = pd.concat(sk_dfs) if sk_dfs else pd.DataFrame()

if not tf_results.empty:
    tf_results['Size'] = tf_results['Dataset'].str.extract(r'data_(\d+)').astype(int)
if not sk_results.empty:
    sk_results['Size'] = sk_results['Dataset'].str.extract(r'data_(\d+)').astype(int)

tf_agg = tf_results.groupby(['Size', 'Batch Size']).agg({
    'Training Time (s)': 'mean',
    'Inference Time (s)': 'mean',
    'Test Accuracy': 'mean'
}).reset_index()

sk_agg = sk_results.groupby(['Size', 'Batch Size']).agg({
    'Training Time (s)': 'mean',
    'Inference Time (s)': 'mean',
    'Test Accuracy': 'mean'
}).reset_index()

tf_agg['Model'] = 'TensorFlow (GPU)'
sk_agg['Model'] = 'Scikit-learn (CPU)'

combined = pd.concat([tf_agg, sk_agg])

plt.figure(figsize=(12, 8))
for batch_size in combined['Batch Size'].unique():
    plt.subplot(1, 3, list(combined['Batch Size'].unique()).index(batch_size) + 1)
    batch_data = combined[combined['Batch Size'] == batch_size]
    sns.lineplot(data=batch_data, x='Size', y='Training Time (s)', hue='Model', marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Training Time (Batch Size: {batch_size})')
    plt.xlabel('Dataset Size')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_comparison.png')
plt.show()

plt.figure(figsize=(12, 8))
for batch_size in combined['Batch Size'].unique():
    plt.subplot(1, 3, list(combined['Batch Size'].unique()).index(batch_size) + 1)
    batch_data = combined[combined['Batch Size'] == batch_size]
    sns.lineplot(data=batch_data, x='Size', y='Inference Time (s)', hue='Model', marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Inference Time (Batch Size: {batch_size})')
    plt.xlabel('Dataset Size')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('inference_comparison.png')
plt.show()
