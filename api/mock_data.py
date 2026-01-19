import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

def get_mock_data(n_samples=1000):
    """
    Generates mock data for V10 and V14 to simulate credit card transactions.
    """
    # Create 3 main blobs to simulate normal transactions and some scattered anomalies
    X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2, random_state=42, cluster_std=2.5)
    
    # Create a DataFrame
    df = pd.DataFrame(X, columns=['V10', 'V14'])
    
    # Add some noise/outliers to make it look realistic distribution-wise
    noise = np.random.uniform(low=-15, high=15, size=(50, 2))
    df_noise = pd.DataFrame(noise, columns=['V10', 'V14'])
    
    df = pd.concat([df, df_noise], ignore_index=True)
    
    return df
