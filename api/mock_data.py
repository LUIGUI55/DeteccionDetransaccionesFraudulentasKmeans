def get_mock_data(n_samples=1000):
    """
    Generates mock data for V10, V14, Amount, and Class to simulate credit card transactions.
    """
    # Create 3 main blobs to simulate normal transactions (Class 0)
    X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2, random_state=42, cluster_std=2.5)
    
    # Create a DataFrame
    df = pd.DataFrame(X, columns=['V10', 'V14'])
    df['Amount'] = np.random.exponential(scale=100, size=n_samples)
    df['Class'] = 0 # Normal transactions
    
    # Add some noise/outliers (Class 1 - Fraud)
    noise_samples = 50
    noise = np.random.uniform(low=-15, high=15, size=(noise_samples, 2))
    df_noise = pd.DataFrame(noise, columns=['V10', 'V14'])
    df_noise['Amount'] = np.random.exponential(scale=500, size=noise_samples) 
    df_noise['Class'] = 1 # Fraudulent transactions
    
    df = pd.concat([df, df_noise], ignore_index=True)
    
    return df
