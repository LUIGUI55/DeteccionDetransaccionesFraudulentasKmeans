import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import io
import base64
from .mock_data import get_mock_data
import os

def perform_fraud_detection():
    # Try to load real data, otherwise use mock
    csv_path = '/home/luisantonio/Escritorio/deploy7MO/creditcard.csv'
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            X = df[["V10", "V14"]].copy()
        except Exception:
             X = get_mock_data()[["V10", "V14"]]
    else:
        X = get_mock_data()[["V10", "V14"]]

    # K-Means with 6 clusters as requested (previously 5)
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Add cluster info to X for plotting convenience (optional, mostly for internal debug if needed)
    X['Cluster'] = clusters

    # Generate Graph
    plt.figure(figsize=(12, 6))
    
    # Plot decision boundaries
    h = .02
    x_min, x_max = X['V10'].min() - 1, X['V10'].max() + 1
    y_min, y_max = X['V14'].min() - 1, X['V14'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict for the meshgrid to show boundaries
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Pastel2,
               aspect='auto', origin='lower')

    # Plot data points
    plt.scatter(X['V10'], X['V14'], c=clusters, s=5, cmap='viridis', alpha=0.5)
    
    # Plot centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='red', zorder=10, label='Centroids')

    plt.title(f'K-Means: Detección de Transacciones Bancarias Fraudulentas ({n_clusters} Clusters)')
    plt.xlabel('V10')
    plt.ylabel('V14')
    plt.legend()
    plt.tight_layout()

    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    
    return graphic, n_clusters
