import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import io
import base64
from .mock_data import get_mock_data
import os

def perform_fraud_detection(n_clusters=6):
    # Try to load real data, otherwise use mock
    csv_path = '/home/luisantonio/Escritorio/deploy7MO/creditcard.csv'
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Ensure required columns exist, fill with mock if missing for robustness
            required_cols = ["V10", "V14", "Amount", "Class"]
            if not all(col in df.columns for col in required_cols):
                 X = get_mock_data()[required_cols]
            else:
                 X = df[required_cols].copy()
        except Exception:
             X = get_mock_data()[["V10", "V14", "Amount", "Class"]]
    else:
        X = get_mock_data()[["V10", "V14", "Amount", "Class"]]

    # K-Means with dynamic clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X[["V10", "V14"]]) # clustering on V10/V14 only as per original logic
    
    X['Cluster'] = clusters

    # --- Statistics Calculation ---
    inertia = kmeans.inertia_
    n_iter = kmeans.n_iter_
    centroids = kmeans.cluster_centers_
    predicted_labels_sample = clusters[:20].tolist()

    # Calculate counts and fraud counts per cluster
    from collections import Counter
    cluster_counts = Counter(clusters)
    
    # Calculate fraud counts (where Class == 1)
    fraud_counts = X[X['Class'] == 1].groupby('Cluster').size().to_dict()
    
    stats_table = []
    for i in range(n_clusters):
        total = cluster_counts.get(i, 0)
        frauds = fraud_counts.get(i, 0)
        stats_table.append({
            'cluster': i,
            'total': total,
            'frauds': frauds
        })

    # --- Graph 1: Main K-Means Plot ---
    plt.figure(figsize=(10, 6))
    h = .02
    x_min, x_max = X['V10'].min() - 1, X['V10'].max() + 1
    y_min, y_max = X['V14'].min() - 1, X['V14'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Pastel2,
               aspect='auto', origin='lower')

    plt.scatter(X['V10'], X['V14'], c=clusters, s=5, cmap='viridis', alpha=0.5)
    centroids_plot = kmeans.cluster_centers_ # Use for plotting
    plt.scatter(centroids_plot[:, 0], centroids_plot[:, 1],
                marker='x', s=169, linewidths=3,
                color='red', zorder=10, label='Centroids')

    plt.title(f'K-Means: Detección de Transacciones ({n_clusters} Clusters)')
    plt.xlabel('V10')
    plt.ylabel('V14')
    plt.legend()
    plt.tight_layout()
    main_graph = get_base64_image(plt)

    # --- Graph 2: Cluster Distribution (Count) ---
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Cluster', data=X, palette='viridis')
    plt.title('Distribución de Transacciones por Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Cantidad de Transacciones')
    plt.tight_layout()
    dist_graph = get_base64_image(plt)

    # --- Graph 3: Amount Distribution by Cluster (Boxplot) ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='Amount', data=X, palette='coolwarm')
    plt.title('Distribución de Montos por Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Monto de Transacción')
    plt.yscale('log') # Log scale to handle large outliers better
    plt.tight_layout()
    amount_graph = get_base64_image(plt)
    
    # Format centroids for display (avoid template filter issues)
    formatted_centroids = []
    for c in centroids:
        formatted_centroids.append([f"{c[0]:.4f}", f"{c[1]:.4f}"])

    return {
        'main': main_graph,
        'distribution': dist_graph,
        'amount': amount_graph,
        'n_clusters': n_clusters,
        'inertia': inertia,
        'n_iter': n_iter,
        'centroids': formatted_centroids,
        'predicted_labels_sample': predicted_labels_sample,
        'stats_table': stats_table
    }

def get_base64_image(plt_obj):
    buffer = io.BytesIO()
    plt_obj.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt_obj.close()
    return base64.b64encode(image_png).decode('utf-8')
