import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def kmeans_clustering(data, n_clusters):
    # Sélectionner les colonnes avec long et lat pour le clustering
    data_clustering = data[['lat', 'long']].dropna()

    # Standardiser les données
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_clustering)
    print(scaled_data)

    scaled_data_df = pd.DataFrame(data=scaled_data, columns=data_clustering.columns)
    scaled_data_df.head()

    # Appliquer KMeans
    kmeans = KMeans(n_clusters=n_clusters, init='kmeans++')
    kmeans.fit(scaled_data_df)

    # Ajouter les labels de cluster au DataFrame original
    data['cluster'] = kmeans.labels_

    inertia = kmeans.inertia_
    print(f"Sum of squared distances: {inertia}")

    return data, kmeans, inertia