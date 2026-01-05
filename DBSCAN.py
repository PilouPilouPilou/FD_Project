import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def dbscan_clustering(data, eps_range=None, min_samples_range=None):

    # Normaliser les noms de colonnes
    data = data.copy()
    data.columns = data.columns.str.strip()
    
    # Préparer les features (lat, long)
    X_df = data[['lat', 'long']].dropna()
    
    # Standardiser
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    
    # Paramètres par défaut
    if eps_range is None:
        eps_range = np.arange(0.1, 1.0, 0.1)
    if min_samples_range is None:
        min_samples_range = range(3, 10)

    best_score = -1
    best_eps = None
    best_min_samples = None
    results = []

    print("Parameter Tuning Results:")
    print("-" * 70)

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            
            # Compter les clusters et les noise points
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Calculer le silhouette score (only if we have valid clusters)
            if n_clusters > 1 and n_noise < len(labels) - 1:
                try:
                    score = silhouette_score(X_scaled, labels)
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette_score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_min_samples = min_samples
                except:
                    pass

    # Display top results
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('silhouette_score', ascending=False)
        print(results_df.head(10).to_string())
    else:
        print("Aucun résultat valide trouvé avec ces paramètres.")
        return data

    print("\n" + "=" * 70)
    print(f"Best Parameters Found:")
    print(f"  eps: {best_eps:.2f}")
    print(f"  min_samples: {best_min_samples}")
    print(f"  Silhouette Score: {best_score:.4f}")
    print("=" * 70 + "\n")

    # Apply DBSCAN with best parameters
    dbscan_final = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    dbscan_labels = dbscan_final.fit_predict(X_scaled)

    # Print results
    n_clusters_final = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise_final = list(dbscan_labels).count(-1)

    print(f"DBSCAN Clustering Results (eps={best_eps}, min_samples={best_min_samples}):")
    print(f"  Number of Clusters: {n_clusters_final}")
    print(f"  Number of Noise Points: {n_noise_final}")
    print(f"  Number of Points in Clusters: {len(dbscan_labels) - n_noise_final}")
    print(f"\nCluster Distribution:")

    for cluster_id in sorted(set(dbscan_labels)):
        count = list(dbscan_labels).count(cluster_id)
        if cluster_id == -1:
            print(f"  Noise Points: {count}")
        else:
            print(f"  Cluster {cluster_id}: {count} points")

    # Add DBSCAN labels au DataFrame (aligner avec les index valides)
    data['cluster_dbscan'] = -1
    data.loc[X_df.index, 'cluster_dbscan'] = dbscan_labels

    return data