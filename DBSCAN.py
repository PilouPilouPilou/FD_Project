import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def dbscan_clustering(data, eps_km_range=None, min_samples_range=None, min_cluster_size=5):
    data = data.copy()
    data.columns = data.columns.str.strip()
    
    X_df = data[['lat', 'long']].dropna()
    print(f"Points à clusteriser: {len(X_df)}")
    
    # Convertir en radians pour metric haversine
    X_rad = np.radians(X_df.values)
    
    # Estimer l'étendue des données (km approximatif)
    lat_range = X_df['lat'].max() - X_df['lat'].min()
    long_range = X_df['long'].max() - X_df['long'].min()
    max_range_km = max(lat_range, long_range) * 111  # approx
    
    print(f"Étendue géographique: ~{max_range_km:.1f} km")
    
    # Paramètres par défaut
    if eps_km_range is None:
        eps_km_range = np.arange(0.14, 0.19, 0.02)
    
    # commencer min_samples à 3 pour éviter trop de clusters unitaires
    if min_samples_range is None:
        min_samples_range = np.arange(30, 90, 10)
    
    print(f"\nTest de {len(eps_km_range)} valeurs d'eps × {len(min_samples_range)} valeurs de min_samples")
    print(f"eps range: {eps_km_range[0]:.2f} à {eps_km_range[-1]:.2f} km")
    print(f"min_samples range: {min(min_samples_range)} à {max(min_samples_range)}")
    print("\nRecherche en cours...\n")
    
    best_config = None
    best_score = -np.inf
    results = []
    tested = 0
    
    for eps_km in eps_km_range:
        eps_rad = eps_km / 6371.0
        for min_samples in min_samples_range:
            tested += 1
            try:
                dbscan = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
                labels = dbscan.fit_predict(X_rad)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)
                
                if n_clusters >= 1:
                    # silhouette only if >1 cluster
                    if n_clusters > 1 and noise_ratio < 0.95:
                        try:
                            score_sil = silhouette_score(X_rad, labels, metric='haversine')
                        except:
                            score_sil = 0.0
                    else:
                        score_sil = 0.0
                    
                    # cluster sizes diagnostics
                    cluster_sizes = [list(labels).count(c) for c in set(labels) if c != -1]
                    if cluster_sizes:
                        tiny_count = sum(1 for s in cluster_sizes if s < min_cluster_size)
                        tiny_ratio = tiny_count / max(1, len(cluster_sizes))
                        avg_size = np.mean(cluster_sizes)
                    else:
                        tiny_ratio = 1.0
                        avg_size = 0.0
                    
                    
                    results.append({
                        'eps_km': eps_km,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_%': noise_ratio * 100,
                        'silhouette': score_sil,
                        'tiny_cluster_ratio': tiny_ratio,
                        'avg_size': avg_size,
                    })
                    
                    if score_sil > best_score:
                        best_score = score_sil
                        best_config = {
                            'eps_km': eps_km,
                            'eps_rad': eps_rad,
                            'min_samples': min_samples,
                            'silhouette': score_sil,
                            'n_clusters': n_clusters,
                            'noise_ratio': noise_ratio,
                            'tiny_ratio': tiny_ratio
                        }
            except Exception as e:
                # garde le debug mais continue
                print(f"Erreur avec eps={eps_km:.2f}, min_samples={min_samples}: {e}")
    
    print(f"Configurations testées: {tested}")
    print(f"Configurations valides: {len(results)}")
    
    if not results:
        print("\nPas de résultats valides obtenus.")
        return data
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('silhouette', ascending=False)
    
    print("TOP 5 CONFIGURATIONS:")
    print(results_df.head(5).to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    
    print("MEILLEURE CONFIGURATION:")
    print(f"  eps: {best_config['eps_km']:.2f} km")
    print(f"  min_samples: {best_config['min_samples']}")
    print(f"  Clusters attendus: {best_config['n_clusters']}")
    print(f"  Noise: {best_config['noise_ratio']*100:.1f}%")
    print(f"  Tiny cluster ratio: {best_config['tiny_ratio']:.3f}")
    print(f"  Silhouette: {best_config['silhouette']:.4f}")
    
    # Appliquer le meilleur modèle
    dbscan_final = DBSCAN(eps=best_config['eps_rad'], min_samples=best_config['min_samples'], metric='haversine')
    dbscan_labels = dbscan_final.fit_predict(X_rad)
    
    # Statistiques finales
    n_clusters_final = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise_final = list(dbscan_labels).count(-1)
    
    print(f"RÉSULTATS FINAUX:")
    print(f"  Clusters: {n_clusters_final}")
    print(f"  Noise: {n_noise_final} ({n_noise_final/len(dbscan_labels)*100:.1f}%)")
    print(f"  Points clusterisés: {len(dbscan_labels) - n_noise_final}")
    
    cluster_sizes = [list(dbscan_labels).count(i) for i in set(dbscan_labels) if i != -1]
    if cluster_sizes:
        print(f"\nTailles des clusters: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.1f}")
    
    # POST-TRAITEMENT (hierarchical DBScan): subdiviser les clusters trop gros, avec nos paramètres
    dbscan_labels = refine_large_clusters(dbscan_labels, X_rad, max_cluster_size=(len(dbscan_labels) - n_noise_final)*0.05,eps_km_refine=0.035,min_samples_refine=40)
    
    data['cluster'] = -1
    data.loc[X_df.index, 'cluster'] = dbscan_labels
    
    return data


def refine_large_clusters(labels, X_rad, max_cluster_size=150, eps_km_refine=0.01, min_samples_refine=5):

    labels = labels.copy()
    next_cluster_id = max([l for l in labels if l != -1] or [0]) + 1
    eps_rad_refine = eps_km_refine / 6371.0
    
    print(f"\n{'='*90}")
    print(f"POST-TRAITEMENT: Raffinement des gros clusters")
    print(f"{'='*90}")
    print(f"Max cluster size: {max_cluster_size}")
    print(f"eps_km: {eps_km_refine}, min_samples: {min_samples_refine}\n")
    
    large_clusters = []
    for cluster_id in sorted([c for c in set(labels) if c != -1]):
        cluster_mask = (labels == cluster_id)
        cluster_size = np.sum(cluster_mask)
        if cluster_size > max_cluster_size:
            large_clusters.append((cluster_id, cluster_size))
    
    if not large_clusters:
        print("Aucun cluster trop gros détecté\n")
        return labels
    
    print(f"Clusters trop gros trouvés: {len(large_clusters)}\n")
    
    for cluster_id, size in large_clusters:
        print(f"Raffinement du Cluster {cluster_id} ({size} points)...")
        
        # Extraire les points du cluster
        cluster_mask = (labels == cluster_id)
        cluster_points = X_rad[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        # Appliquer DBSCAN fin sur ce cluster
        dbscan_refine = DBSCAN(eps=eps_rad_refine, min_samples=min_samples_refine, metric='haversine')
        sub_labels = dbscan_refine.fit_predict(cluster_points)
        
        # Compter les sous-clusters créés
        n_subclusters = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
        
        if n_subclusters <= 1:
            # Pas de subdivision possible, garder comme avant
            print(f"  → Pas de subdivision possible, cluster conservé\n")
            continue
        
        # Assigner les nouveaux IDs aux sous-clusters
        for sub_label in set(sub_labels):
            if sub_label == -1:  # Le bruit garde l'ID du cluster parent (périmètre)
                sub_mask = (sub_labels == sub_label)
                labels[cluster_indices[sub_mask]] = cluster_id
            else:
                sub_mask = (sub_labels == sub_label)
                labels[cluster_indices[sub_mask]] = next_cluster_id
                next_cluster_id += 1
        
        print(f"  → Divisé en {n_subclusters} sous-clusters (IDs {next_cluster_id - n_subclusters} à {next_cluster_id - 1})\n")
    
    return labels
