import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def dbscan_clustering(data, eps_km_range=None, min_samples_range=None, min_cluster_size=5):
    data = data.copy()
    data.columns = data.columns.str.strip()
    
    X_df = data[['lat', 'long']].dropna()
    print(f"Points à clusteriser: {len(X_df)}")
    
    if len(X_df) < 10:
        print("Pas assez de points pour clustering!")
        return data
    
    # Convertir en radians pour metric haversine
    X_rad = np.radians(X_df.values)
    
    # Estimer l'étendue des données (km approximatif)
    lat_range = X_df['lat'].max() - X_df['lat'].min()
    long_range = X_df['long'].max() - X_df['long'].min()
    max_range_km = max(lat_range, long_range) * 111  # approx
    
    print(f"Étendue géographique: ~{max_range_km:.1f} km")
    
    # Paramètres par défaut
    if eps_km_range is None:
        eps_km_range = np.arange(0.04, 0.14, 0.02)
    
    # commencer min_samples à 3 pour éviter trop de clusters unitaires
    if min_samples_range is None:
        min_samples_range = np.arange(10, 80, 10)
    
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
                    
                    # composite score — favorise silhouette & faible noise, pénalise clusters trop petits
                    # ajuster coefficients si besoin
                    composite_score = (score_sil * 2.0 + (1 - noise_ratio)) * np.log1p(n_clusters)
                    composite_score -= (tiny_ratio * 5.0)  # pénalité forte si beaucoup de très petits clusters
                    
                    results.append({
                        'eps_km': eps_km,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_%': noise_ratio * 100,
                        'silhouette': score_sil,
                        'tiny_cluster_ratio': tiny_ratio,
                        'avg_size': avg_size,
                        'composite': composite_score
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
        print("\nAUCUN RÉSULTAT VALIDE!")
        print("Suggestions: augmenter eps_km_range / réduire min_cluster_size / vérifier densité des points")
        return data
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('silhouette', ascending=False)
    
    print("\n" + "=" * 90)
    print("TOP 15 CONFIGURATIONS:")
    print("=" * 90)
    print(results_df.head(15).to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    
    print("\n" + "=" * 90)
    print("MEILLEURE CONFIGURATION:")
    print(f"  eps: {best_config['eps_km']:.2f} km")
    print(f"  min_samples: {best_config['min_samples']}")
    print(f"  Clusters attendus: {best_config['n_clusters']}")
    print(f"  Noise: {best_config['noise_ratio']*100:.1f}%")
    print(f"  Tiny cluster ratio: {best_config['tiny_ratio']:.3f}")
    print(f"  Silhouette: {best_config['silhouette']:.4f}")
    print("=" * 90 + "\n")
    
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
    
    data['cluster'] = -1
    data.loc[X_df.index, 'cluster'] = dbscan_labels
    
    return data
