from load_data import load_data
from clean_data import convert_types, remove_duplicates, detect_anomalies
from visualization import create_map
from kmeans import kmeans_clustering
from hierarchical import hierarchical_clustering
from DBSCAN import dbscan_clustering
import matplotlib.pyplot as plt

# Chargement
data = load_data("./data/flickr_data2.csv")

# Nettoyage
data = convert_types(data)

# Dédoublonnage 
print("\n--- Dédoublonnage des données ---")
data = remove_duplicates(data)


# Détection d'anomalies (génère un CSV de synthèse)
_anomalies, anomaly_counts = detect_anomalies(data, save_path="./output/anomalies.csv")

# Afficher un tableau synthétique des anomalies (nom | count | % of total anomalies)
total_anom = len(_anomalies)
if total_anom > 0:
    max_name = max(len(n) for n in anomaly_counts.keys())
    header = f"{'Anomalie'.ljust(max_name)} | Count | %"
    sep = '-' * len(header)
    print('\nRésumé des anomalies (par type):')
    print(header)
    print(sep)
    for name, count in anomaly_counts.items():
        pct = (count / total_anom) * 100 if total_anom else 0
        print(f"{name.ljust(max_name)} | {str(count).rjust(5)} | {pct:5.1f}%")
else:
    print('\nAucune anomalie détectée.')

# Retirer les anomalies avant la visualisation et le clustering
_before = len(data)
data = data.drop(index=_anomalies.index)
_removed = _before - len(data)
print(f"Suppression anomalies: {_removed} (attendues: {len(_anomalies)})")


# Visualisation
data = data.head(10000) # Limiter à 10000 entrées pour la visualisation pas trop lente
create_map(data, output="./output/flickr_map.html")

# Calcul et visualisation avec la méthode des KMeans
data, kmeans, inertia = kmeans_clustering(data, n_clusters=50)  # Récupérer data modifiée
create_map(data, output="./output/flickr_map_kmeans.html")

# Méthode du coude pour choisir le nombre optimal de clusters pour KMeans
inertias = []
k_values = range(5, 100)  # Test k from 1 to 100

for k in k_values:
    data, kmeans, inertia = kmeans_clustering(data, k)
    inertias.append(inertia)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.savefig("./output/elbow.png")
plt.close()
print("Elbow plot saved to ./output/elbow.png")


