from load_data import load_data
from clean_data import convert_types, analyze_missing, remove_duplicates, detect_anomalies
from visualization import create_map
from kmeans import kmeans_clustering
from hierarchical import hierarchical_clustering
from DBSCAN import dbscan_clustering
import matplotlib.pyplot as plt

# Chargement
data = load_data("./data/flickr_data2.csv")

# Nettoyage
data = convert_types(data)

# Détection d'anomalies (génère un CSV de synthèse)
_anomalies = detect_anomalies(data, save_path="./output/anomalies.csv")

# analyze_missing(data)
data = remove_duplicates(data)

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


