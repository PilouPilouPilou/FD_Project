from load_data import load_data
from visualization import create_map
from kmeans import kmeans_clustering
from hierarchical import hierarchical_clustering
from DBSCAN import dbscan_clustering
import matplotlib.pyplot as plt
from text_pattern_mining import preprocess_texts, calcule_top_terms

# Chargement des données nettoyées
data = load_data("./data/cleaned_flickr_data.csv")


# Visualisation
data = data.head(3000) # Limiter à 10000 entrées pour la visualisation pas trop lente
create_map(data, output="./output/flickr_map.html")

# Calcul et visualisation avec la méthode des KMeans
data_kmeans, kmeans, inertia = kmeans_clustering(data, n_clusters=50)  # Récupérer data modifiée
create_map(data_kmeans, output="./output/flickr_map_kmeans.html")

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

# Calcul et visualisation avec la méthode DBSCAN
data_dbscan = dbscan_clustering(data, None, None)
create_map(data_dbscan, output="./output/flickr_map_dbscan.html")

# Clustering hiérarchique agglomératif
data_hierarchical, hier_results = hierarchical_clustering(data, n_clusters=56)


# TEXT PATTERN MINING 
cluster_texts = preprocess_texts(data_kmeans)
top_terms = calcule_top_terms(cluster_texts)

# Ajouter les top termes à la carte KMeans
create_map(data_kmeans, output="./output/flickr_map_kmeans_with_terms.html", top_terms=top_terms)

# TEXT PATTERN MINING 
cluster_texts_db = preprocess_texts(data_dbscan)
top_terms = calcule_top_terms(cluster_texts_db)

# Ajouter les top termes à la carte DBSCAN
create_map(data_dbscan, output="./output/flickr_map_dbscan_with_terms.html", top_terms=top_terms)
