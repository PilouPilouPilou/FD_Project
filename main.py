from load_data import load_data
from clean_data import convert_types, analyze_missing, remove_duplicates
from visualization import create_map
from kmeans import kmeans_clustering
from hierarchical import hierarchical_clustering
from DBSCAN import dbscan_clustering

# Chargement
data = load_data("./data/flickr_data2.csv")

# Nettoyage
data = convert_types(data)
analyze_missing(data)
data = remove_duplicates(data)

# Visualisation
data = data.head(10000) # Limiter à 10000 entrées pour la visualisation pas trop lente
create_map(data, output="./output/flickr_map.html")

# Calcul et visualisation avec la méthode des KMeans
data, kmeans, inertia = kmeans_clustering(data, n_clusters=50)  # Récupérer data modifiée
create_map(data, output="./output/flickr_map_kmeans.html")

