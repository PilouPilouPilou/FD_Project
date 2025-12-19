from load_data import load_data
from clean_data import convert_types, analyze_missing, remove_duplicates
from visualization import create_map
from kmeans import kmeans_clustering

# Chargement
data = load_data("./data/flickr_data2.csv")

# Nettoyage
data = convert_types(data)
analyze_missing(data)
data = remove_duplicates(data)

# Visualisation
data = data.head(100) # Limiter à 100 entrées pour la visualisation
kmeans_clustering(data, n_clusters=5)
create_map(data, output="./output/flickr_map.html")
