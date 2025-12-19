# pip install --user -U nltk
# pip install folium
import nltk
import folium
import pandas as pd

# Importer le fichier CSV dans un DataFrame pandas
data = pd.read_csv("./flickr_data2.csv", low_memory=False)
data.columns = data.columns.str.strip() # ça permet de supprimer les espaces trop bizarres au début des colonnes


# Afficher les premières lignes du dataset
print(data.head())
print(f"\nDimensions du dataset: {data.shape}")
print(f"\nColonnes: {data.columns.tolist()}")

# Créer une carte
# On centre la carte sur une lat et long moyennes
map_center = [data['lat'].mean(), data['long'].mean()]

map_init = folium.Map(location=map_center, zoom_start=2)

for idx, row in data.iterrows():
    folium.Marker(
        location=[row['lat'], row['long']],
        popup=row['title'],
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(map_init)

# Sauvegarder la carte pour la regarder après
map_init.save("flickr_map.html")
print("Carte sauvegardée sous 'flickr_map.html'")