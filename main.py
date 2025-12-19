# pip install --user -U nltk
# pip install folium
import nltk
import folium
import pandas as pd

# Importer le fichier CSV dans un DataFrame pandas
data = pd.read_csv("./flickr_data2.csv", low_memory=False)
data.columns = data.columns.str.strip() # ça permet de supprimer les espaces trop bizarres au début des colonnes


# Supprimer les colonnes vides et les colonnes "Unnamed"
data = data.loc[:, ~data.columns.str.contains('Unnamed', na=False)]

data = data.head(100) # A ACTIVER SI ON VEUT VISUALISER LA MAP

# Afficher les premières lignes du dataset
print(data.head())
print(f"\nDimensions du dataset: {data.shape}")
print(f"\nColonnes: {data.columns.tolist()}")

print(f"Nombre de lignes: {len(data)}")



###### DUPLICATION ##########
# Vérifier s'il y a des duplicats
data.duplicated()
print(f"Initial: {len(data)}")
print("Nombre de duplicats :", data.duplicated().sum())

# Supprimer les duplicats complets
data_cleaned = data.drop_duplicates(keep='first')
print(f"After removing duplicates: {len(data_cleaned)}")
print(data.info())  # column names and data types

data_cleaned.describe()

# Créer une carte
# On centre la carte sur une lat et long moyennes
map_center = [data['lat'].mean(), data['long'].mean()]

map_init = folium.Map(location=map_center, zoom_start=12, tiles='Esri.WorldImagery') # On change le fond de carte sur l'option Tiles

for idx, row in data.iterrows():

    popup_text = f"""
    <b>User:</b> {row['user']}<br>
    <b>Tags:</b> {row['tags']}<br>
    <b>Title:</b> {row['title']}<br>
    <b>Date Taken:</b> {row['date_taken_year']}-{row['date_taken_month']}-{row['date_taken_day']}
    """

    folium.Marker(
        location=[row['lat'], row['long']],
        radius=1,
        popup=folium.Popup(popup_text, max_width=300),
        color='blue', 
        fill=True,
        fillColor='blue',
        fillOpacity=0.6
    ).add_to(map_init)

# Sauvegarder la carte pour la regarder après
map_init.save("flickr_map.html")
print("Carte sauvegardée sous 'flickr_map.html'")
