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

# data = data.head(100) A ACTIVER SI ON VEUT VISUALISER LA MAP


###### CONVERSION DES TYPES DE DONNÉES ##########
# Convertir les colonnes numériques en int ou float
numeric_columns = ['id', 'lat', 'long', 
                   'date_taken_minute', 'date_taken_hour', 'date_taken_day', 
                   'date_taken_month', 'date_taken_year',
                   'date_upload_minute', 'date_upload_hour', 'date_upload_day', 
                   'date_upload_month', 'date_upload_year']

for col in numeric_columns:
    if col in data.columns:
        # Convertir en float d'abord (gère les décimales), puis en int si pas de décimales
        data[col] = pd.to_numeric(data[col], errors='coerce')
        # Si la colonne n'a pas de NaN après conversion, on peut la passer en int
        if data[col].isna().sum() == 0 and data[col].dtype == 'float64' and (data[col] % 1 == 0).all():
            data[col] = data[col].astype('int64')

print("Types de données convertis:")
print(data.dtypes)

###### ANALYSE DES VALEURS MANQUANTES ##########
print("\n--- Analyse des valeurs manquantes (NaN) ---")
print("\nNombre de NaN par colonne:")
missing_data = data.isna().sum()
print(missing_data[missing_data > 0])

print(f"\nPourcentage de NaN par colonne:")
missing_percent = (data.isna().sum() / len(data)) * 100
print(missing_percent[missing_percent > 0].round(2))

# Identifier les lignes avec des valeurs manquantes dans les dates
date_columns = ['date_taken_minute', 'date_taken_hour', 'date_taken_day', 'date_taken_month', 'date_taken_year',
                'date_upload_minute', 'date_upload_hour', 'date_upload_day', 'date_upload_month', 'date_upload_year']
rows_with_missing_dates = data[data[date_columns].isna().any(axis=1)]

print(f"\nNombre de lignes avec au moins une date manquante: {len(rows_with_missing_dates)}")
print("\nExemples de lignes avec des dates manquantes:")
print(rows_with_missing_dates[['id', 'user', 'title'] + date_columns].head(10))

###### DUPLICATION ##########
# Vérifier s'il y a des duplicats
data.duplicated()
print(f"\nInitial: {len(data)}")
print("Nombre de duplicats :", data.duplicated().sum())

# Supprimer les duplicats complets
data_cleaned = data.drop_duplicates(keep='first')
print(f"After removing duplicates: {len(data_cleaned)}")
print(data.info())  # column names and data types

# Créer une carte
# On centre la carte sur une lat et long moyennes
map_center = [data['lat'].mean(), data['long'].mean()]

map_init = folium.Map(location=map_center, zoom_start=12)

for idx, row in data.iterrows():
    folium.Marker(
        location=[row['lat'], row['long']],
        popup=row['title'],
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(map_init)

# Sauvegarder la carte pour la regarder après
map_init.save("flickr_map.html")
print("Carte sauvegardée sous 'flickr_map.html'")
