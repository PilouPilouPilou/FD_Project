# pip install --user -U nltk
# pip install folium
import nltk
import folium
import pandas as pd

# Importer le fichier CSV dans un DataFrame pandas
data = pd.read_csv("./flickr_data2.csv", low_memory=False)

# Supprimer les colonnes vides et les colonnes "Unnamed"
data = data.loc[:, ~data.columns.str.contains('Unnamed', na=False)]

# Afficher les premières lignes du dataset
print(data.head())
print(f"\nDimensions du dataset: {data.shape}")
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
