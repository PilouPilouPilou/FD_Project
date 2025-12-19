# pip install --user -U nltk
# pip install folium
import nltk
import folium
import pandas as pd

# Importer le fichier CSV dans un DataFrame pandas
data = pd.read_csv("./flickr_data2.csv", low_memory=False)

# Afficher les premières lignes du dataset
print(data.head())
print(f"\nDimensions du dataset: {data.shape}")
print(f"\nColonnes: {data.columns.tolist()}")