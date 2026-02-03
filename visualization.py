import folium
import os
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def create_map(data, output, top_terms=None, top_ngrams=None):
    # Créer le dossier s'il existe pas
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("\nCréation de la carte...")

    # Normaliser les noms de colonnes (car certains espaces existent)
    data = data.copy()
    data.columns = data.columns.str.strip()

    map_center = [data['lat'].mean(), data['long'].mean()]
    m = folium.Map(location=map_center, zoom_start=12, tiles='Esri.WorldImagery') # Utiliser une carte satellite avec l'option Tiles, plein d'autres sont possibles

    # Générer une palette de couleurs distinctes basée sur le nombre de clusters
    if 'cluster' in data.columns:
        unique_clusters = sorted([c for c in data['cluster'].unique() if c != -1])
        n_clusters = len(unique_clusters)
        
        if n_clusters > 0:
            # Utiliser une colormap avec des couleurs bien distinctes
            colormap_name = 'tab20' if n_clusters <= 20 else 'hsv'
            colormap = plt.colormaps.get_cmap(colormap_name).resampled(n_clusters)
            colors = {}
            for i, cluster_id in enumerate(unique_clusters):
                rgb = colormap(i)[:3]  # Récupérer RGB sans alpha
                colors[cluster_id] = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                )
        else:
            colors = {}
    else:
        colors = {}

    # Tracer périmètre (convex hull) par cluster AVANT les points pour ne pas les recouvrir
    if 'cluster' in data.columns:
        for cluster_id in sorted([c for c in data['cluster'].unique() if c != -1]): 
            cluster_df = data.loc[data['cluster'] == cluster_id, ['lat', 'long']]
            if cluster_df.shape[0] == 0: 
                continue
            coords = list(zip(cluster_df['long'].values, cluster_df['lat'].values))  # shapely : (x=lon, y=lat)
            if len(coords) >= 3:
                hull = MultiPoint(coords).convex_hull # Convex hull : un polygone englobant tous les points 
                if hull.geom_type == 'Polygon':
                    poly_coords = [[y, x] for x, y in hull.exterior.coords]  # folium : [lat, lon]
                    color = colors.get(cluster_id, 'gray')
                    folium.Polygon(
                        locations=poly_coords,
                        color=color,
                        weight=2,
                        opacity=0.8,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.15
                    ).add_to(m)
            else:
                # pour 1 ou 2 points, dessiner un cercle centré
                lat_mean = float(cluster_df['lat'].mean())
                lon_mean = float(cluster_df['long'].mean())
                folium.Circle(
                    location=[lat_mean, lon_mean],
                    radius=100,  
                    color=colors.get(cluster_id, 'gray'),
                    fill=True,
                    fill_opacity=0.15
                ).add_to(m)

    # Ajouter les points
    for _, row in data.iterrows():

        # Appliquer une couleur selon le cluster (sinon gris)
        cluster = int(row['cluster']) if 'cluster' in row and row['cluster'] != -1 else -1
        color = colors.get(cluster, 'gray')

        # Formater les dates
        date_taken_str = row['datetime_taken'] if pd.notna(row['datetime_taken']) else 'N/A'
        date_upload_str = row['datetime_upload'] if pd.notna(row['datetime_upload']) else 'N/A'

        # Popup avec infos
        popup_text = f"""
        <b>Cluster:</b> {cluster}<br>
        <b>User:</b> {row['user']}<br>
        <b>Title:</b> {row['title']}<br>
        <b>Tags:</b> {row['tags']}<br>
        <b>Date Taken:</b> {date_taken_str}<br>
        <b>Date Upload:</b> {date_upload_str}<br>
        """
        
        # Ajouter les top termes du cluster si disponibles
        if top_terms and cluster in top_terms:
            top_words_list = top_terms[cluster]
            if top_words_list:
                words_str = ", ".join([f"{w} ({c})" for w, c in top_words_list])
                popup_text += f"<b>Top mots:</b> {words_str}<br>"

        # Ajouter les n-grams pertinents (bigrammes/trigrammes) si fournis
        if top_ngrams and cluster in top_ngrams:
            ngram_info = top_ngrams[cluster]
            bigrams = ngram_info.get('bigrams', [])
            trigrams = ngram_info.get('trigrams', [])
            if bigrams:
                bigram_str = ", ".join([f"{ng} ({cnt}x)" for ng, cnt in bigrams[:3]])
                popup_text += f"<b>Bigrammes:</b> {bigram_str}<br>"
            if trigrams:
                trigram_str = ", ".join([f"{ng} ({cnt}x)" for ng, cnt in trigrams[:2]])
                popup_text += f"<b>Trigrammes:</b> {trigram_str}<br>"

        popup_text += f"<a href=\"{row['url']}\" target=\"_blank\">🔗 Voir la photo sur Flickr</a>"

        folium.CircleMarker(
            location=[row['lat'], row['long']],
            popup=folium.Popup(popup_text, max_width=300),
            radius=4,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8
        ).add_to(m)

    m.save(output)
    print(f"Carte sauvegardée sous '{output}'")