import folium
import os
import numpy as np
from shapely.geometry import MultiPoint, Point

def create_map(data, output):
    # Créer le dossier s'il existe pas
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Normaliser les noms de colonnes (car certains espaces existent)
    data = data.copy()
    data.columns = data.columns.str.strip()

    map_center = [data['lat'].mean(), data['long'].mean()]
    m = folium.Map(location=map_center, zoom_start=12, tiles='Esri.WorldImagery') # Utiliser une carte satellite avec l'option Tiles, plein d'autres sont possibles

    # Liste de couleurs pour les clusters (à ajouter d'autres peut-être)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']

    # Ajouter les points
    for _, row in data.iterrows():

        # Appliquer une couleur selon le cluster (sinon gris)
        cluster = int(row['cluster']) if 'cluster' in row and row['cluster'] != -1 else -1
        color = colors[cluster % len(colors)] if cluster >= 0 else 'gray'

        # Popup avec infos
        popup_text = f"""
        <b>Cluster:</b> {cluster}<br>
        <b>User:</b> {row['user']}<br>
        <b>Tags:</b> {row['tags']}<br>
        <b>Title:</b> {row['title']}<br>
        <b>Date Taken:</b> {row['date_taken_year']}-{row['date_taken_month']}-{row['date_taken_day']}
        """

        folium.CircleMarker(
            location=[row['lat'], row['long']],
            popup=folium.Popup(popup_text, max_width=300),
            radius=4,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8
        ).add_to(m)

    # Tracer périmètre (convex hull) par cluster si la colonne 'cluster' existe
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
                    color = colors[int(cluster_id) % len(colors)]
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
                    color=colors[int(cluster_id) % len(colors)],
                    fill=True,
                    fill_opacity=0.15
                ).add_to(m)

    m.save(output)
    print(f"Carte sauvegardée sous '{output}'")