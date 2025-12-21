import folium
import os

def create_map(data, output):
    # Créer le dossier si on l'a oas déjà
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    map_center = [data['lat'].mean(), data['long'].mean()]
    m = folium.Map(location=map_center, zoom_start=12, tiles='Esri.WorldImagery')

    # Liste de couleur pour les clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']

    for _, row in data.iterrows():
        # Déterminer la couleur selon le cluster
        cluster = int(row['cluster']) if 'cluster' in row and row['cluster'] != -1 else -1
        color = colors[cluster % len(colors)] if cluster >= 0 else 'gray'

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
            radius=5,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)

    m.save(output)
    print(f"Carte sauvegardée sous '{output}'")