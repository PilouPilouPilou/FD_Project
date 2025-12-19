import folium

def create_map(data, output="./output/flickr_map.html"):
    map_center = [data['lat'].mean(), data['long'].mean()]
    m = folium.Map(location=map_center, zoom_start=12, tiles='Esri.WorldImagery')

    for _, row in data.iterrows():
        popup_text = f"""
        <b>User:</b> {row['user']}<br>
        <b>Tags:</b> {row['tags']}<br>
        <b>Title:</b> {row['title']}<br>
        <b>Date Taken:</b> {row['date_taken_year']}-{row['date_taken_month']}-{row['date_taken_day']}
        """

        folium.Marker(
            location=[row['lat'], row['long']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color='blue')
        ).add_to(m)

    m.save(output)
    print(f"Carte sauvegardée sous '{output}'")
