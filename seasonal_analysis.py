import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import html
import visualization
import text_pattern_mining

def load_and_prepare_data(filepath, n_samples=None):
    """
    Charge les donnees et prepare les colonnes temporelles
    """
    data = pd.read_csv(filepath)
    
    # Convertir en datetime si ce n'est pas deja fait
    if 'datetime_taken' not in data.columns:
        data['datetime_taken'] = pd.to_datetime(
            data[['date_taken_year', 'date_taken_month', 'date_taken_day', 
                  'date_taken_hour', 'date_taken_minute']].rename(columns={
                'date_taken_year': 'year',
                'date_taken_month': 'month', 
                'date_taken_day': 'day',
                'date_taken_hour': 'hour',
                'date_taken_minute': 'minute'
            })
        )
    else:
        data['datetime_taken'] = pd.to_datetime(data['datetime_taken'])
    
    # Limiter le nombre de samples si demande
    if n_samples:
        data = data.head(n_samples)
    
    return data


def create_seasonal_map(data, output_path="output/flickr_map_seasonal.html"):
    """
    Cree une carte avec des points colores selon le jour de l'annee
    (meme couleur pour fevrier 2000 et fevrier 2005)
    """
    df = data.copy()
    
    # Supprimer les lignes avec des coordonnees manquantes
    df = df.dropna(subset=['lat', 'long', 'datetime_taken'])
    
    # Extraire le jour de l'annee (1-366)
    df['day_of_year'] = df['datetime_taken'].dt.dayofyear
    
    # Creer la carte centree sur Lyon (meme style que visualization.py)
    center_lat = df['lat'].mean()
    center_lon = df['long'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='Esri.WorldImagery'  # Carte satellite comme visualization.py
    )
    
    # Creer une palette de couleurs cyclique (pour les saisons)
    # On utilise une colormap HSV pour avoir un cycle de couleurs
    cmap = plt.cm.hsv
    
    # Normaliser les jours de l'annee entre 0 et 1
    norm = mcolors.Normalize(vmin=1, vmax=365)
    
    print(f"Creation de la carte saisonniere avec {len(df)} points...")
    
    # Ajouter chaque point avec sa couleur basee sur le jour de l'annee
    for idx, row in df.iterrows():
        day = row['day_of_year']
        
        # Obtenir la couleur RGB normalisee
        rgba = cmap(norm(day))
        color_hex = mcolors.rgb2hex(rgba[:3])
        
        # Minimal popup with just date and day of year
        popup_text = f"Day {day}"
        
        # Ajouter le point (meme style que visualization.py)
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=4,
            popup=folium.Popup(popup_text, max_width=150),
            color=color_hex,
            fill=True,
            fillColor=color_hex,
            fillOpacity=0.8
        ).add_to(m)
        
        if (idx + 1) % 10000 == 0:
            print(f"  {idx + 1}/{len(df)} points ajoutes...")
    
    # Ajouter une legende avec les couleurs par mois
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    legend_items = ""
    for i, month in enumerate(months):
        day_of_year = int((i + 0.5) * 365 / 12)
        rgba = cmap(norm(day_of_year))
        color_hex = mcolors.rgb2hex(rgba[:3])
        legend_items += f'<p style="margin:3px 0;"><span style="color: {color_hex}; font-size:20px;">o</span> {month}</p>\n'
    
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 150px; height: auto; max-height: 400px;
                background-color: white; z-index:9999; font-size:12px;
                border:2px solid grey; border-radius: 5px; padding: 10px;
                overflow-y: auto;">
    <p style="margin:0 0 5px 0; font-weight:bold; text-align:center;">Mois</p>
    ''' + legend_items + '''
    </div>
    '''
    # Inject legend using script to avoid Jinja2 template parsing
    script = f"""
    <script>
        var legendDiv = document.createElement('div');
        legendDiv.innerHTML = `{legend_html}`;
        document.body.appendChild(legendDiv);
    </script>
    """
    m.get_root().html.add_child(folium.Element(script))
    
    # Sauvegarder la carte
    m.save(output_path)
    print(f"Carte saisonniere sauvegardee : {output_path}")
    
    return m, df


def create_yearly_map(data, output_path="output/flickr_map_yearly.html"):
    """
    Cree une carte avec des points colores selon l'annee
    (chaque annee a sa propre couleur)
    """
    df = data.copy()
    
    # Supprimer les lignes avec des coordonnees manquantes
    df = df.dropna(subset=['lat', 'long', 'datetime_taken'])
    
    # Extraire l'annee
    df['year'] = df['datetime_taken'].dt.year
    
    # Creer la carte centree sur Lyon
    center_lat = df['lat'].mean()
    center_lon = df['long'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='Esri.WorldImagery'
    )
    
    # Obtenir les annees uniques et creer une palette de couleurs
    unique_years = sorted(df['year'].unique())
    n_years = len(unique_years)
    
    # Utiliser une colormap avec des couleurs distinctes
    cmap = plt.colormaps.get_cmap('hsv').resampled(n_years)
    
    # Creer un dictionnaire annee -> couleur
    year_colors = {}
    for i, year in enumerate(unique_years):
        rgba = cmap(i)
        year_colors[year] = mcolors.rgb2hex(rgba[:3])
    
    print(f"Creation de la carte annuelle avec {len(df)} points sur {n_years} annees...")
    
    # Ajouter chaque point avec sa couleur basee sur l'annee
    for idx, row in df.iterrows():
        year = row['year']
        color_hex = year_colors[year]
        
        # Minimal popup with just year
        popup_text = f"{year}"
        
        # Ajouter le point
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=4,
            popup=folium.Popup(popup_text, max_width=80),
            color=color_hex,
            fill=True,
            fillColor=color_hex,
            fillOpacity=0.8
        ).add_to(m)
        
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{len(df)} points ajoutes...")
    
    # Ajouter une legende avec les couleurs par annee
    legend_items = ""
    for year in unique_years:
        color_hex = year_colors[year]
        legend_items += f'<p style="margin:3px 0;"><span style="color: {color_hex}; font-size:20px;">o</span> {year}</p>\n'
    
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 150px; height: auto; max-height: 400px;
                background-color: white; z-index:9999; font-size:12px;
                border:2px solid grey; border-radius: 5px; padding: 10px;
                overflow-y: auto;">
    <p style="margin:0 0 5px 0; font-weight:bold; text-align:center;">Annees</p>
    ''' + legend_items + '''
    </div>
    '''
    # Inject legend using script to avoid Jinja2 template parsing
    script = f"""
    <script>
        var legendDiv = document.createElement('div');
        legendDiv.innerHTML = `{legend_html}`;
        document.body.appendChild(legendDiv);
    </script>
    """
    m.get_root().html.add_child(folium.Element(script))
    
    # Sauvegarder la carte
    m.save(output_path)
    print(f"Carte annuelle sauvegardee : {output_path}")
    
    return m, df


def plot_seasonal_distribution(data, output_path="output/seasonal_distribution.png"):
    """
    Cree un graphique montrant la distribution des photos par mois
    """
    df = data.copy()
    df = df.dropna(subset=['datetime_taken'])
    
    # Extraire le mois
    df['month'] = df['datetime_taken'].dt.month
    
    # Compter les photos par mois
    monthly_counts = df['month'].value_counts().sort_index()
    
    # Creer le graphique
    fig, ax = plt.subplots(figsize=(12, 6))
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    colors = plt.cm.hsv(np.linspace(0, 1, 12))
    
    ax.bar(range(1, 13), monthly_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months)
    ax.set_xlabel('Mois', fontsize=12)
    ax.set_ylabel('Nombre de photos', fontsize=12)
    ax.set_title('Distribution saisonniere des photos', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution saisonniere sauvegardee : {output_path}")


def create_hourly_map(data, output_path="output/flickr_map_hourly.html"):
    """
    Cree une carte avec des points colores selon l'heure de la journee (0-23h)
    (meme couleur pour 14h le 1er janvier et 14h le 15 mars)
    """
    df = data.copy()
    
    # Supprimer les lignes avec des coordonnees manquantes
    df = df.dropna(subset=['lat', 'long', 'datetime_taken'])
    
    # Extraire l'heure de la journee (0-23)
    df['hour'] = df['datetime_taken'].dt.hour
    
    # Creer la carte centree sur Lyon
    center_lat = df['lat'].mean()
    center_lon = df['long'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='Esri.WorldImagery'
    )
    
    # Creer une palette de couleurs cyclique (pour les heures)
    cmap = plt.cm.hsv
    
    # Normaliser les heures entre 0 et 1
    norm = mcolors.Normalize(vmin=0, vmax=23)
    
    print(f"Creation de la carte horaire avec {len(df)} points...")
    
    # Ajouter chaque point avec sa couleur basee sur l'heure
    for idx, row in df.iterrows():
        hour = row['hour']
        
        # Obtenir la couleur RGB normalisee
        rgba = cmap(norm(hour))
        color_hex = mcolors.rgb2hex(rgba[:3])
        
        # Minimal popup with just hour
        popup_text = f"{hour}h"
        
        # Ajouter le point
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=4,
            popup=folium.Popup(popup_text, max_width=80),
            color=color_hex,
            fill=True,
            fillColor=color_hex,
            fillOpacity=0.8
        ).add_to(m)
        
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{len(df)} points ajoutes...")
    
    # Ajouter une legende avec les couleurs par tranches horaires
    legend_items = ""
    for hour in range(24):
        rgba = cmap(norm(hour))
        color_hex = mcolors.rgb2hex(rgba[:3])
        legend_items += f'<p style="margin:3px 0;"><span style="color: {color_hex}; font-size:20px;">o</span> {hour}h</p>\n'
    
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 150px; height: auto; max-height: 400px;
                background-color: white; z-index:9999; font-size:12px;
                border:2px solid grey; border-radius: 5px; padding: 10px;
                overflow-y: auto;">
    <p style="margin:0 0 5px 0; font-weight:bold; text-align:center;">Heures</p>
    ''' + legend_items + '''
    </div>
    '''
    # Inject legend using script to avoid Jinja2 template parsing
    script = f"""
    <script>
        var legendDiv = document.createElement('div');
        legendDiv.innerHTML = `{legend_html}`;
        document.body.appendChild(legendDiv);
    </script>
    """
    m.get_root().html.add_child(folium.Element(script))
    
    # Sauvegarder la carte
    m.save(output_path)
    print(f"Carte horaire sauvegardee : {output_path}")
    
    return m, df


def plot_hourly_distribution(data, output_path="output/hourly_distribution.png"):
    """
    Cree un graphique montrant la distribution des photos par heure de la journee
    """
    df = data.copy()
    df = df.dropna(subset=['datetime_taken'])
    
    # Extraire l'heure
    df['hour'] = df['datetime_taken'].dt.hour
    
    # Compter les photos par heure
    hourly_counts = df['hour'].value_counts().sort_index()
    
    # S'assurer qu'on a toutes les heures (0-23)
    all_hours = pd.Series(0, index=range(24))
    all_hours.update(hourly_counts)
    hourly_counts = all_hours
    
    # Creer le graphique
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = plt.cm.hsv(np.linspace(0, 1, 24))
    
    ax.bar(range(24), hourly_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h}h' for h in range(24)], rotation=45, ha='right')
    ax.set_xlabel('Heure de la journee', fontsize=12)
    ax.set_ylabel('Nombre de photos', fontsize=12)
    ax.set_title('Distribution horaire des photos', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution horaire sauvegardee : {output_path}")


def plot_hour_month_heatmap(data, output_path="output/hour_month_heatmap.png"):
    """
    Cree une heatmap 2D croisant l'heure de la journee (0-23h) avec le mois de l'annee
    """
    df = data.copy()
    df = df.dropna(subset=['datetime_taken'])
    
    # Extraire l'heure et le mois
    df['hour'] = df['datetime_taken'].dt.hour
    df['month'] = df['datetime_taken'].dt.month
    
    # Creer une matrice de comptage (heure x mois)
    # Index = heures (0-23), Colonnes = mois (1-12)
    heatmap_data = pd.crosstab(df['hour'], df['month'])
    
    # S'assurer qu'on a toutes les heures et tous les mois (meme avec 0 photos)
    for h in range(24):
        if h not in heatmap_data.index:
            heatmap_data.loc[h] = 0
    for m in range(1, 13):
        if m not in heatmap_data.columns:
            heatmap_data[m] = 0
    
    heatmap_data = heatmap_data.sort_index().sort_index(axis=1)
    
    # Creer le graphique
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Utiliser imshow pour creer la heatmap
    im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Configurer les axes
    months = ['Jan', 'Fev', 'Mar', 'Avr', 'Mai', 'Jun', 
              'Jul', 'Aou', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(range(12))
    ax.set_xticklabels(months, fontsize=11)
    ax.set_yticks(range(24))
    ax.set_yticklabels([f'{h}h' for h in range(24)], fontsize=9)
    
    ax.set_xlabel('Mois de l\'annee', fontsize=13, fontweight='bold')
    ax.set_ylabel('Heure de la journee', fontsize=13, fontweight='bold')
    ax.set_title('Distribution des photos par heure et par mois', fontsize=15, fontweight='bold', pad=20)
    
    # Ajouter une barre de couleur
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Nombre de photos', rotation=270, labelpad=25, fontsize=12, fontweight='bold')
    
    # Ajouter les valeurs dans les cellules (optionnel, peut etre dense)
    # Seulement pour les valeurs non-nulles et si la matrice n'est pas trop grande
    if len(df) < 50000:  # Limite pour eviter trop de texte
        for i in range(24):
            for j in range(12):
                value = heatmap_data.values[i, j]
                if value > 0:
                    text_color = 'white' if value > heatmap_data.values.max() * 0.5 else 'black'
                    text = ax.text(j, i, int(value), ha="center", va="center", 
                                 color=text_color, fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap heure-mois sauvegardee : {output_path}")
    
    # Afficher quelques statistiques
    print(f"\nStatistiques de la heatmap:")
    print(f"  Heure la plus active : {heatmap_data.sum(axis=1).idxmax()}h ({int(heatmap_data.sum(axis=1).max())} photos)")
    print(f"  Mois le plus actif : {months[heatmap_data.sum(axis=0).idxmax()-1]} ({int(heatmap_data.sum(axis=0).max())} photos)")
    max_combo = heatmap_data.stack().idxmax()
    print(f"  Combinaison la plus frequente : {months[max_combo[1]-1]} a {max_combo[0]}h ({int(heatmap_data.loc[max_combo])} photos)")


def create_day_of_year_map(
    data,
    start_day,
    end_day,
    output_path="output/flickr_map_day_range.html",
    user_id=None,
):
    """
    Cree une carte pour une plage de jours de l'annee (inclusive).
    Utilise la visualisation standard pour afficher les tags.
    """
    df = data.copy()
    df = df.dropna(subset=['lat', 'long', 'datetime_taken'])

    df['day_of_year'] = df['datetime_taken'].dt.dayofyear
    df = df[(df['day_of_year'] >= start_day) & (df['day_of_year'] <= end_day)]

    if user_id and 'user' in df.columns:
        df = df[df['user'].astype(str).str.strip() == str(user_id)]

    if 'cluster' not in df.columns:
        df['cluster'] = 1

    # Nettoyer les champs texte pour eviter les erreurs HTML/Jinja2
    for col in ['user', 'title', 'tags', 'url']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: html.escape(str(x)) if pd.notna(x) else 'N/A')

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(
        f"Creation de la carte pour les jours {start_day}-{end_day} "
        f"({len(df)} points)..."
    )
    visualization.create_map(df, output_path)


if __name__ == "__main__":
    # Creer le dossier de sortie
    os.makedirs("output", exist_ok=True)
    
    # Charger les donnees
    print("Chargement des donnees...")
    data = load_and_prepare_data("data/cleaned_flickr_data.csv", None)
    
    print(f"\nNombre total de photos : {len(data)}")
    print(f"Periode : {data['datetime_taken'].min()} a {data['datetime_taken'].max()}")
    
    # Creer la carte saisonniere
    #print("\n=== Creation de la carte saisonniere ===")
    #map_obj, df_processed = create_seasonal_map(data)
    
    # Creer la carte par annee
    #print("\n=== Creation de la carte annuelle ===")
    #map_yearly, df_yearly = create_yearly_map(data)
    
    # Creer le graphique de distribution
    #print("\n=== Creation du graphique de distribution ===")
    #plot_seasonal_distribution(data)

    #Creer une carte pour une plage de jours (Fete des Lumières)
    #print("\n=== Creation de la carte pour une plage de jours ===")
    #create_day_of_year_map(data, 338, 350, "output/flickr_map_338_a_350.html", None)

    # Creer une carte pour une plage de jours pour un utilisateur specifique autour du jour 102
    #print("\n=== Creation de la carte pour une plage de jours ===")
    #create_day_of_year_map(data, 101, 103, "output/flickr_map_102_user_127623444N04.html", user_id="127623444@N04")

    # Creer la carte par heure de la journee
    #print("\n=== Creation de la carte par heure ===")
    #map_hourly, df_hourly = create_hourly_map(data)
    
    # Creer le graphique de distribution par heure
    #print("\n=== Creation du graphique de distribution horaire ===")
    #plot_hourly_distribution(data)

    # Creer la heatmap croisant heure et mois
    #print("\n=== Creation de la heatmap heure x mois ===")
    #plot_hour_month_heatmap(data)

    # Text mining sur une plage de jours (Fete des lumieres)
    print("\n=== Text mining (jours 338-350) ===")
    df_text = data.copy()
    df_text = df_text.dropna(subset=['datetime_taken'])
    df_text['day_of_year'] = df_text['datetime_taken'].dt.dayofyear
    df_text = df_text[(df_text['day_of_year'] >= 338) & (df_text['day_of_year'] <= 350)]
    df_text['cluster'] = 1

    cluster_texts = text_pattern_mining.preprocess_texts(df_text)
    if cluster_texts:
        text_pattern_mining.calcule_top_terms(cluster_texts, top_n=10)
    
