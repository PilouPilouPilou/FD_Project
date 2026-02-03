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
        
        if (idx + 1) % 1000 == 0:
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
    
