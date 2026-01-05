import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram

from visualization import create_map

def prepare_features(data, features):
    """
    Applique une normalisation (standardisation).
    """
    for col in features:
        if col not in data.columns:
            raise ValueError(f"Colonne manquante dans les données : {col}")

    X = data[features].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled

# Dendrogrammes

def plot_dendrogram(model, labels, title, outpath=None):
    """
    Construit la matrice de linkage à partir du modèle sklearn
    et trace le dendrogramme correspondant.
    """
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    # Calcul du nombre de points dans chaque fusion
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child in merge:
            if child < n_samples:
                current_count += 1
            else:
                current_count += counts[child - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([
        model.children_,
        model.distances_,
        counts
    ]).astype(float)

    fig = plt.figure(figsize=(12, 8))

    # Si le dataset est trop gros, on tronque le dendrogramme
    try:
        if n_samples > 1000:
            dendrogram(
                linkage_matrix,
                truncate_mode='lastp',
                p=30,
                leaf_rotation=90
            )
        else:
            dendrogram(
                linkage_matrix,
                labels=labels,
                leaf_rotation=90
            )
    except RecursionError:
        # Plan B si matplotlib n'aime vraiment pas la taille
        plt.clf()
        dendrogram(
            linkage_matrix,
            truncate_mode='lastp',
            p=20,
            leaf_rotation=90
        )

    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Distance")

    if outpath is not None:
        fig.savefig(outpath, bbox_inches="tight")

    plt.close(fig)



# Clustering hiérarchique
def run_hierarchical_clustering(X_scaled, n_clusters, linkage, metric):
    """
    Lance AgglomerativeClustering avec les paramètres donnés
    et retourne le modèle entraîné.
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric,
        compute_full_tree=True,
        compute_distances=True
    )

    model.fit(X_scaled)
    return model


# Silhouette
def compute_silhouette(X_scaled, labels, metric):
    """
    Calcule le score de silhouette moyen et par point.
    En cas de problème, renvoie None et des zéros.
    """
    try:
        sil_avg = silhouette_score(X_scaled, labels, metric=metric)
        sil_values = silhouette_samples(X_scaled, labels, metric=metric)
    except Exception:
        sil_avg = None
        sil_values = np.zeros(len(labels))

    return sil_avg, sil_values



def hierarchical_clustering(data):
    """
    Applique un clustering hiérarchique agglomératif
    avec plusieurs types de linkage et génère les :
    - dendrogrammes
    - scores de silhouette
    - cartes HTML
    """

    features = ['lat', 'long']
    linkages = ['complete', 'average', 'single']
    n_clusters = 10
    metric = 'euclidean'

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("\nLancement du clustering hiérarchique avec", linkages)

    # Copie du DataFrame pour éviter les effets de bord
    df = data.copy().reset_index(drop=True)

    # Préparation des données
    X_scaled = prepare_features(df, features)

    results = {}

    for link in linkages:
        print(f"→ Traitement du linkage : {link}")

        # Clustering
        model = run_hierarchical_clustering(
            X_scaled,
            n_clusters=n_clusters,
            linkage=link,
            metric=metric
        )

        # Dendrogramme
        dendro_path = os.path.join(output_dir, f"dendrogram_{link}.png")
        plot_dendrogram(
            model=model,
            labels=list(df.index.astype(str)),
            title=f"Dendrogramme – linkage : {link}",
            outpath=dendro_path
        )

        # Colonnes clusters et silhouette
        cluster_col = f"cluster_{link}"
        silhouette_col = f"silhouette_{link}"

        df[cluster_col] = model.labels_

        sil_avg, sil_values = compute_silhouette(
            X_scaled,
            model.labels_,
            metric
        )

        df[silhouette_col] = sil_values

        # Carte HTML (create_map attend une colonne 'cluster')
        map_df = df.copy()
        map_df["cluster"] = map_df[cluster_col]

        map_path = os.path.join(
            output_dir,
            f"flickr_map_hierarchical_{link}.html"
        )

        try:
            create_map(map_df, map_path)
        except Exception:
            map_path = None

        results[link] = {
            "dendrogram": dendro_path,
            "map": map_path,
            "silhouette_avg": float(sil_avg) if sil_avg is not None else None
        }

    return df, results
