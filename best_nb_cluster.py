from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Recherche du meilleur k via le score de silhouette
def find_best_n_clusters(X_scaled, linkage, metric, k_min, k_max):
    """
    Teste plusieurs nombres de clusters sur une plage et retourne celui
    qui maximise le score de silhouette moyen.
    """
    best_k = None
    best_score = -1

    for k in range(k_min, k_max + 1):
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
            metric=metric
        )

        labels = model.fit_predict(X_scaled)

        # Silhouette uniquement si plus d'un cluster
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X_scaled, labels, metric=metric)
        else:
            score = -1

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score


# Trace la courbe du score de silhouette
def plot_silhouette_curve(X, linkage,  metric, k_min=2, k_max=100, outpath="output/silhouette_curve.png"):
    """
    Calcule et trace la courbe du score de silhouette
    en fonction du nombre de clusters.
    """

    ks = []
    scores = []

    print("Calcul de la courbe de silhouette...")

    for k in range(k_min, k_max + 1):
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
            metric=metric
        )

        labels = model.fit_predict(X)

        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels, metric=metric)
            ks.append(k)
            scores.append(score)

    ks = np.array(ks)
    scores = np.array(scores)

    # Affichage
    plt.figure(figsize=(10, 6))
    plt.plot(ks, scores, marker='o', linewidth=2)
    plt.xlabel("Nombre de clusters (k)")
    plt.ylabel("Score de silhouette moyen")
    plt.title(f"Courbe de silhouette – linkage = {linkage}")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

    print(f"Courbe de silhouette sauvegardée : {outpath}")

# Préparation des données pour le clustering 
def prepare_features(data, features):
    """
    Applique une normalisation avec StandardScaler
    """
    X = data[features].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled



# Exemple : chargement des données
data = pd.read_csv("data/cleaned_flickr_data.csv")

# Limiter le dataset
nb_datas = 10000
data = data.head(nb_datas)

features = ['lat', 'long']
metric = 'euclidean'
linkage = 'complete'

# Préparation des données
X_scaled = prepare_features(data, features)

# Tracer la courbe de silhouette pour trouver visuellement le meilleur k
plot_silhouette_curve(X_scaled, linkage=linkage, metric=metric, k_min=2, k_max=100, outpath="output/silhouette_curve.png")

#Rechercher le meilleur k automatiquement
print("Recherche du nombre optimal de clusters avec", nb_datas, "données...")

best_k, best_sil = find_best_n_clusters(X_scaled, linkage, metric, k_min=10, k_max=100)
print(f"Nombre optimal de clusters : {best_k}")
print(f"Silhouette moyenne : {best_sil:.3f}")