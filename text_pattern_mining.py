import re
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import unicodedata

# Télécharger les ressources NLTK si nécessaire (à commenter après la première exécution)
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('punkt')


# ============================================
# PHASE 1 : PREPROCESSING
# ============================================

def basic_cleaning(text):
    """Nettoie le texte : minuscules, ponctuation, espaces multiples"""
    if pd.isna(text):
        return ""
    text = str(text).lower()

    # Supprimer les accents
    nfd = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    
    # Garder lettres, chiffres, espaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Retirer espaces multiples
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_stopwords(text, language='multilingual'):
    """Retire les stop words (anglais et français)"""
    if language == 'multilingual':
        stop_words = set(stopwords.words('english')) | set(stopwords.words('french'))
    else:
        stop_words = set(stopwords.words(language))

    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(filtered_words)


def stem_text(text):
    """Applique le stemming (drinking -> drink) -> voir si on peut améliorer en détectant la langue"""
    stemmer = PorterStemmer()
    words = text.split()
    stemmed = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed)


def remove_words_with_digits(text):
    """Supprime les mots qui contiennent au moins un chiffre"""
    words = text.split()
    filtered = [w for w in words if not any(ch.isdigit() for ch in w)]
    return ' '.join(filtered)


def aggregate_texts_by_cluster(data):
    """
    Regroupe les titres et tags par cluster
    Retourne un dictionnaire : {cluster_id: "texte combiné"}
    """
    cluster_texts = {} # : Initialise un dict cluster_texts pour stocker {cluster_id: texte}.

    for cluster_id in sorted(data['cluster'].unique()):
        if cluster_id == -1:  # Ignorer le bruit (DBSCAN)
            continue

        cluster_data = data[data['cluster'] == cluster_id] # on ne garde que les data du cluster courant

        # Combiner tous les titres et tags
        all_titles = cluster_data['title'].fillna('').astype(str)
        all_tags = cluster_data['tags'].fillna('').astype(str)

        combined_text = ' '.join(all_titles) + ' ' + ' '.join(all_tags)
        cluster_texts[cluster_id] = combined_text

    return cluster_texts


def get_most_common_words(cluster_texts, top_n=50):
    """Trouve les N mots les plus fréquents dans tout le corpus"""
    all_words = []
    for text in cluster_texts.values(): # les textes de chaque cluster
        all_words.extend(text.split()) # divise le texte en mots et les ajoute à la liste all_words

    word_counts = Counter(all_words) # chaque moi est une clé et son nb occurence est la valeur
    return word_counts.most_common(top_n) # renvoie une liste triée de tuples (mot, nbOccurence) les plus fréquents


def top_terms_by_cluster(cluster_texts, top_n=5):
    """Calcule les top mots par cluster (fréquence brute)"""
    results = {}
    for cid, text in cluster_texts.items(): # pour chaque cluster_id et son texte associé
        words = text.split()
        if not words:
            results[cid] = []
            continue
        counter = Counter(words)
        results[cid] = counter.most_common(top_n)
    return results # renvoie un dict {cluster_id: [(mot, nbOccurence), ...]}


def create_global_wordcloud(cluster_texts, output_path='./output/wordcloud_global.png'):
    """Crée un word cloud de tous les textes"""
    all_text = ' '.join(cluster_texts.values())

    if not all_text.strip():
        print("   [!] Pas assez de texte pour creer un word cloud")
        return

    # Compter les fréquences de chaque mot
    word_freq = Counter(all_text.split())

    # Générer le word cloud à partir des fréquences
    wordcloud = WordCloud(width=1200, height=600,
                          background_color='white',
                          max_words=100,
                          colormap='viridis',
                          prefer_horizontal=0.7).generate_from_frequencies(word_freq)

    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud.to_image(), interpolation='bilinear')
    plt.axis('off')
    plt.title('Mots les plus fréquents (tous clusters)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Word cloud dans : {output_path}")


def preprocess_texts(data, custom_stopwords=None):
    """
    Pipeline complet de preprocessing Phase 1
    Retourne les textes nettoyés par cluster
    """
    print("\n" + "="*60)
    print("PHASE 1 : PREPROCESSING DES TEXTES")
    print("="*60)

    if 'cluster' not in data.columns:
        print("[ERREUR] Colonne 'cluster' manquante. Faites le clustering d'abord!")
        return None

    # 1. Agréger par cluster
    print("\n1. Agregation des textes par cluster...")
    cluster_texts = aggregate_texts_by_cluster(data)
    print(f"   -> {len(cluster_texts)} clusters trouves")
    # mots du premier cluster :
    print(f"   -> Exemple (cluster {list(cluster_texts.keys())[0]}): {cluster_texts[list(cluster_texts.keys())[0]][:200]}")

    # 2. Nettoyage basique
    print("2. Nettoyage basique (minuscules, ponctuation)...")
    for cid in cluster_texts:
        cluster_texts[cid] = basic_cleaning(cluster_texts[cid])
    print(f"   -> Exemple (cluster {list(cluster_texts.keys())[0]}): {cluster_texts[list(cluster_texts.keys())[0]][:200]}")

    # 3. Stop words
    print("3. Retrait des stop words (EN + FR)...")
    for cid in cluster_texts:
        cluster_texts[cid] = remove_stopwords(cluster_texts[cid], 'multilingual')
    print(f"   -> Exemple (cluster {list(cluster_texts.keys())[0]}): {cluster_texts[list(cluster_texts.keys())[0]][:200]}")

    # 4. Retirer les mots contenant des chiffres
    print("4. Suppression des mots contenant des chiffres...")
    for cid in cluster_texts:
        cluster_texts[cid] = remove_words_with_digits(cluster_texts[cid])

    # 5. Stemming
    print("5. Stemming (reduction a la racine)...")
    for cid in cluster_texts:
        cluster_texts[cid] = stem_text(cluster_texts[cid])
    print(f"   -> Exemple (cluster {list(cluster_texts.keys())[0]}): {cluster_texts[list(cluster_texts.keys())[0]][:200]}")

        # 6. Mots trop fréquents
    print("6. Analyse des mots frequents...")
    common = get_most_common_words(cluster_texts, 30)
    print(f"   -> Top 10 mots : {[w for w, c in common[:10]]}")

    # Word cloud global
    create_global_wordcloud(cluster_texts)

    # Retirer mots personnalisés
    if custom_stopwords is None:
        custom_stopwords = {'photo', 'picture', 'flickr', 'image', 'jpg', 'img', 'lyon', 'franc', 'rhonealp',
                            'europ','iphon', 'bokeh', 'portrait', 'rue','vill','street','city', 'frankreich','french'
                            ,'fujifilm', 'nikon', 'canon', 'sony', 'lumix', 'panasonic', 'chat', 'cat', 'kitten',
                            'geotag', 'cut', 'mignon', 'katz', 'katzchen', 'chaton', 'int', 'rieur', 'light'}

    print(f"7. Retrait des mots personnalises : {custom_stopwords}")
    for cid in cluster_texts:
        words = cluster_texts[cid].split()
        filtered = [w for w in words if w not in custom_stopwords]
        cluster_texts[cid] = ' '.join(filtered)

    print("\n[OK] Preprocessing termine!\n")

    return cluster_texts

def calcule_top_terms(cluster_texts, top_n=10):
    # 8. Top termes par cluster (fréquences brutes)
    print(f"\n8. Top {top_n} mots par cluster (frequences)")
    top_terms = top_terms_by_cluster(cluster_texts, top_n=top_n)
    for cid in sorted(top_terms.keys()):
        pairs = top_terms[cid]
        if not pairs:
            print(f"   Cluster {cid}: (aucun mot)")
            continue
        formatted = ", ".join([f"{w}: {c}" for w, c in pairs])
        print(f"   Cluster {cid}: {formatted}")
    return top_terms

def calcule_top_terms_TFIDF(cluster_texts, top_n=10):
    """Calcule les top termes par cluster selon TF-IDF.
    Prend un dict {cluster_id: texte} et retourne {cluster_id: [(mot, score), ...]}.
    """
    print(f"\n8b. Top {top_n} mots par cluster (TF-IDF)")

    if not cluster_texts:
        return {}

    # Ordonner les clusters pour un affichage stable
    ordered_ids = sorted(cluster_texts.keys())
    documents = [cluster_texts[cid] for cid in ordered_ids]

    # Vectoriseur TF-IDF (on ignore les tokens < 3 caractères, TF sublinéaire)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w{3,}\b", # mots d'au moins 3 caractères
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        norm='l2',
        max_df=0.8,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        # Vocabulaire vide (par ex. textes vides après nettoyage)
        return {cid: [] for cid in ordered_ids}

    feature_names = vectorizer.get_feature_names_out()

    results = {}
    for i, cid in enumerate(ordered_ids):
        row = tfidf_matrix[i].toarray().ravel()
        # Vérifier si le cluster a des termes significatifs
        if row.size == 0 or (row <= 0).all():
            results[cid] = []
            print(f"   Cluster {cid}: (aucun terme significatif)")
            continue

        # Indices triés par score décroissant et gadner les top_n premiers
        top_idx = row.argsort()[::-1][:top_n]
        pairs = [(feature_names[j], float(row[j])) for j in top_idx if row[j] > 0] # crée tuples (mot, score), en gardant que les scores > 0
        results[cid] = pairs # stocke dans le dict {cluster_id: [(mot, score), ...]}

        formatted = ", ".join([f"{w}: {score:.3f}" for w, score in pairs])
        print(f"   Cluster {cid}: {formatted}")

    return results

