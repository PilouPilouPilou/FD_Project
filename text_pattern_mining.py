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
        custom_stopwords = {'photo', 'picture', 'flickr', 'image', 'jpg', 'img', 'lyon', 'franc', 'rhônealp',
                            'europ','iphon', 'bokeh', 'portrait', 'rue','vill','street','city', 'frankreich','french'
                            ,'fujifilm', 'nikon', 'canon', 'sony', 'lumix', 'panasonic', 'chat', 'cat', 'kitten',
                            'geotag', 'cut', 'mignon', 'katz', 'kätzchen', 'chaton' }

    print(f"7. Retrait des mots personnalises : {custom_stopwords}")
    for cid in cluster_texts:
        words = cluster_texts[cid].split()
        filtered = [w for w in words if w not in custom_stopwords]
        cluster_texts[cid] = ' '.join(filtered)

    print("\n[OK] Preprocessing termine!\n")

    return cluster_texts

def calcule_top_terms(cluster_texts, top_n=10):
    # 8. Top termes par cluster (fréquences brutes)
    print("\n8. Top 5 mots par cluster (frequences)")
    top_terms = top_terms_by_cluster(cluster_texts, top_n=5)
    for cid in sorted(top_terms.keys()):
        pairs = top_terms[cid]
        if not pairs:
            print(f"   Cluster {cid}: (aucun mot)")
            continue
        formatted = ", ".join([f"{w}: {c}" for w, c in pairs])
        print(f"   Cluster {cid}: {formatted}")
    return top_terms

    
# # ============================================
# # PHASE 2 : TF-IDF
# # ============================================

# def analyze_tfidf(cluster_texts, top_n=15):
#     """
#     Analyse TF-IDF pour identifier les mots caractéristiques de chaque cluster
#     """
#     print("\n" + "="*60)
#     print("PHASE 2 : ANALYSE TF-IDF")
#     print("="*60 + "\n")

#     # Préparer les données
#     cluster_ids = sorted(cluster_texts.keys())
#     documents = [cluster_texts[cid] for cid in cluster_ids]

#     # Calculer TF-IDF
#     vectorizer = TfidfVectorizer(
#         min_df=1,  # Apparaît dans au moins 1 document
#         max_df=0.85,  # Maximum 85% des documents
#         ngram_range=(1, 2),  # Mots simples et bigrammes
#         max_features=1000
#     )

#     tfidf_matrix = vectorizer.fit_transform(documents)
#     feature_names = vectorizer.get_feature_names_out()

#     print(f"Matrice TF-IDF : {tfidf_matrix.shape} (clusters × mots)\n")

#     # Extraire les top mots par cluster
#     results = {}
#     for idx, cluster_id in enumerate(cluster_ids):
#         scores = tfidf_matrix[idx].toarray()[0]
#         top_indices = scores.argsort()[-top_n:][::-1]
#         top_words = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
#         results[cluster_id] = top_words

#         print(f"Cluster {cluster_id} - Top {len(top_words)} mots:")
#         for word, score in top_words[:10]:
#             print(f"  • {word:20s} (score: {score:.4f})")
#         print()

#     return results, vectorizer, tfidf_matrix


# def create_cluster_wordclouds(cluster_texts, tfidf_results, output_dir='./output'):
#     """Crée un word cloud par cluster basé sur les scores TF-IDF"""
#     print("Création des word clouds par cluster...\n")

#     for cluster_id, top_words in tfidf_results.items():
#         if not top_words:
#             continue

#         # Créer un dictionnaire {mot: score}
#         word_freq = {word: score for word, score in top_words}

#         wordcloud = WordCloud(width=800, height=400,
#                               background_color='white',
#                               colormap='plasma',
#                               max_words=50).generate_from_frequencies(word_freq)

#         plt.figure(figsize=(12, 6))
#         plt.imshow(wordcloud, interpolation='bilinear')
#         plt.axis('off')
#         plt.title(f'Cluster {cluster_id} - Mots caractéristiques', fontsize=14, fontweight='bold')

#         output_path = f"{output_dir}/wordcloud_cluster_{cluster_id}.png"
#         plt.savefig(output_path, dpi=200, bbox_inches='tight')
#         plt.close()

#     print(f"✓ Word clouds sauvegardés dans {output_dir}/\n")


# def export_tfidf_results(tfidf_results, output_path='./output/tfidf_results.csv'):
#     """Exporte les résultats TF-IDF dans un CSV"""
#     rows = []
#     for cluster_id, words in tfidf_results.items():
#         for word, score in words:
#             rows.append({'cluster': cluster_id, 'word': word, 'tfidf_score': score})

#     df = pd.DataFrame(rows)
#     df.to_csv(output_path, index=False)
#     print(f"✓ Résultats TF-IDF exportés : {output_path}\n")


# # ============================================
# # PHASE 3 : RÈGLES D'ASSOCIATION
# # ============================================

# def prepare_transactions(data):
#     """
#     Prépare les données en format transactionnel pour Apriori
#     Chaque photo = une transaction (liste de mots)
#     """
#     print("\n" + "="*60)
#     print("PHASE 3 : RÈGLES D'ASSOCIATION")
#     print("="*60 + "\n")

#     print("Préparation des transactions...")

#     # Combiner title et tags pour chaque photo
#     data['combined_text'] = (
#         data['title'].fillna('').astype(str) + ' ' +
#         data['tags'].fillna('').astype(str)
#     )

#     # Nettoyer
#     data['cleaned_text'] = data['combined_text'].apply(basic_cleaning)
#     data['cleaned_text'] = data['cleaned_text'].apply(lambda x: remove_stopwords(x, 'multilingual'))
#     data['cleaned_text'] = data['cleaned_text'].apply(stem_text)

#     # Retirer custom stopwords
#     custom_stops = {'photo', 'picture', 'flickr', 'image', 'jpg', 'img'}
#     data['cleaned_text'] = data['cleaned_text'].apply(
#         lambda x: ' '.join([w for w in x.split() if w not in custom_stops and len(w) > 2])
#     )

#     # Convertir en liste de listes
#     transactions = data['cleaned_text'].apply(lambda x: x.split()).tolist()

#     print(f"   → {len(transactions)} transactions créées")
#     print(f"   → Exemple : {transactions[0][:10]}\n")

#     return transactions, data


# def analyze_association_rules_by_cluster(data, min_support=0.1, min_confidence=0.5):
#     """
#     Analyse les règles d'association pour chaque cluster séparément
#     """
#     transactions_all, data_clean = prepare_transactions(data)

#     results = {}

#     for cluster_id in sorted(data['cluster'].unique()):
#         if cluster_id == -1:
#             continue

#         print(f"\n--- Cluster {cluster_id} ---")

#         # Filtrer les transactions du cluster
#         cluster_mask = data['cluster'] == cluster_id
#         cluster_transactions = [t for t, mask in zip(transactions_all, cluster_mask) if mask]

#         # Filtrer les transactions vides
#         cluster_transactions = [t for t in cluster_transactions if len(t) > 0]

#         if len(cluster_transactions) < 2:
#             print(f"   ⚠ Pas assez de transactions ({len(cluster_transactions)})")
#             continue

#         print(f"   {len(cluster_transactions)} transactions dans ce cluster")

#         # Encoder en format binaire
#         te = TransactionEncoder()
#         te_ary = te.fit(cluster_transactions).transform(cluster_transactions)
#         df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

#         # Appliquer Apriori
#         try:
#             frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

#             if len(frequent_itemsets) == 0:
#                 print(f"   ⚠ Aucun itemset fréquent (essayez min_support plus faible)")
#                 continue

#             print(f"   → {len(frequent_itemsets)} itemsets fréquents trouvés")

#             # Trier par support
#             frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)

#             # Afficher les top itemsets
#             print(f"\n   Top 10 itemsets fréquents:")
#             for idx, row in frequent_itemsets.head(10).iterrows():
#                 items = list(row['itemsets'])
#                 print(f"     • {items} (support: {row['support']:.3f})")

#             results[cluster_id] = frequent_itemsets

#             # Essayer de générer des règles d'association
#             if len(frequent_itemsets[frequent_itemsets['itemsets'].apply(len) >= 2]) > 0:
#                 try:
#                     rules = association_rules(frequent_itemsets, metric="confidence",
#                                              min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))

#                     if len(rules) > 0:
#                         print(f"\n   Top 5 règles d'association:")
#                         rules_sorted = rules.sort_values('confidence', ascending=False)
#                         for idx, row in rules_sorted.head(5).iterrows():
#                             ant = list(row['antecedents'])
#                             cons = list(row['consequents'])
#                             print(f"     • {ant} → {cons} (conf: {row['confidence']:.3f})")
#                 except Exception as e:
#                     print(f"   ⚠ Pas assez de règles générées")

#         except Exception as e:
#             print(f"   ⚠ Erreur Apriori : {e}")
#             continue

#     return results


# def export_association_results(association_results, output_path='./output/association_rules.csv'):
#     """Exporte les itemsets fréquents dans un CSV"""
#     rows = []
#     for cluster_id, itemsets_df in association_results.items():
#         for idx, row in itemsets_df.iterrows():
#             items = list(row['itemsets'])
#             rows.append({
#                 'cluster': cluster_id,
#                 'itemset': ', '.join(items),
#                 'support': row['support'],
#                 'length': len(items)
#             })

#     df = pd.DataFrame(rows)
#     df.to_csv(output_path, index=False)
#     print(f"\n✓ Résultats des règles d'association exportés : {output_path}\n")


# # ============================================
# # FONCTION PRINCIPALE
# # ============================================

# def analyze_text_patterns(data, custom_stopwords=None, min_support=0.1):
#     """
#     Pipeline complet : preprocessing + TF-IDF + règles d'association

#     Paramètres:
#         data: DataFrame avec colonnes 'cluster', 'title', 'tags'
#         custom_stopwords: set de mots à ignorer (optionnel)
#         min_support: support minimum pour Apriori (0.1 = 10%)

#     Retourne:
#         preprocessed_texts, tfidf_results, association_results
#     """

#     # Phase 1 : Preprocessing
#     preprocessed_texts = preprocess_texts(data, custom_stopwords)

#     if preprocessed_texts is None:
#         return None, None, None

#     # Phase 2 : TF-IDF
#     tfidf_results, vectorizer, tfidf_matrix = analyze_tfidf(preprocessed_texts, top_n=15)
#     create_cluster_wordclouds(preprocessed_texts, tfidf_results)
#     export_tfidf_results(tfidf_results)

#     # Phase 3 : Règles d'association
#     association_results = analyze_association_rules_by_cluster(data, min_support=min_support)
#     export_association_results(association_results)

#     print("\n" + "="*60)
#     print("✓ ANALYSE COMPLÈTE TERMINÉE")
#     print("="*60 + "\n")

#     return preprocessed_texts, tfidf_results, association_results
