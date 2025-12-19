import pandas as pd


def convert_types(data):

    # Convertir les colonnes numériques en int ou float
    numeric_columns = ['id', 'lat', 'long', 
                    'date_taken_minute', 'date_taken_hour', 'date_taken_day', 
                    'date_taken_month', 'date_taken_year',
                    'date_upload_minute', 'date_upload_hour', 'date_upload_day', 
                    'date_upload_month', 'date_upload_year']

    for col in numeric_columns:
        if col in data.columns:
            # Convertir en float d'abord (gère les décimales), puis en int si pas de décimales
            data[col] = pd.to_numeric(data[col], errors='coerce')
            # Si la colonne n'a pas de NaN après conversion, on peut la passer en int
            if data[col].isna().sum() == 0 and data[col].dtype == 'float64' and (data[col] % 1 == 0).all():
                data[col] = data[col].astype('int64')

    print("Types de données convertis:")
    print(data.dtypes)
    return data


def analyze_missing(data):
    ###### ANALYSE DES VALEURS MANQUANTES ##########
    print("\n--- Analyse des valeurs manquantes (NaN) ---")
    print("\nNombre de NaN par colonne:")
    missing_data = data.isna().sum()
    print(missing_data[missing_data > 0])

    print(f"\nPourcentage de NaN par colonne:")
    missing_percent = (data.isna().sum() / len(data)) * 100
    print(missing_percent[missing_percent > 0].round(2))

    # Identifier les lignes avec des valeurs manquantes dans les dates
    date_columns = ['date_taken_minute', 'date_taken_hour', 'date_taken_day', 'date_taken_month', 'date_taken_year',
                    'date_upload_minute', 'date_upload_hour', 'date_upload_day', 'date_upload_month', 'date_upload_year']
    rows_with_missing_dates = data[data[date_columns].isna().any(axis=1)]

    print(f"\nNombre de lignes avec au moins une date manquante: {len(rows_with_missing_dates)}")
    print("\nExemples de lignes avec des dates manquantes:")
    print(rows_with_missing_dates[['id', 'user', 'title'] + date_columns].head(10))
    return


def remove_duplicates(data):
    print(f"\nInitial: {len(data)}")
    print("Nombre de duplicats :", data.duplicated().sum())

    data_cleaned = data.drop_duplicates(keep='first')

    print(f"After removing duplicates: {len(data_cleaned)}")
    return data_cleaned
