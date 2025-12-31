import pandas as pd
import os


def _cast_numeric_columns(df, numeric_columns):
    """
    Convertit les colonnes listées en numériques, puis en Int64 si toutes les
    valeurs non nulles sont entières, sinon en float64. Conserve les NaN.
    """
    for col in numeric_columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        non_null = series.dropna()
        if not non_null.empty and (non_null % 1 == 0).all():
            df[col] = series.astype("Int64")  # garde les NaN
        else:
            df[col] = series.astype("float64")
    return df


def convert_types(data):
    # Colonnes numériques attendues
    numeric_columns = [
        "id",
        "lat",
        "long",
        "date_taken_minute",
        "date_taken_hour",
        "date_taken_day",
        "date_taken_month",
        "date_taken_year",
        "date_upload_minute",
        "date_upload_hour",
        "date_upload_day",
        "date_upload_month",
        "date_upload_year",
    ]

    data = _cast_numeric_columns(data.copy(), numeric_columns)

    # Créer des colonnes datetime directement après conversion
    data["datetime_taken"] = _build_datetime(data, "date_taken")
    data["datetime_upload"] = _build_datetime(data, "date_upload")

    print("Types de données convertis:")
    # Afficher les types de façon lisible
    try:
        print(data.dtypes.to_string())
    except Exception:
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
    print(f"Nombre de lignes initialement: {len(data)}")
    # Dédoublonnage spécifique: on conserve une seule ligne par paire (id, user)
    dup_mask = data.duplicated(subset=["id", "user"], keep="first")
    print("Duplicats sur (id, user) :", int(dup_mask.sum()))

    data_cleaned = data.drop_duplicates(subset=["id", "user"], keep="first")

    print(f"Nombre de lignes après suppression des duplicats: {len(data_cleaned)}")
    return data_cleaned


def _build_datetime(df, prefix):
    """
    Construit une colonne datetime à partir de colonnes séparées
    (year, month, day, hour, minute) avec le préfixe donné.
    Retourne une Series de type datetime (NaT en cas d'erreur).
    """
    required = [
        f"{prefix}_year",
        f"{prefix}_month",
        f"{prefix}_day",
        f"{prefix}_hour",
        f"{prefix}_minute",
    ]
    # Si une des colonnes requises manque, retourner une Series de NaT
    if not all(c in df.columns for c in required):
        return pd.Series(pd.NaT, index=df.index)

    # Préparer le DataFrame avec les noms standard attendus par pd.to_datetime
    tmp = df[required].rename(
        columns={
            f"{prefix}_year": "year",
            f"{prefix}_month": "month",
            f"{prefix}_day": "day",
            f"{prefix}_hour": "hour",
            f"{prefix}_minute": "minute",
        }
    )

    # Conversion en numérique avec coercition pour gérer les valeurs non conformes
    for c in ["year", "month", "day", "hour", "minute"]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    # Calculer uniquement pour les lignes complètes; les autres restent NaT
    complete_mask = tmp[["year", "month", "day", "hour", "minute"]].notna().all(axis=1)
    result = pd.Series(pd.NaT, index=df.index)
    if complete_mask.any():
        result.loc[complete_mask] = pd.to_datetime(
            tmp.loc[complete_mask, ["year", "month", "day", "hour", "minute"]],
            errors="coerce",
        )
    return result


def detect_anomalies(data, save_path=None):
    """
    Détecte des anomalies courantes dans le dataset et retourne un DataFrame
    des lignes problématiques avec une colonne "anomaly_reasons".

    Anomalies détectées:
    - Coordonnées manquantes
    - Coordonnées hors bornes (lat ∉ [-90, 90], long ∉ [-180, 180])
    - Coordonnées (0, 0) (souvent valeurs par défaut/sentinelles)
    - Coordonnées hors de Lyon (lat ∉ [45.65, 45.85], long ∉ [4.72, 5.05])
    - Parties de date manquantes (taken/upload)
    - Parties de date hors plage (minutes 0-59, heures 0-23, jours 1-31, mois 1-12)
    - Date d'upload antérieure à la date de prise

    Si save_path est fourni, enregistre le rapport en CSV.
    """
    df = data.copy()
    df.columns = df.columns.str.strip()

    # Vérifications coordonnées
    has_lat = "lat" in df.columns
    has_long = "long" in df.columns
    missing_coords = (df["lat"].isna() if has_lat else False) | (df["long"].isna() if has_long else False)
    out_of_bounds = (
        (~df["lat"].between(-90, 90, inclusive="both") if has_lat else False)
        | (~df["long"].between(-180, 180, inclusive="both") if has_long else False)
    )
    zero_coords = ((df["lat"] == 0) & (df["long"] == 0)) if (has_lat and has_long) else False
    
    # Coordonnées hors de Lyon (bounding box approximative de Lyon métropolitain)
    # Lyon: lat ~45.65 à 45.85, long ~4.72 à 5.05
    out_of_lyon = (
        (~df["lat"].between(45.65, 45.85, inclusive="both") if has_lat else False)
        | (~df["long"].between(4.72, 5.05, inclusive="both") if has_long else False)
    )

    # Vérifications des dates
    taken_prefix = "date_taken"
    upload_prefix = "date_upload"

    def exists_all(prefix):
        req = [f"{prefix}_minute", f"{prefix}_hour", f"{prefix}_day", f"{prefix}_month", f"{prefix}_year"]
        return all(c in df.columns for c in req)

    def out_of_range(series, low, high):
        s = pd.to_numeric(series, errors="coerce")
        return (~s.between(low, high, inclusive="both")) & s.notna()

    # Taken
    if exists_all(taken_prefix):
        taken_missing_parts = df[[
            f"{taken_prefix}_minute",
            f"{taken_prefix}_hour",
            f"{taken_prefix}_day",
            f"{taken_prefix}_month",
            f"{taken_prefix}_year",
        ]].isna().any(axis=1)
        taken_minute_oob = out_of_range(df[f"{taken_prefix}_minute"], 0, 59)
        taken_hour_oob = out_of_range(df[f"{taken_prefix}_hour"], 0, 23)
        taken_day_oob = out_of_range(df[f"{taken_prefix}_day"], 1, 31)
        taken_month_oob = out_of_range(df[f"{taken_prefix}_month"], 1, 12)
        # Années: bornes larges pour Flickr
        taken_year_oob = out_of_range(df[f"{taken_prefix}_year"], 1900, 2026)
    else:
        taken_missing_parts = False
        taken_minute_oob = False
        taken_hour_oob = False
        taken_day_oob = False
        taken_month_oob = False
        taken_year_oob = False

    # Upload
    if exists_all(upload_prefix):
        upload_missing_parts = df[[
            f"{upload_prefix}_minute",
            f"{upload_prefix}_hour",
            f"{upload_prefix}_day",
            f"{upload_prefix}_month",
            f"{upload_prefix}_year",
        ]].isna().any(axis=1)
        upload_minute_oob = out_of_range(df[f"{upload_prefix}_minute"], 0, 59)
        upload_hour_oob = out_of_range(df[f"{upload_prefix}_hour"], 0, 23)
        upload_day_oob = out_of_range(df[f"{upload_prefix}_day"], 1, 31)
        upload_month_oob = out_of_range(df[f"{upload_prefix}_month"], 1, 12)
        upload_year_oob = out_of_range(df[f"{upload_prefix}_year"], 2004, 2026) # Flickr lancé en 2004
    else:
        upload_missing_parts = False
        upload_minute_oob = False
        upload_hour_oob = False
        upload_day_oob = False
        upload_month_oob = False
        upload_year_oob = False

    # Chronologie: upload < taken
    dt_taken = df["datetime_taken"]
    dt_upload = df["datetime_upload"]
    upload_before_taken = dt_upload.notna() & dt_taken.notna() & (dt_upload < dt_taken)

    masks = {
        "missing_coords": missing_coords,
        "out_of_bounds": out_of_bounds,
        "zero_coords": zero_coords,
        "out_of_lyon": out_of_lyon,
        "taken_missing_parts": taken_missing_parts,
        "taken_minute_oob": taken_minute_oob,
        "taken_hour_oob": taken_hour_oob,
        "taken_day_oob": taken_day_oob,
        "taken_month_oob": taken_month_oob,
        "taken_year_oob": taken_year_oob,
        "upload_missing_parts": upload_missing_parts,
        "upload_minute_oob": upload_minute_oob,
        "upload_hour_oob": upload_hour_oob,
        "upload_day_oob": upload_day_oob,
        "upload_month_oob": upload_month_oob,
        "upload_year_oob": upload_year_oob,
        "upload_before_taken": upload_before_taken,
    }

    any_anomaly = False
    for m in masks.values():
        any_anomaly = any_anomaly | m

    anomalies_df = df.loc[any_anomaly].copy()

    def reasons_for_idx(idx):
        rs = []
        for name, m in masks.items():
            try:
                if bool(m.loc[idx]):
                    rs.append(name)
            except KeyError:
                # index absent dans le masque (ne devrait pas arriver), ignorer
                pass
        return ", ".join(rs)

    anomalies_df["anomaly_reasons"] = [reasons_for_idx(i) for i in anomalies_df.index]

    # Afficher résumé
    print("\n--- Détection d'anomalies ---")
    rows = [(name, int(m.sum())) for name, m in masks.items()]
    if rows:
        max_name = max(len(r[0]) for r in rows)
        header = f"{'Anomalie'.ljust(max_name)} | Count"
        sep = '-' * len(header)
        print(header)
        print(sep)
        for name, count in rows:
            print(f"{name.ljust(max_name)} | {count}")
    print(f"\nTotal lignes avec au moins une anomalie: {len(anomalies_df)}")

    # Sauvegarde éventuelle
    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        anomalies_df.to_csv(save_path, index=False)
        print(f"Rapport d'anomalies sauvegardé sous '{save_path}'")

    # Préparer un résumé dict {anomaly_name: count}
    summary = {name: int(m.sum()) for name, m in masks.items()}

    # Retourner à la fois le DataFrame d'anomalies et le résumé
    return anomalies_df, summary
