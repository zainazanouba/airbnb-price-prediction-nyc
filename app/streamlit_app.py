import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Optionnel (si dispo)
try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False


# =========================
# PATHS
# =========================
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.joblib")


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Airbnb NYC — Projet complet", layout="wide")
st.title("🏠 Airbnb NYC — EDA complet • Slides • Modélisation • Prédiction")


# =========================
# UTILS ROBUSTES
# =========================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les noms de colonnes pour éviter les mismatches (espaces, majuscules, etc.)."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


def clean_money_to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan


@st.cache_data
def load_data():
    candidates = [
        os.path.join(HERE, "data", "row", "Airbnb_Open_Data.csv"),
        os.path.join(HERE, "data", "raw", "Airbnb_Open_Data.csv"),
        os.path.join(HERE, "data", "Airbnb_Open_Data.csv"),
        os.path.join(HERE, "Airbnb_Open_Data.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df_ = pd.read_csv(path, low_memory=False)
            df_ = normalize_columns(df_)
            return df_, path

    candidates_xlsx = [
        os.path.join(HERE, "Airbnb_Open_Data.xlsx"),
        os.path.join(HERE, "Airbnb_Open_Data.xls"),
    ]
    for path in candidates_xlsx:
        if os.path.exists(path):
            df_ = pd.read_excel(path)
            df_ = normalize_columns(df_)
            return df_, path

    return None, None


def pick_existing(cols_set, *names):
    for n in names:
        n2 = n.strip().lower()
        if n2 in cols_set:
            return n2
    return None


def detect_columns(df: pd.DataFrame):
    cols = set(df.columns)
    meta = {
        "price": pick_existing(cols, "price"),
        "lat": pick_existing(cols, "lat", "latitude"),
        "lon": pick_existing(cols, "long", "longitude", "lng"),
        "min_nights": pick_existing(cols, "minimum nights", "minimum_nights"),
        "n_reviews": pick_existing(cols, "number of reviews", "number_of_reviews"),
        "rpm": pick_existing(cols, "reviews per month", "reviews_per_month"),
        "avail": pick_existing(cols, "availability 365", "availability_365"),
        "year": pick_existing(cols, "construction year", "construction_year"),
        "neigh_group": pick_existing(cols, "neighbourhood group", "neighborhood group", "neighbourhood_group"),
        "neigh": pick_existing(cols, "neighbourhood", "neighborhood"),
        "room": pick_existing(cols, "room type", "room_type"),
        "host_verif": pick_existing(cols, "host_identity_verified", "host identity verified"),
        "instant": pick_existing(cols, "instant_bookable", "instant bookable"),
        "cancel": pick_existing(cols, "cancellation_policy", "cancellation policy"),
        "service_fee": pick_existing(cols, "service fee", "service_fee"),
    }
    return meta


def leakage_safe_drop(df: pd.DataFrame):
    """
    Évite la fuite (leakage) : on supprime les colonnes qui contiennent du prix
    ou des variables dérivées du prix si présentes.
    """
    df = df.copy()
    drop_cols = []
    for c in df.columns:
        c_low = c.lower()
        # évite toute variable explicitement liée au prix
        if c_low in ["price", "price_num", "log_price"]:
            continue
        if "price" in c_low:
            drop_cols.append(c)
        if "service fee" in c_low or "service_fee" in c_low:
            drop_cols.append(c)

    drop_cols = sorted(set(drop_cols))
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df, drop_cols


def get_ml_features(df: pd.DataFrame, meta: dict):
    num = []
    cat = []

    for key in ["lat", "lon", "min_nights", "n_reviews", "rpm", "avail", "year"]:
        col = meta.get(key)
        if col and col in df.columns:
            num.append(col)

    for key in ["neigh_group", "room", "host_verif", "instant", "cancel"]:
        col = meta.get(key)
        if col and col in df.columns:
            cat.append(col)

    return num, cat


def build_pipeline(num_features, cat_features, model_name, rf_estimators=400):
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_features),
            ("cat", cat_pipe, cat_features)
        ],
        remainder="drop"
    )

    if model_name == "ridge":
        model = Ridge(alpha=1.0)
    else:
        model = RandomForestRegressor(
            n_estimators=rf_estimators,
            random_state=42,
            n_jobs=-1
        )

    return Pipeline(steps=[("preprocessor", pre), ("model", model)])


def iqr_bounds(series, k=1.5):
    s = series.dropna()
    if len(s) < 50:
        return None, None
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def safe_make_X(bundle, input_dict: dict) -> pd.DataFrame:
    """
    CORRECTION PRINCIPALE : Recrée X_new avec EXACTEMENT les features sauvegardées au training.
    """
    expected = bundle["expected_features"]  # ordre exact
    num_feats = bundle["features_num"]
    cat_feats = bundle["features_cat"]

    row = {}
    for col in expected:
        if col in input_dict:
            row[col] = input_dict[col]
        else:
            # manquante → NaN (sera imputée)
            row[col] = np.nan

    X_new = pd.DataFrame([row], columns=expected)

    # cast : assure num vs cat
    for c in num_feats:
        if c in X_new.columns:
            X_new[c] = pd.to_numeric(X_new[c], errors="coerce")

    for c in cat_feats:
        if c in X_new.columns:
            X_new[c] = X_new[c].astype(str)

    return X_new


def safe_predict_price(bundle, X_new: pd.DataFrame):
    """
    Prédiction stable :
    - prédire en log
    - clip log_pred dans plage train (anti explosions)
    - expm1 -> dollars
    """
    pipe = bundle["pipeline"]
    log_pred = float(pipe.predict(X_new)[0])

    y_min = bundle.get("y_min", None)
    y_max = bundle.get("y_max", None)
    if y_min is not None and y_max is not None:
        log_pred = float(np.clip(log_pred, y_min, y_max))

    price = float(np.expm1(log_pred))
    price = max(0.0, price)
    return log_pred, price


# =========================
# LOAD
# =========================
df, used_path = load_data()
if df is None:
    st.error("❌ Dataset introuvable. Place `Airbnb_Open_Data.csv` dans le projet.")
    st.stop()

meta = detect_columns(df)
if meta["price"] is None:
    st.error("❌ Colonne `price` introuvable après normalisation (lowercase).")
    st.stop()

st.caption(f"📌 Dataset chargé depuis : `{used_path}`")

# cible
df_base = df.copy()
df_base["price_num"] = df_base[meta["price"]].apply(clean_money_to_float)
df_valid = df_base.dropna(subset=["price_num"]).copy()
df_valid["log_price"] = np.log1p(df_valid["price_num"])

# anti leakage
df_safe, dropped_leak = leakage_safe_drop(df_valid)


# =========================
# MENU
# =========================
menu = st.sidebar.radio(
    "Navigation",
    ["🖼️ Slides (présentation)", "📦 Dataset", "🔎 EDA FULL", "🧼 Préprocessing", "🤖 Modélisation", "🔮 Prédiction"]
)

st.sidebar.markdown("---")
eda_sample = st.sidebar.slider("Échantillon EDA", 5000, min(120000, len(df_safe)), min(30000, len(df_safe)), 1000)
st.sidebar.caption("Astuce : baisse l’échantillon si c’est lent.")


# =========================
# SLIDES (ENRICHIES)
# =========================
if menu == "🖼️ Slides (présentation)":
    st.subheader("🖼️ Présentation (format slides dans l’app)")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Problème", "2. Données", "3. EDA", "4. Préprocessing", "5. Modèles", "6. Déploiement"
    ])

    with tab1:
        st.markdown("## 1) Contexte & objectif")
        st.markdown("""
**Objectif :** prédire le **prix d’une annonce Airbnb à NYC** à partir de variables observables.  
**Pourquoi ML :** relation non-linéaire, catégories, outliers, valeurs manquantes.  
**Cible :** `price` → conversion en `price_num` puis stabilisation via `log_price = log1p(price_num)`.
""")
        c1, c2, c3 = st.columns(3)
        c1.metric("Annonces (brut)", f"{df.shape[0]:,}".replace(",", " "))
        c2.metric("Prix valides", f"{df_valid.shape[0]:,}".replace(",", " "))
        c3.metric("Colonnes", f"{df.shape[1]:,}".replace(",", " "))

    with tab2:
        st.markdown("## 2) Dataset & qualité")
        st.markdown("""
- Normalisation des noms de colonnes (lowercase, espaces).  
- Conversion robuste du prix (`$`, `,`).  
- Suppression des lignes où le prix est manquant ou invalide.  
- Suppression des colonnes à risque de **fuite (leakage)** (ex: variables dérivées du prix).
""")
        if dropped_leak:
            st.warning("Colonnes retirées (leakage) : " + ", ".join(dropped_leak))
        st.dataframe(df.head(15), use_container_width=True)

    with tab3:
        st.markdown("## 3) EDA — ce qu’on cherche à démontrer")
        st.markdown("""
**EDA = preuve.**  
- Distribution du prix (asymétrie + outliers)  
- Différences par borough et room type  
- Impact des variables numériques (reviews, disponibilité, min nights)  
- Dimension géographique (lat/lon)
""")

    with tab4:
        st.markdown("## 4) Préprocessing (pipeline)")
        st.markdown("""
**Numériques** : médiane (robuste) + standardisation.  
**Catégorielles** : mode + one-hot (`handle_unknown='ignore'`).  
**Pourquoi pipeline :** même transformations au train et en production → reproductible.
""")

    with tab5:
        st.markdown("## 5) Modélisation & métriques")
        st.markdown("""
- **Ridge** : baseline rapide, interprétable.  
- **RandomForest** : capte non-linéarités & interactions.  
**Métriques :** R² / RMSE / MAE sur log + lecture en dollars.
""")
        st.info("La prédiction est stabilisée : clip du log_pred dans la plage vue au training.")

    with tab6:
        st.markdown("## 6) Déploiement Streamlit")
        st.markdown("""
- App interactive : dataset, EDA complet, entraînement, sauvegarde (`model.joblib`), prédiction.  
- Correction clé : **aucune erreur de features** grâce à la reconstruction `X_new` sur `expected_features`.
""")


# =========================
# DATASET
# =========================
elif menu == "📦 Dataset":
    st.subheader("📦 Dataset Explorer")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Shape**")
        st.write(df.shape)
    with c2:
        st.write("**Types**")
        st.dataframe(df.dtypes.astype(str).to_frame("dtype"), use_container_width=True)
    with c3:
        st.write("**Missing (Top 20)**")
        miss = df.isna().sum().sort_values(ascending=False).head(20)
        st.dataframe(miss.to_frame("missing_count"), use_container_width=True)

    st.write("Aperçu :")
    st.dataframe(df.head(40), use_container_width=True)


# =========================
# EDA FULL
# =========================
elif menu == "🔎 EDA FULL":
    st.subheader("🔎 EDA FULL — tous les graphes importants")

    if df_safe.empty:
        st.error("❌ Pas de lignes exploitables.")
        st.stop()

    df_eda = df_safe.sample(eda_sample, random_state=42) if len(df_safe) > eda_sample else df_safe.copy()

    # Filtres
    st.markdown("### 🎛️ Filtres")
    borough_col = meta.get("neigh_group")
    room_col = meta.get("room")

    f1, f2 = st.columns(2)
    with f1:
        if borough_col and borough_col in df_eda.columns:
            opts = ["all"] + sorted(df_eda[borough_col].dropna().astype(str).unique().tolist())
            sel_b = st.selectbox("neighbourhood group", opts)
        else:
            sel_b = "all"
    with f2:
        if room_col and room_col in df_eda.columns:
            opts = ["all"] + sorted(df_eda[room_col].dropna().astype(str).unique().tolist())
            sel_r = st.selectbox("room type", opts)
        else:
            sel_r = "all"

    df_f = df_eda.copy()
    if sel_b != "all" and borough_col:
        df_f = df_f[df_f[borough_col].astype(str) == sel_b]
    if sel_r != "all" and room_col:
        df_f = df_f[df_f[room_col].astype(str) == sel_r]

    st.caption(f"EDA sur {len(df_f):,} lignes".replace(",", " "))

    # 1) Prix
    st.markdown("---")
    st.markdown("## 1) Distribution du prix")
    fig = plt.figure(figsize=(11, 4))
    plt.hist(df_f["price_num"].dropna(), bins=70)
    plt.xlabel("price ($)")
    plt.ylabel("count")
    plt.title("Distribution price")
    plt.tight_layout()
    st.pyplot(fig)

    fig = plt.figure(figsize=(11, 4))
    plt.hist(df_f["log_price"].dropna(), bins=70)
    plt.xlabel("log1p(price)")
    plt.ylabel("count")
    plt.title("Distribution log(price)")
    plt.tight_layout()
    st.pyplot(fig)

    low, high = iqr_bounds(df_f["price_num"])
    if low is not None:
        out_rate = ((df_f["price_num"] < low) | (df_f["price_num"] > high)).mean() * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("IQR low", f"{low:,.1f}$")
        c2.metric("IQR high", f"{high:,.1f}$")
        c3.metric("% outliers", f"{out_rate:.2f}%")

    # 2) Catégories
    st.markdown("---")
    st.markdown("## 2) Catégories (comptage + prix médian)")
    if room_col and room_col in df_f.columns:
        vc = df_f[room_col].value_counts()
        fig = plt.figure(figsize=(10, 3))
        plt.bar(vc.index.astype(str), vc.values)
        plt.xticks(rotation=30, ha="right")
        plt.title("Room type count")
        plt.tight_layout()
        st.pyplot(fig)

        med = df_f.groupby(room_col)["price_num"].median().sort_values(ascending=False)
        fig = plt.figure(figsize=(10, 3))
        plt.bar(med.index.astype(str), med.values)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("median price ($)")
        plt.title("Room type vs median price")
        plt.tight_layout()
        st.pyplot(fig)

    if borough_col and borough_col in df_f.columns:
        vc = df_f[borough_col].value_counts()
        fig = plt.figure(figsize=(10, 3))
        plt.bar(vc.index.astype(str), vc.values)
        plt.xticks(rotation=30, ha="right")
        plt.title("Neighbourhood group count")
        plt.tight_layout()
        st.pyplot(fig)

        med = df_f.groupby(borough_col)["price_num"].median().sort_values(ascending=False)
        fig = plt.figure(figsize=(10, 3))
        plt.bar(med.index.astype(str), med.values)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("median price ($)")
        plt.title("Neighbourhood group vs median price")
        plt.tight_layout()
        st.pyplot(fig)

    # 3) Numériques vs prix
    st.markdown("---")
    st.markdown("## 3) Numériques vs prix (scatter)")
    candidates_num = [meta.get("min_nights"), meta.get("n_reviews"), meta.get("rpm"), meta.get("avail"), meta.get("year")]
    candidates_num = [c for c in candidates_num if c and c in df_f.columns]

    for col in candidates_num:
        tmp = df_f[[col, "price_num"]].dropna()
        if len(tmp) > 5000:
            tmp = tmp.sample(5000, random_state=42)
        fig = plt.figure(figsize=(8, 4))
        plt.scatter(tmp[col], tmp["price_num"], alpha=0.4)
        plt.xlabel(col)
        plt.ylabel("price ($)")
        plt.title(f"{col} vs price")
        plt.tight_layout()
        st.pyplot(fig)

    # 4) Corrélation
    st.markdown("---")
    st.markdown("## 4) Corrélations (numériques)")
    num_only = df_f.select_dtypes(include=[np.number])
    if num_only.shape[1] >= 2:
        corr = num_only.corr(numeric_only=True)
        if HAS_SNS:
            fig = plt.figure(figsize=(10, 6))
            sns.heatmap(corr, cmap="viridis")
            plt.title("Correlation heatmap")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            fig = plt.figure(figsize=(10, 6))
            plt.imshow(corr.values, aspect="auto")
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
            plt.yticks(range(len(corr.columns)), corr.columns)
            plt.title("Correlation heatmap")
            plt.tight_layout()
            st.pyplot(fig)

    # 5) Géo
    st.markdown("---")
    st.markdown("## 5) Géographie")
    lat_col, lon_col = meta.get("lat"), meta.get("lon")
    if lat_col and lon_col and lat_col in df_f.columns and lon_col in df_f.columns:
        map_df = df_f[[lat_col, lon_col]].dropna().rename(columns={lat_col: "lat", lon_col: "lon"})
        st.map(map_df.head(8000))
    else:
        st.info("lat/long non trouvés.")


# =========================
# PREPROCESSING
# =========================
elif menu == "🧼 Préprocessing":
    st.subheader("🧼 Préprocessing (pipeline exact)")

    df_work = df_safe.copy()
    num_features, cat_features = get_ml_features(df_work, meta)

    st.write("**Features numériques :**", num_features)
    st.write("**Features catégorielles :**", cat_features)

    st.markdown("""
### Pourquoi ces choix ?
- **Imputation médiane (num)** : robuste aux valeurs extrêmes.
- **StandardScaler (num)** : met les variables à échelle comparable (utile pour Ridge).
- **Most frequent (cat)** : remplace les NaN par la modalité la plus plausible.
- **OneHotEncoder ignore unknown** : aucune erreur si une catégorie nouvelle apparaît en production.
""")

    cols_show = [c for c in (num_features + cat_features + ["price_num", "log_price"]) if c in df_work.columns]
    st.dataframe(df_work[cols_show].head(30), use_container_width=True)


# =========================
# MODELISATION
# =========================
elif menu == "🤖 Modélisation":
    st.subheader("🤖 Modélisation (train + save) — sans erreurs de features")

    df_work = df_safe.copy()

    num_features, cat_features = get_ml_features(df_work, meta)
    if len(num_features) + len(cat_features) == 0:
        st.error("❌ Aucune feature ML reconnue. Vérifie les colonnes.")
        st.stop()

    X = df_work[num_features + cat_features].copy()
    y = df_work["log_price"].copy()

    model_choice = st.selectbox("Modèle", ["RandomForest", "Ridge"])
    test_size = st.slider("Taille test", 0.1, 0.4, 0.2, 0.05)
    rf_estimators = 400
    if model_choice == "RandomForest":
        rf_estimators = st.slider("n_estimators", 200, 900, 400, 50)

    if st.button("✅ Entraîner & sauvegarder", type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        pipe = build_pipeline(
            num_features=num_features,
            cat_features=cat_features,
            model_name=("ridge" if model_choice == "Ridge" else "rf"),
            rf_estimators=rf_estimators
        )

        with st.spinner("Entraînement..."):
            pipe.fit(X_train, y_train)
            pred_log = pipe.predict(X_test)

        r2 = r2_score(y_test, pred_log)
        rmse_log = float(np.sqrt(mean_squared_error(y_test, pred_log)))
        mae_log = float(mean_absolute_error(y_test, pred_log))

        y_true = np.expm1(y_test)
        y_pred = np.expm1(pred_log)
        rmse_d = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae_d = float(mean_absolute_error(y_true, y_pred))

        c1, c2, c3 = st.columns(3)
        c1.metric("R² (log)", f"{r2:.4f}")
        c2.metric("RMSE (log)", f"{rmse_log:.4f}")
        c3.metric("MAE (log)", f"{mae_log:.4f}")

        c4, c5 = st.columns(2)
        c4.metric("RMSE ($)", f"{rmse_d:,.2f}".replace(",", " "))
        c5.metric("MAE ($)", f"{mae_d:,.2f}".replace(",", " "))

        # Sauvegarde qui GARANTIT l'absence d'erreur "new features"
        bundle = {
            "pipeline": pipe,
            "features_num": num_features,
            "features_cat": cat_features,
            "expected_features": (num_features + cat_features),  # ordre exact !
            "y_min": float(np.min(y_train)),
            "y_max": float(np.max(y_train)),
        }
        joblib.dump(bundle, MODEL_PATH)
        st.success(f"✅ Modèle sauvegardé : `{MODEL_PATH}`")

        # Plot diagnostic
        k = min(700, len(y_true))
        idx = np.random.RandomState(42).choice(len(y_true), size=k, replace=False)
        fig = plt.figure(figsize=(7, 5))
        plt.scatter(y_true.values[idx], y_pred.values[idx], alpha=0.5)
        mn, mx = float(y_true.min()), float(y_true.max())
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel("True price ($)")
        plt.ylabel("Pred price ($)")
        plt.title("True vs Pred (dollars)")
        plt.tight_layout()
        st.pyplot(fig)


# =========================
# PREDICTION (SANS ERREUR)
# =========================
elif menu == "🔮 Prédiction":
    st.subheader("🔮 Prédiction — stable & sans erreur de features")

    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ modèle introuvable : `{MODEL_PATH}`. Entraîne d’abord dans Modélisation.")
        st.stop()

    bundle = joblib.load(MODEL_PATH)
    num_features = bundle["features_num"]
    cat_features = bundle["features_cat"]

    # Form
    with st.form("predict_form"):
        st.write("### Entrées")
        input_data = {}