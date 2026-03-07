import os
import time
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
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Optional (only if installed)
try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False


# =============================================================================
# PATHS
# =============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CANDIDATES = [
    os.path.join(HERE, "data", "raw", "Airbnb_Open_Data.csv"),
    os.path.join(HERE, "data", "row", "Airbnb_Open_Data.csv"),
    os.path.join(HERE, "data", "Airbnb_Open_Data.csv"),
    os.path.join(HERE, "Airbnb_Open_Data.csv"),
    os.path.join(HERE, "Airbnb_Open_Data.xlsx"),
    os.path.join(HERE, "Airbnb_Open_Data.xls"),
]
MODEL_PATH = os.path.join(HERE, "model.joblib")


# =============================================================================
# APP CONFIG (pro + clean)
# =============================================================================
st.set_page_config(page_title="Airbnb NYC — Analyse & Prédiction", layout="wide")

st.markdown(
    """
    <style>
      .small-muted { color: #8a8f98; font-size: 0.9rem; }
      .kpi-card { border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 14px 16px; }
      .section-title { margin-top: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Airbnb NYC — Analyse • Modélisation • Prédiction")
st.caption("Application Streamlit pour explorer le dataset, analyser les variables et prédire le prix d’une annonce.")


# =============================================================================
# UTILS
# =============================================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les noms de colonnes: lowercase + espaces normalisés."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


def clean_money_to_float(x):
    """Convertit '$1,234.00' -> 1234.0 (robuste)."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_float_any(x):
    """Accepte '40,72' ou '40.72'."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace(" ", "")
    if s == "":
        return np.nan
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def pick_existing(cols_set, *names):
    for n in names:
        n2 = n.strip().lower()
        if n2 in cols_set:
            return n2
    return None


def detect_columns(df: pd.DataFrame):
    cols = set(df.columns)
    return {
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


def leakage_safe_drop(df: pd.DataFrame):
    """
    Evite la fuite (leakage) : supprime les colonnes liées au prix si présentes
    (ex: 'service fee', 'price something').
    """
    df = df.copy()
    drop_cols = []
    for c in df.columns:
        c_low = c.lower()
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
    num, cat = [], []

    for key in ["lat", "lon", "min_nights", "n_reviews", "rpm", "avail", "year"]:
        col = meta.get(key)
        if col and col in df.columns:
            num.append(col)

    for key in ["neigh_group", "neigh", "room", "host_verif", "instant", "cancel"]:
        col = meta.get(key)
        if col and col in df.columns:
            cat.append(col)

    return num, cat


def build_pipeline(num_features, cat_features, model_name: str, rf_estimators=400):
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
            ("cat", cat_pipe, cat_features),
        ],
        remainder="drop"
    )

    if model_name == "ridge":
        model = Ridge(alpha=1.0)
    elif model_name == "hgb":
        # très bon compromis perf/vitesse sans dépendance externe (pas besoin de xgboost)
        model = HistGradientBoostingRegressor(
            random_state=42,
            max_depth=6,
            learning_rate=0.08,
            max_iter=350,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=int(rf_estimators),
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
    """Recrée X_new avec EXACTEMENT les features du training (ordre + colonnes)."""
    expected = bundle["expected_features"]
    num_feats = bundle["features_num"]
    cat_feats = bundle["features_cat"]

    row = {}
    for col in expected:
        row[col] = input_dict.get(col, np.nan)

    X_new = pd.DataFrame([row], columns=expected)

    for c in num_feats:
        if c in X_new.columns:
            X_new[c] = pd.to_numeric(X_new[c], errors="coerce")

    for c in cat_feats:
        if c in X_new.columns:
            # garde NaN si vide, sinon string
            X_new[c] = X_new[c].apply(lambda v: np.nan if (pd.isna(v) or str(v).strip() == "") else str(v))

    return X_new


def safe_predict_price(bundle, X_new: pd.DataFrame):
    """Prédiction stable: log_pred -> clip -> expm1."""
    pipe = bundle["pipeline"]
    log_pred = float(pipe.predict(X_new)[0])

    y_min, y_max = bundle.get("y_min"), bundle.get("y_max")
    if y_min is not None and y_max is not None:
        log_pred = float(np.clip(log_pred, y_min, y_max))

    price = float(np.expm1(log_pred))
    return log_pred, max(0.0, price)


@st.cache_data(show_spinner=False)
def load_data():
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            if path.lower().endswith(".csv"):
                df_ = pd.read_csv(path, low_memory=False)
            else:
                df_ = pd.read_excel(path)
            df_ = normalize_columns(df_)
            return df_, path
    return None, None


# =============================================================================
# LOAD DATA + BASIC PREP
# =============================================================================
df, used_path = load_data()
if df is None:
    st.error("Dataset introuvable. Place `Airbnb_Open_Data.csv` (ou .xlsx) dans le dossier `app/` ou `app/data/`.")
    st.stop()

meta = detect_columns(df)
if meta["price"] is None:
    st.error("Colonne `price` introuvable après normalisation (lowercase).")
    st.stop()

df_base = df.copy()
df_base["price_num"] = df_base[meta["price"]].apply(clean_money_to_float)
df_valid = df_base.dropna(subset=["price_num"]).copy()
df_valid["log_price"] = np.log1p(df_valid["price_num"])

df_safe, dropped_leak = leakage_safe_drop(df_valid)

num_features, cat_features = get_ml_features(df_safe, meta)
all_features = num_features + cat_features

st.caption(f"Dataset: `{used_path}` • lignes: {len(df_safe):,} • variables: {df.shape[1]}".replace(",", " "))


# =============================================================================
# SIDEBAR NAV
# =============================================================================
page = st.sidebar.radio(
    "Navigation",
    ["Présentation", "Dataset", "EDA (complet)", "Prétraitement", "Modélisation", "Prédiction"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Performance")
eda_sample = st.sidebar.slider(
    "Taille échantillon (EDA)",
    5_000,
    int(min(120_000, len(df_safe))),
    int(min(25_000, len(df_safe))),
    1_000,
)
train_sample = st.sidebar.slider(
    "Taille échantillon (FAST)",
    5_000,
    int(min(80_000, len(df_safe))),
    int(min(20_000, len(df_safe))),
    1_000,
)
st.sidebar.caption("Conseil: commence en mode FAST pour entraîner rapidement.")


# =============================================================================
# PAGE: PRESENTATION (PRO)
# =============================================================================
if page == "Présentation":
    st.subheader("Vue d’ensemble")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annonces (brut)", f"{df.shape[0]:,}".replace(",", " "))
    c2.metric("Prix valides", f"{df_valid.shape[0]:,}".replace(",", " "))
    c3.metric("Variables", f"{df.shape[1]:,}".replace(",", " "))
    c4.metric("Variables retirées (fuite)", f"{len(dropped_leak)}")

    st.markdown("### Objectif")
    st.write(
        "Prédire le **prix d’une annonce Airbnb à New York** à partir de variables observables "
        "(localisation, type de logement, disponibilité, avis…). "
        "Pour stabiliser la variance du prix et limiter l’impact des valeurs extrêmes, "
        "la cible utilisée pour l’apprentissage est `log1p(price)`."
    )

    st.markdown("### Méthodologie")
    st.markdown(
        """
- **Nettoyage** : conversion du prix, suppression des valeurs invalides.
- **Prétraitement** : imputation (médiane / mode), standardisation des variables numériques, encodage one-hot des catégories.
- **Modélisation** : comparaison de plusieurs modèles (Ridge, RandomForest, Gradient Boosting).
- **Évaluation** : R², MAE et RMSE (sur log, puis interprétation en dollars).
- **Déploiement** : entraînement + sauvegarde du modèle et page de prédiction.
        """
    )

    if dropped_leak:
        st.info("Variables supprimées pour éviter la fuite d’information : " + ", ".join(dropped_leak))

    with st.expander("Limites et pistes d’amélioration"):
        st.markdown(
            """
- Le dataset peut contenir du bruit, des valeurs manquantes et des catégories rares.
- Certains facteurs réels ne sont pas inclus (qualité des photos, réputation, saisonnalité, événements…).
- Pour améliorer les performances, on peut tester des modèles plus puissants (Gradient Boosting, XGBoost/LightGBM si autorisé).
            """
        )


# =============================================================================
# PAGE: DATASET
# =============================================================================
elif page == "Dataset":
    st.subheader("Exploration du dataset")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Shape**")
        st.write(df.shape)
    with c2:
        st.write("**Types**")
        st.dataframe(df.dtypes.astype(str).to_frame("dtype"), use_container_width=True)
    with c3:
        st.write("**Valeurs manquantes (Top 20)**")
        miss = df.isna().sum().sort_values(ascending=False).head(20)
        st.dataframe(miss.to_frame("missing_count"), use_container_width=True)

    st.markdown("### Aperçu")
    st.dataframe(df.head(40), use_container_width=True)

    st.markdown("### Variables utilisées par le modèle")
    st.write("**Numériques**:", num_features if num_features else "—")
    st.write("**Catégorielles**:", cat_features if cat_features else "—")


# =============================================================================
# PAGE: EDA COMPLET
# =============================================================================
elif page == "EDA (complet)":
    st.subheader("Analyse exploratoire (EDA)")

    if df_safe.empty:
        st.error("Pas de lignes exploitables après nettoyage.")
        st.stop()

    df_eda = df_safe.sample(eda_sample, random_state=42) if len(df_safe) > eda_sample else df_safe.copy()

    # Filters
    st.markdown("### Filtres")
    borough_col = meta.get("neigh_group")
    room_col = meta.get("room")

    f1, f2 = st.columns(2)
    with f1:
        if borough_col and borough_col in df_eda.columns:
            opts = ["Tous"] + sorted(df_eda[borough_col].dropna().astype(str).unique().tolist())
            sel_b = st.selectbox("Neighbourhood group (borough)", opts)
        else:
            sel_b = "Tous"
    with f2:
        if room_col and room_col in df_eda.columns:
            opts = ["Tous"] + sorted(df_eda[room_col].dropna().astype(str).unique().tolist())
            sel_r = st.selectbox("Room type", opts)
        else:
            sel_r = "Tous"

    df_f = df_eda.copy()
    if sel_b != "Tous" and borough_col:
        df_f = df_f[df_f[borough_col].astype(str) == sel_b]
    if sel_r != "Tous" and room_col:
        df_f = df_f[df_f[room_col].astype(str) == sel_r]

    st.caption(f"EDA sur {len(df_f):,} lignes".replace(",", " "))

    # 1) Price distribution
    st.markdown("---")
    st.markdown("## 1) Distribution du prix")
    fig = plt.figure(figsize=(11, 4))
    plt.hist(df_f["price_num"].dropna(), bins=70)
    plt.xlabel("price ($)")
    plt.ylabel("count")
    plt.title("Distribution du prix")
    plt.tight_layout()
    st.pyplot(fig)

    fig = plt.figure(figsize=(11, 4))
    plt.hist(df_f["log_price"].dropna(), bins=70)
    plt.xlabel("log1p(price)")
    plt.ylabel("count")
    plt.title("Distribution de log1p(price)")
    plt.tight_layout()
    st.pyplot(fig)

    low, high = iqr_bounds(df_f["price_num"])
    if low is not None:
        out_rate = ((df_f["price_num"] < low) | (df_f["price_num"] > high)).mean() * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Seuil IQR bas", f"{low:,.1f}$")
        c2.metric("Seuil IQR haut", f"{high:,.1f}$")
        c3.metric("Outliers (IQR)", f"{out_rate:.2f}%")

    # 2) Categories vs price
    st.markdown("---")
    st.markdown("## 2) Variables catégorielles")
    if room_col and room_col in df_f.columns:
        vc = df_f[room_col].value_counts().head(15)
        fig = plt.figure(figsize=(10, 3))
        plt.bar(vc.index.astype(str), vc.values)
        plt.xticks(rotation=30, ha="right")
        plt.title("Room type (top 15) — nombre d'annonces")
        plt.tight_layout()
        st.pyplot(fig)

        med = df_f.groupby(room_col)["price_num"].median().sort_values(ascending=False).head(15)
        fig = plt.figure(figsize=(10, 3))
        plt.bar(med.index.astype(str), med.values)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("median price ($)")
        plt.title("Room type (top 15) — prix médian")
        plt.tight_layout()
        st.pyplot(fig)

    if borough_col and borough_col in df_f.columns:
        vc = df_f[borough_col].value_counts()
        fig = plt.figure(figsize=(10, 3))
        plt.bar(vc.index.astype(str), vc.values)
        plt.xticks(rotation=30, ha="right")
        plt.title("Neighbourhood group — nombre d'annonces")
        plt.tight_layout()
        st.pyplot(fig)

        med = df_f.groupby(borough_col)["price_num"].median().sort_values(ascending=False)
        fig = plt.figure(figsize=(10, 3))
        plt.bar(med.index.astype(str), med.values)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("median price ($)")
        plt.title("Neighbourhood group — prix médian")
        plt.tight_layout()
        st.pyplot(fig)

    neigh_col = meta.get("neigh")
    if neigh_col and neigh_col in df_f.columns:
        top_neigh = (
            df_f.groupby(neigh_col)["price_num"]
            .median()
            .sort_values(ascending=False)
            .head(15)
        )
        fig = plt.figure(figsize=(10, 3))
        plt.bar(top_neigh.index.astype(str), top_neigh.values)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("median price ($)")
        plt.title("Neighbourhood (top 15) — prix médian")
        plt.tight_layout()
        st.pyplot(fig)

    # 3) Numeric vs price
    st.markdown("---")
    st.markdown("## 3) Variables numériques vs prix")
    candidates_num = [meta.get("min_nights"), meta.get("n_reviews"), meta.get("rpm"), meta.get("avail"), meta.get("year")]
    candidates_num = [c for c in candidates_num if c and c in df_f.columns]

    for col in candidates_num:
        tmp = df_f[[col, "price_num"]].dropna()
        if len(tmp) > 8000:
            tmp = tmp.sample(8000, random_state=42)
        fig = plt.figure(figsize=(8, 4))
        plt.scatter(tmp[col], tmp["price_num"], alpha=0.35)
        plt.xlabel(col)
        plt.ylabel("price ($)")
        plt.title(f"{col} vs price")
        plt.tight_layout()
        st.pyplot(fig)

    # 4) Correlation
    st.markdown("---")
    st.markdown("## 4) Corrélations (numériques)")
    num_only = df_f.select_dtypes(include=[np.number])
    if num_only.shape[1] >= 2:
        corr = num_only.corr(numeric_only=True)
        if HAS_SNS:
            fig = plt.figure(figsize=(10, 6))
            sns.heatmap(corr, cmap="viridis")
            plt.title("Heatmap de corrélation")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            fig = plt.figure(figsize=(10, 6))
            plt.imshow(corr.values, aspect="auto")
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
            plt.yticks(range(len(corr.columns)), corr.columns)
            plt.title("Heatmap de corrélation")
            plt.tight_layout()
            st.pyplot(fig)

    # 5) Geo
    st.markdown("---")
    st.markdown("## 5) Dimension géographique")
    lat_col, lon_col = meta.get("lat"), meta.get("lon")
    if lat_col and lon_col and lat_col in df_f.columns and lon_col in df_f.columns:
        map_df = df_f[[lat_col, lon_col]].dropna().rename(columns={lat_col: "lat", lon_col: "lon"})
        st.map(map_df.head(10_000))
        st.caption("Carte (échantillon) : affichage des points lat/lon.")
    else:
        st.info("Colonnes latitude/longitude non trouvées.")


# =============================================================================
# PAGE: PRETRAITEMENT
# =============================================================================
elif page == "Prétraitement":
    st.subheader("Prétraitement (pipeline)")

    if not all_features:
        st.error("Aucune feature ML reconnue. Vérifie les colonnes.")
        st.stop()

    st.markdown("### Variables utilisées")
    st.write("**Numériques :**", num_features)
    st.write("**Catégorielles :**", cat_features)

    st.markdown("### Transformations")
    st.markdown(
        """
- **Numériques** : imputation par la médiane + standardisation (utile pour Ridge).
- **Catégorielles** : imputation par la modalité la plus fréquente + encodage One-Hot.
- **Robustesse production** : `handle_unknown='ignore'` évite les erreurs si une nouvelle catégorie apparaît.
        """
    )

    cols_show = [c for c in (all_features + ["price_num", "log_price"]) if c in df_safe.columns]
    st.dataframe(df_safe[cols_show].head(30), use_container_width=True)


# =============================================================================
# PAGE: MODELISATION
# =============================================================================
elif page == "Modélisation":
    st.subheader("Entraînement & sauvegarde du modèle")

    if not all_features:
        st.error("Aucune feature ML reconnue. Vérifie les colonnes du dataset.")
        st.stop()

    st.info("Conseil : commence avec **Ridge** (très rapide), puis teste **Gradient Boosting** ou **RandomForest**.")

    model_choice = st.selectbox("Modèle", ["Ridge", "Gradient Boosting (rapide)", "RandomForest"])
    mode = st.radio("Mode d'entraînement", ["FAST (échantillon)", "FULL (plus lent)"], horizontal=True)
    test_size = st.slider("Taille du jeu de test", 0.1, 0.4, 0.2, 0.05)

    rf_estimators = 400
    if model_choice == "RandomForest":
        rf_estimators = st.slider("n_estimators", 200, 900, 350 if mode.startswith("FAST") else 500, 50)

    # Data for training
    df_train = df_safe.copy()

    if mode.startswith("FAST"):
        if len(df_train) > train_sample:
            df_train = df_train.sample(train_sample, random_state=42)

    X = df_train[all_features].copy()
    y = df_train["log_price"].copy()

    # default stats for prediction page
    train_num_median = X[num_features].median(numeric_only=True).to_dict() if num_features else {}
    train_cat_mode = {}
    for c in cat_features:
        s = X[c].dropna().astype(str)
        train_cat_mode[c] = s.mode().iloc[0] if len(s) else ""

    if st.button("Entraîner & sauvegarder", type="primary"):
        t0 = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model_key = "ridge" if model_choice == "Ridge" else ("hgb" if "Gradient" in model_choice else "rf")
        pipe = build_pipeline(
            num_features=num_features,
            cat_features=cat_features,
            model_name=model_key,
            rf_estimators=rf_estimators,
        )

        with st.spinner("Entraînement en cours..."):
            pipe.fit(X_train, y_train)
            pred_log = pipe.predict(X_test)

        # Metrics
        r2 = float(r2_score(y_test, pred_log))
        rmse_log = float(np.sqrt(mean_squared_error(y_test, pred_log)))
        mae_log = float(mean_absolute_error(y_test, pred_log))

        y_true = np.expm1(y_test)
        y_pred = np.expm1(pred_log)
        rmse_d = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae_d = float(mean_absolute_error(y_true, y_pred))
        dur = time.time() - t0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² (log)", f"{r2:.3f}")
        c2.metric("RMSE (log)", f"{rmse_log:.3f}")
        c3.metric("MAE (log)", f"{mae_log:.3f}")
        c4.metric("Durée (s)", f"{dur:.1f}")

        c5, c6 = st.columns(2)
        c5.metric("RMSE ($)", f"{rmse_d:,.2f}".replace(",", " "))
        c6.metric("MAE ($)", f"{mae_d:,.2f}".replace(",", " "))

        # Save bundle
        bundle = {
            "pipeline": pipe,
            "features_num": num_features,
            "features_cat": cat_features,
            "expected_features": all_features,  # order matters
            "y_min": float(np.min(y_train)),
            "y_max": float(np.max(y_train)),
            "trained_rows": int(len(df_train)),
            "model_name": model_choice,
            "train_num_median": train_num_median,
            "train_cat_mode": train_cat_mode,
        }
        joblib.dump(bundle, MODEL_PATH)
        st.success(f"Modèle sauvegardé : `{MODEL_PATH}`")

        # Diagnostics plot (fixes ndarray/.values issue)
        k = min(800, len(y_true))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y_true), size=k, replace=False) if len(y_true) > k else np.arange(len(y_true))

        y_true_arr = np.asarray(y_true)[idx]
        y_pred_arr = np.asarray(y_pred)[idx]

        fig = plt.figure(figsize=(7, 5))
        plt.scatter(y_true_arr, y_pred_arr, alpha=0.45)
        mn, mx = float(np.min(y_true_arr)), float(np.max(y_true_arr))
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel("True price ($)")
        plt.ylabel("Pred price ($)")
        plt.title("True vs Pred ($)")
        plt.tight_layout()
        st.pyplot(fig)


# =============================================================================
# PAGE: PREDICTION
# =============================================================================
elif page == "Prédiction":
    st.subheader("Prédire le prix d’une annonce")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Modèle introuvable : `{MODEL_PATH}`. Va dans **Modélisation** et entraîne un modèle.")
        st.stop()

    bundle = joblib.load(MODEL_PATH)
    num_features_b = bundle["features_num"]
    cat_features_b = bundle["features_cat"]

    st.success(
        f"Modèle chargé : **{bundle.get('model_name','—')}** • entraîné sur **{bundle.get('trained_rows','?')}** lignes"
    )

    # Defaults
    num_defaults = bundle.get("train_num_median", {})
    cat_defaults = bundle.get("train_cat_mode", {})

    # Helper: quick example from dataset
    with st.expander("Insérer un exemple depuis le dataset"):
        if st.button("Remplir avec une annonce aléatoire"):
            row = df_safe.sample(1, random_state=int(time.time()) % 10_000)[all_features].iloc[0].to_dict()
            st.session_state["_example_row"] = row
        st.caption("Cela remplit automatiquement le formulaire avec des valeurs réalistes.")

    example_row = st.session_state.get("_example_row", {})

    st.markdown("### Saisie")
    st.markdown('<div class="small-muted">Les champs vides seront imputés automatiquement (médiane/mode).</div>', unsafe_allow_html=True)

    # Build category options from dataset (limited for speed)
    cat_options = {}
    for c in cat_features_b:
        s = df_safe[c].dropna().astype(str)
        # limit options to avoid huge dropdowns
        opts = sorted(s.unique().tolist())
        if len(opts) > 200:
            opts = opts[:200]
        cat_options[c] = [""] + opts

    input_dict = {}

    left, right = st.columns(2)

    with left:
        st.markdown("#### Numériques")
        for c in num_features_b:
            default = example_row.get(c, num_defaults.get(c, np.nan))
            default_str = "" if pd.isna(default) else f"{float(default):.4f}"
            val = st.text_input(c, value=default_str, placeholder="ex: 40.72")
            input_dict[c] = parse_float_any(val)

    with right:
        st.markdown("#### Catégorielles")
        for c in cat_features_b:
            default = example_row.get(c, cat_defaults.get(c, ""))
            opts = cat_options.get(c, [""])
            if str(default) not in opts:
                opts = [str(default)] + opts  # ensure default is selectable
            sel = st.selectbox(c, options=opts, index=opts.index(str(default)) if str(default) in opts else 0)
            input_dict[c] = sel

    st.markdown("---")
    if st.button("Prédire", type="primary"):
        X_new = safe_make_X(bundle, input_dict)
        log_pred, price = safe_predict_price(bundle, X_new)

        c1, c2 = st.columns(2)
        c1.metric("Prix prédit ($)", f"{price:,.2f}".replace(",", " "))
        c2.metric("log(price) prédit", f"{log_pred:.4f}")

        st.markdown("### Valeurs envoyées au modèle")
        st.dataframe(X_new, use_container_width=True)
