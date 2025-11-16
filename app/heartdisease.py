# app.py (modified)
"""
Heart Disease Predictor (Streamlit)
- Uses Framingham dataset (TenYearCHD target)
- Trains two models (LogisticRegression + RandomForest) and builds an ensemble + simple rule-based points.
- Improved UI + gradient background + richer diagnosis output.
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from io import StringIO
import matplotlib.pyplot as plt

# --- Page config and custom CSS for background / layout ---
st.set_page_config(page_title="Heart Disease Predictor", layout="wide", page_icon="❤️")

# Custom CSS for dark background and light cards
st.markdown(
    """
    <style>
    /* Full-page dark background */
    .stApp {
        background-color: #000000;  /* pure black background */
        color: #f0f0f0;              /* light text */
    }

    /* Card container for sections */
    .card {
        background: rgba(20, 20, 20, 0.85);  /* semi-transparent dark grey */
        padding: 18px 24px;
        border-radius: 12px;
        box-shadow: 0 6px 24px rgba(255, 255, 255, 0.05);
        margin-bottom: 16px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(30, 30, 30, 0.9) !important;
        color: #f5f5f5;
    }

    /* General text, headers */
    h1, h2, h3, h4, h5, h6, p, label {
        color: #f5f5f5 !important;
    }

    /* Streamlit slider labels and select boxes */
    .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #f5f5f5 !important;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #1a73e8;
        color: white;
        border-radius: 8px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #1558b0;
        color: #fff;
    }

    /* Tables */
    table {
        color: #f5f5f5 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.title("❤️ Heart Disease Predictor")
st.caption("A demo app — educational only. Not medical advice.")
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Options")
threshold = st.sidebar.slider("Threshold for 'At risk' (probability %)", min_value=1, max_value=99, value=50)
mode = st.sidebar.selectbox("Input mode", ["Single patient (form)", "Batch (CSV upload)"])
show_data = st.sidebar.checkbox("Show raw dataset (first 200 rows)", value=False)
retrain = st.sidebar.checkbox("Retrain models on app start", value=False)
random_seed = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42)

# DATA_PATH default
DATA_PATH = "framingham.csv"

@st.cache_data(ttl=3600)
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    # We return preprocessor object for reuse by multiple models
    return preprocessor

@st.cache_resource
def train_models(df, retrain_flag=False, seed=42):
    np.random.seed(seed)

    df = df.copy()
    if "TenYearCHD" not in df.columns:
        raise ValueError("Dataset must contain 'TenYearCHD' target column.")
    df = df.dropna(subset=["TenYearCHD"])

    # Select features used (intersect with dataset columns)
    features = [
        "male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds",
        "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
        "sysBP", "diaBP", "BMI", "heartRate", "glucose"
    ]
    features = [f for f in features if f in df.columns]
    X = df[features].copy()
    y = df["TenYearCHD"].astype(int)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=seed, stratify=y)

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = []  # dataset columns are numeric/binary here

    preprocessor = build_pipeline(numeric_features, categorical_features)

    # Logistic Regression pipeline
    lr_pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=500, random_state=seed))
    ])

    # Random Forest pipeline
    rf_pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed))
    ])

    # Fit models
    lr_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    # Evaluate
    y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]
    # simple ensemble: average probs
    y_prob_ens = (y_prob_lr + y_prob_rf) / 2.0
    try:
        auc_lr = roc_auc_score(y_test, y_prob_lr)
    except Exception:
        auc_lr = float("nan")
    try:
        auc_rf = roc_auc_score(y_test, y_prob_rf)
    except Exception:
        auc_rf = float("nan")
    try:
        auc_ens = roc_auc_score(y_test, y_prob_ens)
    except Exception:
        auc_ens = float("nan")

    medians = X.median().to_dict()
    trained = {
        "lr": lr_pipeline,
        "rf": rf_pipeline,
        "features": features,
        "metrics": {"auc_lr": auc_lr, "auc_rf": auc_rf, "auc_ens": auc_ens},
        "defaults": medians
    }
    return trained

# Load dataset
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Dataset file not found at {DATA_PATH}. Place framingham.csv in the app directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

if show_data:
    st.write(df.head(200))

with st.spinner("Training models..."):
    model_bundle = train_models(df, retrain_flag=retrain, seed=int(random_seed))

lr_model = model_bundle["lr"]
rf_model = model_bundle["rf"]
features = model_bundle["features"]
metrics = model_bundle["metrics"]
defaults = model_bundle["defaults"]

# Display model metrics in sidebar
st.sidebar.markdown("### Model metrics (test set)")
st.sidebar.write(f"- Logistic ROC AUC: {metrics['auc_lr']:.3f}" if not np.isnan(metrics['auc_lr']) else "- Logistic ROC AUC: n/a")
st.sidebar.write(f"- RandomForest ROC AUC: {metrics['auc_rf']:.3f}" if not np.isnan(metrics['auc_rf']) else "- RandomForest ROC AUC: n/a")
st.sidebar.write(f"- Ensemble ROC AUC: {metrics['auc_ens']:.3f}" if not np.isnan(metrics['auc_ens']) else "- Ensemble ROC AUC: n/a")

st.markdown("---")

# --- Utility prediction function (ensemble + rule-based) ---
def rule_based_points(record):
    """
    A small clinical points system (illustrative, simplified):
    - age > 65 : +2
    - systolic BP > 140 : +1
    - diastolic BP > 90 : +1
    - BMI > 30 : +1
    - total cholesterol > 240 : +1
    - current smoker: +1
    - diabetes: +1
    Normalizes to [0,1] by dividing by max possible points (8 here).
    """
    pts = 0
    pts += 2 if record.get("age", 0) >= 65 else 0
    pts += 1 if record.get("sysBP", 0) >= 140 else 0
    pts += 1 if record.get("diaBP", 0) >= 90 else 0
    pts += 1 if record.get("BMI", 0) >= 30 else 0
    pts += 1 if record.get("totChol", 0) >= 240 else 0
    pts += 1 if record.get("currentSmoker", 0) == 1 else 0
    pts += 1 if record.get("diabetes", 0) == 1 else 0
    max_pts = 8.0
    return pts / max_pts, pts

def ensemble_predict(input_df):
    # returns ensemble probability, details
    # ensure columns ordering
    X = input_df[features].copy()
    prob_lr = lr_model.predict_proba(X)[:, 1]
    prob_rf = rf_model.predict_proba(X)[:, 1]
    prob_ens = (prob_lr + prob_rf) / 2.0

    # rule-based points
    rb_scores = []
    rb_raw = []
    for _, row in X.iterrows():
        sc_norm, pts = rule_based_points(row.to_dict())
        rb_scores.append(sc_norm)
        rb_raw.append(pts)

    rb_scores = np.array(rb_scores)
    # Combine ensemble probability with rule-based (weights can be tuned)
    # weighted_avg = 0.75 * prob_ens + 0.25 * rb_scores
    combined = (0.7 * prob_ens) + (0.3 * rb_scores)
    return {
        "prob_lr": prob_lr,
        "prob_rf": prob_rf,
        "prob_ens": prob_ens,
        "rule_score": rb_scores,
        "rule_points": rb_raw,
        "combined": combined
    }

def risk_tier(p):
    """Map probability [0,1] to risk tiers and a color"""
    if p >= 0.70:
        return "Very high", "red"
    if p >= 0.50:
        return "High", "orange"
    if p >= 0.20:
        return "Moderate", "yellow"
    return "Low", "green"

# --- Single patient form ---
if mode == "Single patient (form)":
    st.subheader("Enter patient details")
    card_style = '<div class="card">'
    st.markdown(card_style, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=int(defaults.get("age", 50)))
        male = st.selectbox("Sex", options=["Male", "Female"], index=0 if defaults.get("male", 1) == 1 else 1)
        education = st.number_input("Education (1-20)", min_value=0.0, max_value=20.0, value=float(defaults.get("education", 3)))
        currentSmoker = st.selectbox("Current smoker?", options=["No", "Yes"], index=1 if defaults.get("currentSmoker", 0)==1 else 0)
    with col2:
        cigsPerDay = st.number_input("Cigarettes per day", min_value=0.0, max_value=100.0, value=float(defaults.get("cigsPerDay", 0)))
        BPMeds = st.selectbox("On BP meds?", options=["No", "Yes"], index=1 if defaults.get("BPMeds", 0)==1 else 0)
        prevalentStroke = st.selectbox("Prior stroke?", options=["No", "Yes"], index=1 if defaults.get("prevalentStroke", 0)==1 else 0)
    with col3:
        prevalentHyp = st.selectbox("Prevalent hypertension?", options=["No", "Yes"], index=1 if defaults.get("prevalentHyp", 0)==1 else 0)
        diabetes = st.selectbox("Diabetes?", options=["No", "Yes"], index=1 if defaults.get("diabetes", 0)==1 else 0)
        totChol = st.number_input("Total cholesterol (mg/dL)", min_value=100.0, max_value=400.0, value=float(defaults.get("totChol", 200.0)))

    col4, col5, col6 = st.columns(3)
    with col4:
        sysBP = st.number_input("Systolic BP (mmHg)", min_value=80.0, max_value=250.0, value=float(defaults.get("sysBP", 120.0)))
        diaBP = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=160.0, value=float(defaults.get("diaBP", 80.0)))
    with col5:
        BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=float(defaults.get("BMI", 25.0)))
        heartRate = st.number_input("Heart rate (bpm)", min_value=30.0, max_value=200.0, value=float(defaults.get("heartRate", 75.0)))
    with col6:
        glucose = st.number_input("Glucose (mg/dL)", min_value=30.0, max_value=300.0, value=float(defaults.get("glucose", 80.0)))
        st.write("")  # spacer

    if st.button("Predict"):
        input_record = {
            "male": 1 if male == "Male" else 0,
            "age": age,
            "education": education,
            "currentSmoker": 1 if currentSmoker == "Yes" else 0,
            "cigsPerDay": cigsPerDay,
            "BPMeds": 1 if BPMeds == "Yes" else 0,
            "prevalentStroke": 1 if prevalentStroke == "Yes" else 0,
            "prevalentHyp": 1 if prevalentHyp == "Yes" else 0,
            "diabetes": 1 if diabetes == "Yes" else 0,
            "totChol": totChol,
            "sysBP": sysBP,
            "diaBP": diaBP,
            "BMI": BMI,
            "heartRate": heartRate,
            "glucose": glucose
        }
        # Keep only features expected by model
        input_df = pd.DataFrame([{k: input_record.get(k, defaults.get(k, 0)) for k in features}])
        preds = ensemble_predict(input_df)
        combined_prob = preds["combined"][0]
        tier, color = risk_tier(combined_prob)
        prob_pct = round(combined_prob * 100, 1)

        # Show summary metrics
        left, right = st.columns([2,1])
        with left:
            st.markdown(f"### Overall assessment — **{tier} risk**")
            st.markdown(f"<h2 style='color:{color};'>{prob_pct}% risk (combined)</h2>", unsafe_allow_html=True)
            st.write(f"Ensemble model probability: {preds['prob_ens'][0]*100:.1f}%")
            st.write(f"Rule-based points: {preds['rule_points'][0]} (normalized {preds['rule_score'][0]*100:.1f}%)")
            st.write(f"Threshold you set: {threshold}% => classification: **{'At risk' if prob_pct >= threshold else 'Low risk'}**")
        with right:
            # small horizontal gauge using st.progress (visual)
            st.write("Risk gauge")
            gauge = min(100, max(0, int(prob_pct)))
            st.progress(gauge)

        st.markdown("#### Inputs")
        st.table(input_df.T.rename(columns={0: "value"}))

        # Feature importance: logistic coefficients + RF importances
        # Extract coefficients (after preprocessor, get names via numeric features)
        # For logistic, coefficients are available as clf.coef_ after preprocessing transforms
        try:
            # Build a transformed sample to extract feature order from preprocessor
            preproc = lr_model.named_steps["preproc"]
            # Use a single-row to obtain transformed column count (names not trivial); we'll show original features
            lr_coef = lr_model.named_steps["clf"].coef_.flatten()
            rf_imp = rf_model.named_steps["clf"].feature_importances_
            # They align with numeric_features order; we will display them side-by-side
            feat_names = features  # original features
            imp_df = pd.DataFrame({
                "feature": feat_names,
                "lr_coef_abs": np.abs(lr_coef[:len(feat_names)] if len(lr_coef) >= len(feat_names) else np.concatenate([lr_coef, np.zeros(len(feat_names)-len(lr_coef))])),
                "rf_imp": rf_imp[:len(feat_names)] if len(rf_imp) >= len(feat_names) else np.concatenate([rf_imp, np.zeros(len(feat_names)-len(rf_imp))]),
            })
            imp_df["combined_importance"] = (imp_df["lr_coef_abs"] / (imp_df["lr_coef_abs"].sum()+1e-9)) + (imp_df["rf_imp"] / (imp_df["rf_imp"].sum()+1e-9))
            imp_df = imp_df.sort_values("combined_importance", ascending=False).reset_index(drop=True)
            st.markdown("### Drivers of risk (model-derived)")
            st.dataframe(imp_df[["feature", "lr_coef_abs", "rf_imp"]].head(12).rename(columns={
                "lr_coef_abs": "Abs(LR coef)",
                "rf_imp": "RF importance"
            }))
            # Bar chart
            fig, ax = plt.subplots(figsize=(8,3))
            ax.barh(imp_df["feature"].head(10)[::-1], imp_df["combined_importance"].head(10)[::-1])
            ax.set_xlabel("Combined importance (normed)")
            ax.set_title("Top features contributing to predicted risk")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not compute feature importance display: {e}")

        st.info("This combined assessment uses two trained models and a simple rule-based points system. It is for educational/demonstration purposes only — not a clinical diagnosis.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Batch CSV upload ---
else:
    st.subheader("Batch predictions via CSV upload")
    st.markdown("CSV should contain the predictor columns used by the model. Required columns:")
    st.write(features)
    sample = st.button("Show sample input & download sample CSV")
    if sample:
        sample_df = pd.DataFrame([{
            f: float(defaults.get(f, 0)) for f in features
        }])
        st.write(sample_df)
        st.download_button("Download sample CSV", data=sample_df.to_csv(index=False), file_name="sample_heart_input.csv", mime="text/csv")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            st.stop()

        missing = [c for c in features if c not in batch_df.columns]
        if missing:
            st.error(f"Uploaded file is missing required columns: {missing}")
        else:
            preds = ensemble_predict(batch_df)
            batch_out = batch_df.copy()
            batch_out["probability_pct"] = np.round(preds["combined"] * 100, 1)
            batch_out["rule_points"] = preds["rule_points"]
            batch_out["classification"] = np.where(batch_out["probability_pct"] >= threshold, "At risk", "Low risk")
            st.success("Predictions complete")
            st.dataframe(batch_out)
            csv = batch_out.to_csv(index=False)
            st.download_button("Download results CSV", data=csv, file_name="heart_predictions.csv", mime="text/csv")

st.markdown("---")
st.markdown(
    """
**Disclaimer:** This app is an educational demo only. It is not a medical device and is not for clinical use.
If you have health concerns, consult a qualified healthcare professional.
"""
)
