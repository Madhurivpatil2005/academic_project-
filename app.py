# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Page config + basic CSS
# ---------------------------
st.set_page_config(page_title="Rice Fertilizer Optimizer", page_icon="🌾", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #020617, #020617, #0f172a);
}
[data-testid="stSidebar"] { background-color: #020617; }
html, body, [class*="css"]  { color: #e5e7eb; }
.card { border-radius:18px; padding:1.2rem 1.5rem;
       background: linear-gradient(135deg,#16a34a,#22c55e); color:white;
       box-shadow:0 18px 45px rgba(0,0,0,0.45); }
.card-dark { border-radius:18px; padding:1.2rem 1.5rem; background:#020617;
            border:1px solid #1f2937; box-shadow:0 12px 30px rgba(0,0,0,0.6); }
.card-value { font-size:2.1rem; font-weight:700; }
.card-label { font-size:0.95rem; opacity:0.9; }
.block-container { padding-top:1.2rem; padding-bottom:1.2rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helpers: load model & data
# ---------------------------
@st.cache_resource
def load_model_cached(model_path="best_fertilizer_model.pkl"):
    # returns the sklearn MultiOutputRegressor (or any saved model)
    return joblib.load(model_path)

@st.cache_resource
def load_dataset(path="final_davangere_kvk_1000_rows.csv"):
    df = pd.read_csv(path)
    # detect feature/target column names (support both variants)
    # Primary fallback: columns N,P,K,pH and urea,ssp,mop
    if set(["soil_N_kg_per_ha", "soil_P_kg_per_ha", "soil_K_kg_per_ha", "soil_pH"]).issubset(df.columns):
        features = ["soil_N_kg_per_ha", "soil_P_kg_per_ha", "soil_K_kg_per_ha", "soil_pH"]
        targets = ["urea_kg_per_acre", "ssp_kg_per_acre", "mop_kg_per_acre"]
    else:
        # fallback to simple names used earlier
        features = ["N", "P", "K", "pH"]
        targets = ["urea", "ssp", "mop"]
    return df, features, targets

# load
model = load_model_cached()   # Make sure "best_fertilizer_model.pkl" exists
df, FEATURE_COLS, TARGET_COLS = load_dataset()

# small check and rename if dataset uses uppercase etc (not required but helpful)
FEATURE_COLS = list(FEATURE_COLS)
TARGET_COLS = list(TARGET_COLS)

# ---------------------------
# SHAP helpers (avoid hashing model by naming param _model)
# ---------------------------
@st.cache_resource
def create_shap_explainers(_model, data_frame, feature_cols, background_samples=200):
    """
    Returns:
      - X_background (DataFrame)
      - tuple per target: (explainer, shap_values)
    Note: _model is intentionally named with leading underscore to avoid Streamlit hashing errors.
    """
    # sample background
    n = min(background_samples, len(data_frame))
    X_background = data_frame[feature_cols].sample(n, random_state=42)

    # extract the estimators (assumes MultiOutputRegressor or list-like)
    # If model is not multi-output, we fallback: treat single estimator for all targets
    try:
        estimators = _model.estimators_
    except Exception:
        # fallback: treat whole model as single tree-based regressor
        estimators = [_model] * len(feature_cols)

    explainers = []
    shap_values_all = []
    for est in estimators[: len(TARGET_COLS) ]:
        expl = shap.TreeExplainer(est)
        sv = expl.shap_values(X_background)  # returns ndarray
        explainers.append(expl)
        shap_values_all.append(sv)

    return X_background, list(zip(explainers, shap_values_all))


def plot_shap_summary(shap_vals, X_bg, feature_names, title="SHAP summary (beeswarm)"):
    """Plot a beeswarm (matplotlib) using shap.summary_plot and return the figure."""
    plt.figure(figsize=(8, 3.2))  # wide and short to fit in Streamlit page
    # shap.summary_plot uses the global matplotlib figure
    shap.summary_plot(shap_vals, X_bg[feature_names], show=False)
    plt.title(title)
    fig = plt.gcf()
    return fig


def plot_shap_bar_single(shap_vals_single, feature_names, title="SHAP per feature (single sample)"):
    """Bar chart for single sample shap contributions - shap_vals_single is 1D array length = n_features"""
    vals = np.array(shap_vals_single).flatten()
    df_plot = pd.DataFrame({"feature": feature_names, "shap": vals})
    df_plot = df_plot.reindex(df_plot["shap"].abs().sort_values(ascending=False).index)  # sort by impact
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(df_plot["feature"], df_plot["shap"])
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title(title)
    plt.tight_layout()
    return fig

def build_text_explanation(feature_names, shap_vals_single, feature_values, top_k=3):
    """
    Return a short human-readable explanation like:
      "Why? N decreases predicted dose (SHAP -4.86) & P decreases ... "
    shap_vals_single: 1D array of shap contributions for the sample
    feature_values: dict {feature: value}
    """
    shap_arr = np.array(shap_vals_single).flatten()
    df_tmp = pd.DataFrame({
        "feature": feature_names,
        "shap": shap_arr,
        "value": [feature_values.get(f, None) for f in feature_names]
    })
    # pick top_k by absolute impact
    top = df_tmp.reindex(df_tmp["shap"].abs().sort_values(ascending=False).index).head(top_k)
    phrases = []
    for _, r in top.iterrows():
        direction = "increases" if r["shap"] > 0 else "decreases"
        phrases.append(f"{r['feature']} {direction} the predicted dose (SHAP {r['shap']:.3f})")
    return " & ".join(phrases), top  # also return top rows for any table display


# create explainers (cached)
X_background, explainer_tuples = create_shap_explainers(model, df, FEATURE_COLS, background_samples=200)

# ---------------------------
# Prediction function
# ---------------------------
def predict_fertilizer(N, P, K, pH):
    X = np.array([[N, P, K, pH]])
    pred = model.predict(X)[0]
    return {TARGET_COLS[0]: float(pred[0]), TARGET_COLS[1]: float(pred[1]), TARGET_COLS[2]: float(pred[2])}

# ---------------------------
# Header + Tabs
# ---------------------------
st.markdown("<h1 style='text-align:center; color:#e5e7eb;'>🌾 AI-Powered Fertilizer Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#9ca3af;'>Tuned Random Forest • Davangere Soil Data</p>", unsafe_allow_html=True)
st.markdown("---")

tab_reco, tab_insights = st.tabs(["🧮 Recommendation", "🧠 Model Insights (SHAP & EDA)"])

# ===========================
# TAB 1: Recommendation UI
# ===========================
with tab_reco:
    left, right = st.columns([1, 1])
    with left:
        st.markdown("<div class='card-dark'>", unsafe_allow_html=True)
        st.markdown("### 🧪 Soil Test Inputs")
        st.caption("Enter soil test values (units: kg/ha for N,P,K and standard pH).")

        # default values from your dataset statistics if present
        default_N = float(df[FEATURE_COLS[0]].median()) if FEATURE_COLS[0] in df else 80.0
        default_P = float(df[FEATURE_COLS[1]].median()) if FEATURE_COLS[1] in df else 18.0
        default_K = float(df[FEATURE_COLS[2]].median()) if FEATURE_COLS[2] in df else 160.0
        default_pH = float(df[FEATURE_COLS[3]].median()) if FEATURE_COLS[3] in df else 7.2

        N = st.number_input(f"Soil {FEATURE_COLS[0]}", min_value=0.0, max_value=1000.0, value=default_N, step=0.1)
        P = st.number_input(f"Soil {FEATURE_COLS[1]}", min_value=0.0, max_value=1000.0, value=default_P, step=0.1)
        K = st.number_input(f"Soil {FEATURE_COLS[2]}", min_value=0.0, max_value=1000.0, value=default_K, step=0.1)
        pH = st.number_input(f"Soil {FEATURE_COLS[3]}", min_value=0.0, max_value=14.0, value=default_pH, step=0.01)

        clicked = st.button("🔍 Get Fertilizer Recommendation", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card-dark'>", unsafe_allow_html=True)
        st.markdown("### 📊 Recommended Fertilizer Dosage (per Acre)")

        if clicked:
            results = predict_fertilizer(N, P, K, pH)
            st.session_state["last_input"] = {"N": N, "P": P, "K": K, "pH": pH}
            st.session_state["last_pred"] = results

            # compute SHAP for the single sample for each target
            X_sample = pd.DataFrame([[N, P, K, pH]], columns=FEATURE_COLS)
            per_target_shap = {}
            per_target_text = {}
            for idx, target in enumerate(TARGET_COLS):
                expl, _ = explainer_tuples[idx]
                shap_vals_single = expl.shap_values(X_sample)[0]  # 1D
                explanation_text, top_df = build_text_explanation(FEATURE_COLS, shap_vals_single, dict(zip(FEATURE_COLS, X_sample.iloc[0].to_list())), top_k=3)
                per_target_shap[target] = shap_vals_single
                per_target_text[target] = (explanation_text, top_df)

            st.session_state["last_shap"] = per_target_shap
            st.session_state["last_shap_text"] = per_target_text

        if "last_pred" in st.session_state:
            res = st.session_state["last_pred"]
            col1, col2, col3 = st.columns(3)
            # show cards
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='card-label'>🌿 {TARGET_COLS[0].upper()}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='card-value'>{res[TARGET_COLS[0]]:0.2f} kg</div>", unsafe_allow_html=True)
                st.markdown("<small>Supplies N for crop growth.</small>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='card-label'>🧪 {TARGET_COLS[1].upper()}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='card-value'>{res[TARGET_COLS[1]]:0.2f} kg</div>", unsafe_allow_html=True)
                st.markdown("<small>Supplies P for roots & early vigor.</small>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col3:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='card-label'>💧 {TARGET_COLS[2].upper()}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='card-value'>{res[TARGET_COLS[2]]:0.2f} kg</div>", unsafe_allow_html=True)
                st.markdown("<small>Supplies K for grain filling & stress tolerance.</small>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # personalized explanations area
            st.markdown("---")
            st.markdown("### 🔎 Why did the model suggest these doses? (Personalized explanation)")
            for t in TARGET_COLS:
                st.markdown(f"**{t.upper()} — Explanation**")
                explanation_text, top_df = st.session_state.get("last_shap_text", {}).get(t, ("No SHAP available", None))
                st.write("Why? ", explanation_text)
                # plot small bar for the single sample contributions
                shap_vals_single = st.session_state.get("last_shap", {}).get(t, None)
                if shap_vals_single is not None:
                    fig_bar = plot_shap_bar_single(shap_vals_single, FEATURE_COLS, title=f"SHAP contributions — {t}")
                    st.pyplot(fig_bar)
                # show the numeric top table
                if top_df is not None:
                    st.table(top_df.reset_index(drop=True))
        else:
            st.warning("Enter soil values and click **Get Fertilizer Recommendation**.")

        st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# TAB 2: Model Insights (SHAP + EDA)
# ===========================
with tab_insights:
    st.markdown("## Global SHAP (sampled background)")
    # show per-target global summary plots (beeswarm) but keep small and sampled
    col1, col2, col3 = st.columns(3)
    for idx, t in enumerate(TARGET_COLS):
        expl, sv = explainer_tuples[idx]
        # shap.summary_plot expects shap_values and X
        st.markdown(f"**Global: {t}**")
        try:
            fig = plot_shap_summary(sv, X_background, FEATURE_COLS, title=f"Global SHAP — {t}")
            st.pyplot(fig)
        except Exception as e:
            st.write("Could not render SHAP summary:", e)

    st.markdown("---")
    st.markdown("## Dataset Exploratory (simple plots)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Feature distributions (hist + KDE)")
        fig, ax = plt.subplots(2,2, figsize=(8,6))
        ax = ax.flatten()
        for i, f in enumerate(FEATURE_COLS):
            try:
                df[f].plot(kind="hist", bins=25, ax=ax[i], density=True, alpha=0.6)
                df[f].plot(kind="kde", ax=ax[i])
                ax[i].set_title(f)
            except Exception:
                ax[i].text(0.2, 0.5, f"Plot failed for {f}", transform=ax[i].transAxes)
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        st.write("Correlation (heatmap) - features vs targets")
        corr_cols = FEATURE_COLS + TARGET_COLS
        try:
            fig2, ax2 = plt.subplots(figsize=(7,5))
            sns = __import__("seaborn")
            sns.heatmap(df[corr_cols].corr(), cmap="Greens", annot=True, fmt=".2f", ax=ax2)
            ax2.set_title("Correlation: features vs targets")
            st.pyplot(fig2)
        except Exception as e:
            st.write("Heatmap failed:", e)

    st.markdown("---")
    st.markdown("## Save / Download")
    col1, col2 = st.columns([1,2])
    with col1:
        if st.button("Download sample predictions CSV"):
            # build sample predictions for entire df features (may be heavy)
            X_all = df[FEATURE_COLS].values
            try:
                preds_all = model.predict(X_all)
                out_df = df[FEATURE_COLS].copy()
                out_df[TARGET_COLS[0]] = preds_all[:,0]
                out_df[TARGET_COLS[1]] = preds_all[:,1]
                out_df[TARGET_COLS[2]] = preds_all[:,2]
                st.download_button("Download CSV", out_df.to_csv(index=False), file_name="predictions_sample.csv", mime="text/csv")
            except Exception as e:
                st.error("Could not compute predictions for full dataset: " + str(e))
    with col2:
        st.info("Global SHAP explains which features (N,P,K,pH) most influence urea/ssp/mop predictions.")

# ---------------------------
# Small KB / Chat (retrieval)
# ---------------------------
DATA_URL = "final_davangere_kvk_1000_rows.csv"

@st.cache_resource
def build_kb_and_vectorizer(data_path=DATA_URL):
    faqs = [
        ("What is this app", "This application recommends Urea, SSP and MOP doses per acre using soil N, P, K and pH."),
        ("Units", "Soil nutrients are in kg/ha and fertilizer outputs are in kg/acre."),
        ("How to use", "Enter soil N, P, K and pH on the Recommendation tab, then click Get Fertilizer Recommendation."),
        ("Why SHAP", "SHAP is used to explain which soil features most influenced the recommendation."),
        ("pH range", "Typical pH for rice is 5.5 to 8.5. Optimal 6.0 to 7.5."),
        ("How are fertilizers calculated", "We convert recommended nutrients to fertilizers using: Urea = N/0.46, SSP = P2O5/0.16, MOP = K2O/0.60.")
    ]
    df_local = pd.read_csv(data_path)
    sentences = []
    sentences.append("Dataset summary: {} samples. Soil N range {:.1f}-{:.1f} kg/ha. Soil P range {:.1f}-{:.1f} kg/ha. Soil K range {:.1f}-{:.1f} kg/ha. pH range {:.2f}-{:.2f}."
                     .format(len(df_local),
                             df_local[FEATURE_COLS[0]].min(), df_local[FEATURE_COLS[0]].max(),
                             df_local[FEATURE_COLS[1]].min(), df_local[FEATURE_COLS[1]].max(),
                             df_local[FEATURE_COLS[2]].min(), df_local[FEATURE_COLS[2]].max(),
                             df_local[FEATURE_COLS[3]].min(), df_local[FEATURE_COLS[3]].max()))
    # add some sample rows
    for i, row in df_local.sample(min(20, len(df_local)), random_state=42).iterrows():
        sentences.append(f"Sample: N={row[FEATURE_COLS[0]]:.1f}, P={row[FEATURE_COLS[1]]:.1f}, K={row[FEATURE_COLS[2]]:.1f}, pH={row[FEATURE_COLS[3]]:.2f} -> {TARGET_COLS[0]}={row[TARGET_COLS[0]]:.1f}, {TARGET_COLS[1]}={row[TARGET_COLS[1]]:.1f}, {TARGET_COLS[2]}={row[TARGET_COLS[2]]:.1f}")
    kb_texts = [q + " - " + a for (q,a) in faqs] + sentences
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    X = vect.fit_transform(kb_texts)
    return kb_texts, vect, X

kb_texts, vect, X_kb = build_kb_and_vectorizer()

def retrieve_answer(question, top_k=3):
    q_vec = vect.transform([question])
    sims = cosine_similarity(q_vec, X_kb).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    answers = []
    for i in top_idx:
        answers.append((kb_texts[i], float(sims[i])))
    return answers

with st.expander("💬 Chatbot (FAQ & dataset retrieval)", expanded=False):
    st.markdown("Ask about the app, units, dataset, or simple fertiliser questions.")
    user_q = st.text_input("Ask a question (e.g. 'For N=70,P=18,K=160,pH=7 what's urea?')")
    if st.button("Ask") and user_q.strip() != "":
        import re
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", user_q)
        if len(nums) >= 4:
            try:
                Nq, Pq, Kq, pHq = map(float, nums[:4])
                pred = model.predict(np.array([[Nq, Pq, Kq, pHq]]))[0]
                st.success("Model prediction:")
                st.write(f"{TARGET_COLS[0]}: {pred[0]:.2f} kg/acre  •  {TARGET_COLS[1]}: {pred[1]:.2f} kg/acre  •  {TARGET_COLS[2]}: {pred[2]:.2f} kg/acre")
            except Exception as e:
                st.error("Could not parse numbers for prediction.")
        else:
            answers = retrieve_answer(user_q, top_k=3)
            st.markdown("**Top answers from knowledge base:**")
            for txt, score in answers:
                st.markdown(f"- (score {score:.2f})  {textwrap.fill(txt, 120)}")

# End of app.py
