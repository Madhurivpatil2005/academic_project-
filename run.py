# app.py (UPGRADED FRONTEND)
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import base64
from typing import Tuple

# ---------------------------
# Page config + CSS (modern)
# ---------------------------
st.set_page_config(page_title="Rice Fertilizer Optimizer", page_icon="🌾", layout="wide")
st.markdown(
    """
    <style>
    :root{
      --bg:#071026;
      --panel:#0b1220;
      --muted:#9aa4b2;
      --accent1:#16a34a;
      --accent2:#22c55e;
      --glass: rgba(255,255,255,0.03);
    }
    html, body, [class*="css"]  { background: linear-gradient(180deg,#041425 0%, #071026 40%); color:#e6eef8; }
    .stApp { font-family: "Segoe UI", Roboto, sans-serif; }
    .hero {
      padding: 22px;
      border-radius: 14px;
      background: linear-gradient(90deg, rgba(20,83,45,0.18), rgba(34,197,94,0.06));
      box-shadow: 0 8px 30px rgba(2,6,23,0.6);
      margin-bottom: 12px;
    }
    .muted { color: var(--muted); font-size:0.95rem; }
    .card {
      border-radius:14px;
      padding:12px;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border: 1px solid rgba(255,255,255,0.03);
    }
    .value-large { font-weight:700; font-size:1.9rem; }
    .small-muted { color:var(--muted); font-size:0.85rem; }
    .kpi {
      border-radius:12px; padding:12px; text-align:center;
      background: linear-gradient(90deg, rgba(2,6,23,0.6), rgba(4,10,33,0.8));
      box-shadow: 0 6px 20px rgba(2,6,23,0.6);
    }
    .pill { background: rgba(255,255,255,0.03); padding:6px 10px; border-radius:999px; color:var(--muted); font-size:0.85rem; }
    /* responsive tweaks for narrow screens */
    @media (max-width: 600px) {
      .hero { padding:12px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers: load model & data
# ---------------------------
@st.cache_resource
def load_model_cached(model_path="best_fertilizer_model.pkl"):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.session_state["model_load_error"] = str(e)
        return None

@st.cache_resource
def load_dataset(path="final_davangere_kvk_1000_rows.csv") -> Tuple[pd.DataFrame, list, list]:
    df = pd.read_csv(path)
    # detect feature/target column names
    if set(["soil_N_kg_per_ha", "soil_P_kg_per_ha", "soil_K_kg_per_ha", "soil_pH"]).issubset(df.columns):
        features = ["soil_N_kg_per_ha", "soil_P_kg_per_ha", "soil_K_kg_per_ha", "soil_pH"]
        targets = ["urea_kg_per_acre", "ssp_kg_per_acre", "mop_kg_per_acre"]
    else:
        features = ["N", "P", "K", "pH"]
        targets = ["urea", "ssp", "mop"]
    return df, features, targets

# load assets
model = load_model_cached()
df, FEATURE_COLS, TARGET_COLS = load_dataset()

# show warning if load failed
if model is None:
    st.error("Model not loaded. Please ensure 'best_fertilizer_model.pkl' is present and valid. See session_state for details.")
    if "model_load_error" in st.session_state:
        st.caption(st.session_state["model_load_error"])

# ---------------------------
# SHAP helpers (cached)
# ---------------------------
@st.cache_resource
def create_shap_explainers(_model, data_frame, feature_cols, background_samples=200):
    if _model is None:
        return None, []
    n = min(background_samples, len(data_frame))
    X_background = data_frame[feature_cols].sample(n, random_state=42)
    try:
        estimators = _model.estimators_
    except Exception:
        estimators = [_model] * len(feature_cols)
    explainers = []
    shap_values_all = []
    for est in estimators[: len(TARGET_COLS) ]:
        try:
            expl = shap.TreeExplainer(est)
            sv = expl.shap_values(X_background)
            explainers.append(expl)
            shap_values_all.append(sv)
        except Exception:
            explainers.append(None)
            shap_values_all.append(None)
    return X_background, list(zip(explainers, shap_values_all))

def build_text_explanation(feature_names, shap_vals_single, feature_values, top_k=3):
    shap_arr = np.array(shap_vals_single).flatten()
    df_tmp = pd.DataFrame({
        "feature": feature_names,
        "shap": shap_arr,
        "value": [feature_values.get(f, None) for f in feature_names]
    })
    top = df_tmp.reindex(df_tmp["shap"].abs().sort_values(ascending=False).index).head(top_k)
    phrases = []
    for _, r in top.iterrows():
        direction = "increases" if r["shap"] > 0 else "decreases"
        phrases.append(f"{r['feature']} {direction} the predicted dose (impact {r['shap']:.3f})")
    return " & ".join(phrases), top

def plot_shap_bar_single(shap_vals_single, feature_names, title="SHAP (sample)"):
    vals = np.array(shap_vals_single).flatten()
    df_plot = pd.DataFrame({"feature": feature_names, "shap": vals})
    df_plot = df_plot.reindex(df_plot["shap"].abs().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(5,2.6))
    ax.barh(df_plot["feature"], df_plot["shap"])
    ax.set_title(title)
    ax.set_xlabel("SHAP value")
    plt.tight_layout()
    return fig

# create explainers (if possible)
X_background, explainer_tuples = create_shap_explainers(model, df, FEATURE_COLS, background_samples=200)

# ---------------------------
# Prediction function (fixed to use DataFrame with column names)
# ---------------------------
def predict_fertilizer(N, P, K, pH):
    if model is None:
        raise ValueError("Model not loaded.")
    # Build DataFrame with feature names to match how the model was trained
    X_df = pd.DataFrame([[N, P, K, pH]], columns=FEATURE_COLS)
    pred_arr = model.predict(X_df)
    pred_row = np.asarray(pred_arr).reshape(-1)
    return {
        TARGET_COLS[0]: float(pred_row[0]),
        TARGET_COLS[1]: float(pred_row[1]),
        TARGET_COLS[2]: float(pred_row[2])
    }

# ---------------------------
# Header / Hero
# ---------------------------
with st.container():
    left, right = st.columns([3,1])
    with left:
        st.markdown("<div class='hero'>", unsafe_allow_html=True)
        st.markdown("<h1 style='margin:0;'>🌾 AI-Powered Fertilizer Optimizer — Rice</h1>", unsafe_allow_html=True)
        st.markdown("<div class='muted'>Personalized Urea / SSP / MOP dosing per acre using local Davangere soil data.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='kpi card'>", unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>Samples in dataset</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='value-large'>{len(df):,}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# Main layout: Inputs | Results
# ===========================
col_inputs, col_results = st.columns([1,1.4])

with col_inputs:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🧪 Soil Inputs & Presets")
    st.markdown("<div class='small-muted'>Units: soil N,P,K in kg/ha. Fertilizer outputs shown kg/acre.</div>", unsafe_allow_html=True)
    with st.form("input_form"):
        # quick presets based on dataset median
        med_N = float(df[FEATURE_COLS[0]].median()) if FEATURE_COLS[0] in df else 80.0
        med_P = float(df[FEATURE_COLS[1]].median()) if FEATURE_COLS[1] in df else 18.0
        med_K = float(df[FEATURE_COLS[2]].median()) if FEATURE_COLS[2] in df else 160.0
        med_pH = float(df[FEATURE_COLS[3]].median()) if FEATURE_COLS[3] in df else 7.2

        preset = st.selectbox("Quick preset", ["Dataset median", "Low nutrient (deficient)", "High K", "Custom"], index=0)
        if preset == "Dataset median":
            N_default, P_default, K_default, pH_default = med_N, med_P, med_K, med_pH
        elif preset == "Low nutrient (deficient)":
            N_default, P_default, K_default, pH_default = med_N*0.6, med_P*0.6, med_K*0.7, med_pH
        elif preset == "High K":
            N_default, P_default, K_default, pH_default = med_N, med_P, med_K*1.4, med_pH
        else:
            N_default, P_default, K_default, pH_default = med_N, med_P, med_K, med_pH

        N = st.number_input(f"Soil {FEATURE_COLS[0]}", value=round(N_default,2), min_value=0.0, max_value=2000.0, step=0.1)
        P = st.number_input(f"Soil {FEATURE_COLS[1]}", value=round(P_default,2), min_value=0.0, max_value=1000.0, step=0.1)
        K = st.number_input(f"Soil {FEATURE_COLS[2]}", value=round(K_default,2), min_value=0.0, max_value=2000.0, step=0.1)
        pH = st.number_input(f"Soil {FEATURE_COLS[3]}", value=round(pH_default,2), min_value=0.0, max_value=14.0, step=0.01)

        st.caption("Tip: use dataset presets to quickly test realistic inputs.")
        submit = st.form_submit_button("🔍 Get Fertilizer Recommendation")

    st.markdown("</div>", unsafe_allow_html=True)

    # small FAQ / KB quick access
    st.markdown("<div class='card' style='margin-top:12px;'>", unsafe_allow_html=True)
    st.markdown("### 💬 Quick FAQ & KB")
    st.markdown("<div class='small-muted'>Ask about units, dataset, or paste an N,P,K,pH query (e.g. 'N=70,P=18,K=160,pH=7').</div>", unsafe_allow_html=True)
    kb_query = st.text_input("Ask the KB / model", key="kb_query_input")
    if st.button("Ask KB"):
        # build or reuse TF-IDF + retrieval (lazy build)
        @st.cache_resource
        def build_kb_and_vectorizer(data_path="final_davangere_kvk_1000_rows.csv"):
            faqs = [
                ("What is this app", "This application recommends Urea, SSP and MOP doses per acre using soil N, P, K and pH."),
                ("Units", "Soil nutrients are in kg/ha and fertilizer outputs are in kg/acre."),
                ("How to use", "Enter soil N, P, K and pH on the Recommendation tab, then click Get Fertilizer Recommendation."),
                ("Why SHAP", "SHAP is used to explain which features most influenced the recommendation."),
                ("pH range", "Typical pH for rice is 5.5 to 8.5. Optimal 6.0 to 7.5."),
                ("How are fertilizers calculated", "We convert recommended nutrients to fertilizers using Urea = N/0.46, SSP = P2O5/0.16, MOP = K2O/0.60.")
            ]
            df_local = pd.read_csv(data_path)
            sentences = []
            sentences.append("Dataset summary: {} samples. Soil N range {:.1f}-{:.1f} kg/ha. Soil P range {:.1f}-{:.1f} kg/ha. Soil K range {:.1f}-{:.1f} kg/ha. pH range {:.2f}-{:.2f}."
                             .format(len(df_local),
                                     df_local[FEATURE_COLS[0]].min(), df_local[FEATURE_COLS[0]].max(),
                                     df_local[FEATURE_COLS[1]].min(), df_local[FEATURE_COLS[1]].max(),
                                     df_local[FEATURE_COLS[2]].min(), df_local[FEATURE_COLS[2]].max(),
                                     df_local[FEATURE_COLS[3]].min(), df_local[FEATURE_COLS[3]].max()))
            for i, row in df_local.sample(min(20, len(df_local)), random_state=42).iterrows():
                sentences.append(f"Sample: N={row[FEATURE_COLS[0]]:.1f}, P={row[FEATURE_COLS[1]]:.1f}, K={row[FEATURE_COLS[2]]:.1f}, pH={row[FEATURE_COLS[3]]:.2f} -> {TARGET_COLS[0]}={row[TARGET_COLS[0]]:.1f}, {TARGET_COLS[1]}={row[TARGET_COLS[1]]:.1f}, {TARGET_COLS[2]}={row[TARGET_COLS[2]]:.1f}")
            kb_texts = [q + " - " + a for (q,a) in faqs] + sentences
            vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
            X = vect.fit_transform(kb_texts)
            return kb_texts, vect, X

        kb_texts, vect, X_kb = build_kb_and_vectorizer()
        if kb_query.strip() == "":
            st.info("Type a question or an N,P,K,pH string.")
        else:
            import re
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", kb_query)
            if len(nums) >= 4:
                Nq, Pq, Kq, pHq = map(float, nums[:4])
                try:
                    # Use DataFrame with feature names to avoid sklearn feature-name warnings
                    X_kb_df = pd.DataFrame([[Nq, Pq, Kq, pHq]], columns=FEATURE_COLS)
                    pred_arr = model.predict(X_kb_df)
                    pred = np.asarray(pred_arr).reshape(-1)
                    st.success("Model prediction from KB input:")
                    st.write(f"{TARGET_COLS[0]}: {pred[0]:.2f} kg/acre  •  {TARGET_COLS[1]}: {pred[1]:.2f} kg/acre  •  {TARGET_COLS[2]}: {pred[2]:.2f} kg/acre")
                except Exception as e:
                    st.error("Could not predict from KB input: " + str(e))
            else:
                q_vec = vect.transform([kb_query])
                sims = cosine_similarity(q_vec, X_kb).flatten()
                top_idx = sims.argsort()[::-1][:3]
                st.markdown("**Top KB matches:**")
                for i in top_idx:
                    st.markdown(f"- (score {sims[i]:.2f})  {textwrap.fill(kb_texts[i], 120)}")

    st.markdown("</div>", unsafe_allow_html=True)

with col_results:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Recommendation & Explanation")
    if 'last_input' not in st.session_state:
        st.session_state['last_input'] = None
    if submit:
        try:
            results = predict_fertilizer(N, P, K, pH)
            st.session_state["last_input"] = {"N": N, "P": P, "K": K, "pH": pH}
            st.session_state["last_pred"] = results

            # SHAP for the single sample
            X_sample = pd.DataFrame([[N, P, K, pH]], columns=FEATURE_COLS)
            per_target_shap = {}
            per_target_text = {}
            for idx, target in enumerate(TARGET_COLS):
                expl, _ = (explainer_tuples[idx] if explainer_tuples else (None, None))
                shap_vals_single = None
                explanation_text = "No SHAP available"
                top_df = None
                if expl is not None:
                    try:
                        shap_vals = expl.shap_values(X_sample)
                        # shap_values may return array or list; try to flatten
                        if isinstance(shap_vals, list):
                            shap_vals_single = shap_vals[0]
                        else:
                            shap_vals_single = shap_vals[0]
                        explanation_text, top_df = build_text_explanation(FEATURE_COLS, shap_vals_single, dict(zip(FEATURE_COLS, X_sample.iloc[0].to_list())), top_k=3)
                    except Exception:
                        shap_vals_single = None
                per_target_shap[target] = shap_vals_single
                per_target_text[target] = (explanation_text, top_df)

            st.session_state["last_shap"] = per_target_shap
            st.session_state["last_shap_text"] = per_target_text
        except Exception as e:
            st.error("Prediction failed: " + str(e))

    # show last prediction if available
    if st.session_state.get("last_pred", None) is not None:
        res = st.session_state["last_pred"]
        # show three metric cards horizontally
        k1, k2, k3 = st.columns([1,1,1])
        with k1:
            st.metric(label=f"🌿 {TARGET_COLS[0].upper()}", value=f"{res[TARGET_COLS[0]]:.2f} kg/acre", delta=None)
            st.caption("Nitrogen supply (Urea equivalent)")
        with k2:
            st.metric(label=f"🧪 {TARGET_COLS[1].upper()}", value=f"{res[TARGET_COLS[1]]:.2f} kg/acre", delta=None)
            st.caption("Phosphorus supply (SSP equivalent)")
        with k3:
            st.metric(label=f"💧 {TARGET_COLS[2].upper()}", value=f"{res[TARGET_COLS[2]]:.2f} kg/acre", delta=None)
            st.caption("Potassium supply (MOP equivalent)")

        st.markdown("---")
        st.markdown("#### 🔎 Why these doses? (SHAP-based concise explanation)")
        for t in TARGET_COLS:
            st.markdown(f"**{t.upper()}**")
            explanation_text, top_df = st.session_state.get("last_shap_text", {}).get(t, ("No SHAP available", None))
            st.write(explanation_text)
            shap_vals_single = st.session_state.get("last_shap", {}).get(t)
            if shap_vals_single is not None:
                fig_bar = plot_shap_bar_single(shap_vals_single, FEATURE_COLS, title=f"SHAP — {t}")
                st.pyplot(fig_bar)
            if top_df is not None:
                st.table(top_df.reset_index(drop=True))
        # provide download of this single-row result
        out_buf = io.StringIO()
        pd.DataFrame([{"N": st.session_state["last_input"]["N"],
                       "P": st.session_state["last_input"]["P"],
                       "K": st.session_state["last_input"]["K"],
                       "pH": st.session_state["last_input"]["pH"],
                       TARGET_COLS[0]: res[TARGET_COLS[0]],
                       TARGET_COLS[1]: res[TARGET_COLS[1]],
                       TARGET_COLS[2]: res[TARGET_COLS[2]]}]).to_csv(out_buf, index=False)
        st.download_button("⬇️ Download recommendation (CSV)", out_buf.getvalue(), file_name="fertilizer_recommendation.csv", mime="text/csv")
    else:
        st.info("Enter soil inputs on the left and click **Get Fertilizer Recommendation**.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# Insights tab (compact)
# ===========================
with st.expander("Model Insights & Dataset (global)"):
    st.markdown("**Global SHAP (sampled background) and simple EDA**")
    if explainer_tuples:
        for idx, t in enumerate(TARGET_COLS):
            expl, sv = explainer_tuples[idx]
            st.markdown(f"**Global SHAP — {t}**")
            try:
                # try to show beeswarm as small figure
                plt.figure(figsize=(7,2.2))
                shap.summary_plot(sv, X_background[FEATURE_COLS], show=False)
                plt.tight_layout()
                st.pyplot(plt.gcf())
                plt.clf()
            except Exception as e:
                st.write("Could not render global SHAP:", e)
    else:
        st.write("SHAP explainers not available (model may be missing or non-tree).")

    st.markdown("### Dataset quick EDA")
    fig, ax = plt.subplots(2,2, figsize=(8,5))
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

st.caption("Built with your model & data. Customize colors and layout in the CSS block at the top.")

# End of upgraded app.py
