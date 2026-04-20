import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Beam Fire Resistance Predictor",
    page_icon="🔥",
    layout="centered",
)

# ── Beam property lookup table (for auto-fill reference) ───────────────────────
# Key: (depth, width, weight) → (flange_width, flange_thickness, total_depth, web_thickness)
BEAM_LOOKUP = {
    (127, 76,  13):  (76.0,  7.6,  127.0, 4.0),
    (203, 102, 23):  (101.8, 9.3,  203.2, 5.4),
    (203, 133, 30):  (133.9, 9.6,  206.8, 6.4),
    (254, 102, 25):  (133.0, 8.5,  266.0, 5.8),
    (254, 146, 31):  (101.9, 8.4,  257.2, 6.0),
    (305, 102, 28):  (101.8, 8.8,  308.7, 6.0),
    (305, 127, 42):  (124.3, 12.1, 307.2, 8.0),
    (305, 165, 46):  (165.7, 11.8, 306.6, 6.7),
    (356, 127, 39):  (126.0, 10.7, 352.8, 6.6),
    (356, 171, 45):  (171.1, 9.7,  351.4, 7.0),
    (356, 171, 67):  (173.2, 15.7, 363.4, 9.1),
    (406, 140, 53):  (143.3, 12.9, 406.6, 7.9),
    (406, 178, 67):  (178.8, 14.3, 409.4, 8.8),
    (457, 152, 52):  (152.4, 10.9, 449.8, 7.6),
    (457, 152, 74):  (154.4, 17.0, 462.0, 9.6),
}

BEAM_TYPE_MAP = {"Simply supported": 0, "Cantilever": 1, "Propped": 2}

# ── Estimation formulas for unknown beams ──────────────────────────────────────
def estimate_properties(depth_mm, width_mm, weight_kgm):
    """
    Estimate flange_width, flange_thickness, total_depth, web_thickness
    from the three primary dimensions using linear regression on known beams.
    Works for any beam dimensions, not just catalogue values.
    """
    known = np.array(list(BEAM_LOOKUP.keys()), dtype=float)     # (N, 3)
    props = np.array(list(BEAM_LOOKUP.values()), dtype=float)   # (N, 4)

    query = np.array([depth_mm, width_mm, weight_kgm], dtype=float)

    # Weighted average of k nearest neighbours (inverse-distance weighting)
    dists = np.linalg.norm(known - query, axis=1)

    if np.min(dists) < 1e-6:
        # Exact match
        idx = np.argmin(dists)
        fw, ft, td, wt = props[idx]
    else:
        k = min(4, len(known))
        knn_idx = np.argsort(dists)[:k]
        weights = 1.0 / dists[knn_idx]
        weights /= weights.sum()
        fw, ft, td, wt = (props[knn_idx] * weights[:, None]).sum(axis=0)

    return float(fw), float(ft), float(td), float(wt)


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("randomForest_model.joblib")
        return model, None
    except FileNotFoundError:
        return None, "rf_model.joblib not found. Place it in the same folder as app.py."
    except Exception as e:
        return None, f"Error loading model: {e}"

model, model_error = load_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔥 Beam Fire Resistance Predictor")
st.caption("Predict the time (seconds) for a steel universal beam to reach 670 °C under fire exposure.")
st.divider()

if model_error:
    st.error(model_error)
    st.stop()

# ── Section 1: Primary Beam Dimensions ────────────────────────────────────────
st.subheader("1. Enter Beam Dimensions")
st.caption("Enter any values — not limited to standard catalogue sizes.")

col1, col2, col3 = st.columns(3)

with col1:
    depth = st.number_input(
        "Nominal Depth (mm)",
        min_value=50.0,
        max_value=1000.0,
        value=254.0,
        step=1.0,
        help="Nominal depth of the universal beam in mm"
    )

with col2:
    width = st.number_input(
        "Flange Width (mm)",
        min_value=50.0,
        max_value=500.0,
        value=146.0,
        step=1.0,
        help="Nominal flange width of the universal beam in mm"
    )

with col3:
    weight = st.number_input(
        "Weight (kg/m)",
        min_value=5.0,
        max_value=200.0,
        value=31.0,
        step=0.5,
        help="Mass per metre of the beam in kg/m"
    )

# ── Auto-estimate secondary properties ────────────────────────────────────────
fw, ft, td, wt = estimate_properties(depth, width, weight)

# Check if it's a known standard beam
lookup_key = (int(round(depth)), int(round(width)), int(round(weight)))
is_standard = lookup_key in BEAM_LOOKUP

if is_standard:
    st.success(f"✅ Matches standard catalogue beam **{int(depth)}x{int(width)}x{int(weight)}** — exact properties used.")
else:
    st.info("📐 Custom dimensions detected — secondary properties estimated by interpolation from nearest standard beams.")

# Show estimated / looked-up secondary properties
with st.expander("View estimated secondary properties (flange thickness, total depth, web thickness)"):
    prop_df = pd.DataFrame({
        "Property": ["Flange width (mm)", "Flange thickness (mm)", "Total depth (mm)", "Web thickness (mm)"],
        "Value": [f"{fw:.2f}", f"{ft:.2f}", f"{td:.2f}", f"{wt:.2f}"],
        "Source": ["From input" if is_standard else "Estimated"] * 4
    })
    st.dataframe(prop_df, use_container_width=True, hide_index=True)

st.divider()

# ── Section 2: Beam Type ───────────────────────────────────────────────────────
st.subheader("2. Beam Support Condition")

beam_type_label = st.radio(
    "Select beam type",
    options=list(BEAM_TYPE_MAP.keys()),
    horizontal=True,
    help="Structural support condition of the beam"
)
beam_type_encoded = BEAM_TYPE_MAP[beam_type_label]

st.divider()

# ── Section 3: Predict ─────────────────────────────────────────────────────────
st.subheader("3. Predict")

if st.button("🔥 Predict Time to Reach 670 °C", use_container_width=True, type="primary"):

    input_data = pd.DataFrame([{
        "flange_width":     fw,
        "flange_thickness": ft,
        "total_depth":      td,
        "web_thickness":    wt,
        "Beam_type":        beam_type_encoded,
        "Weight":           weight,
    }])

    prediction = model.predict(input_data)[0]

    # Result display
    st.success(f"### ⏱ {prediction:.1f} seconds")
    st.markdown(
        f"Estimated time for a **{depth:.0f}×{width:.0f}×{weight:.1f}** beam "
        f"({beam_type_label}) to reach **670 °C**"
    )

    # Fire resistance rating
    if prediction >= 290:
        st.markdown("🟢 **High fire resistance**")
    elif prediction >= 280:
        st.markdown("🟡 **Moderate fire resistance**")
    else:
        st.markdown("🔴 **Lower fire resistance**")

    # Full input summary
    with st.expander("View all input values sent to model"):
        display_df = pd.DataFrame({
            "Feature": [
                "Flange width (mm)",
                "Flange thickness (mm)",
                "Total depth (mm)",
                "Web thickness (mm)",
                "Beam type (encoded)",
                "Weight (kg/m)"
            ],
            "Value": [
                f"{fw:.2f}",
                f"{ft:.2f}",
                f"{td:.2f}",
                f"{wt:.2f}",
                f"{beam_type_encoded} ({beam_type_label})",
                f"{weight:.1f}"
            ]
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    if not is_standard:
        st.warning(
            "⚠️ Secondary properties (flange thickness, total depth, web thickness) were "
            "estimated by interpolation since your dimensions don't match a standard catalogue beam. "
            "Predictions for non-standard beams may be less accurate."
        )
    else:
        st.warning(
            "⚠️ This prediction is based on an ML model trained on FEA simulation data. "
            "Do not use as the sole basis for structural fire safety decisions."
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: RandomForestRegressor · CV R² = 0.950 · Trained on beam FEA simulation data")