"""Streamlit UI for the L-moments based uncertainty quantification toolbox.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

from lmoments import identify_dist, parameter_estimation, pdf_l, cdf_l, js_div

st.set_page_config(page_title="L-moments UQ", layout="wide")

_MLE_DIST = {
    "uniform": stats.uniform,
    "normal": stats.norm,
    "exponential": stats.expon,
    "gumbel": stats.gumbel_r,
    "logistic": stats.logistic,
    "generalized extreme value": stats.genextreme,
    "generalized pareto": stats.genpareto,
    "lognormal": stats.lognorm,
    "gamma": stats.gamma,
}

st.title("L-moments based Uncertainty Quantification")
st.caption(
    "Distribution-independent UQ for scarce samples, including extremes. "
    "Python port of [DeepanJayaraman/UQ](https://github.com/DeepanJayaraman/UQ)."
)

# ---------------------------------------------------------------- Sidebar --
st.sidebar.header("1. Sample data")
input_mode = st.sidebar.radio(
    "Data source",
    ["Paste values", "Upload CSV", "Generate example (scarce + extreme)"],
)

x = None

if input_mode == "Paste values":
    raw = st.sidebar.text_area(
        "One number per line or comma-separated",
        value="1.2, 0.9, 1.5, 1.1, 0.8, 2.0, 1.3, 0.95, 1.05, 1.4, 1.0, 9.5",
        height=160,
    )
    try:
        tokens = raw.replace("\n", ",").split(",")
        x = np.array([float(t) for t in tokens if t.strip() != ""])
    except ValueError:
        st.sidebar.error("Could not parse values — check for stray text.")

elif input_mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("CSV file, single column of numbers", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded, header=None)
        x = df.iloc[:, 0].astype(float).to_numpy()

else:
    st.sidebar.write("Synthetic scarce sample with one injected extreme value:")
    n = st.sidebar.slider("Sample size (n)", 5, 50, 12)
    true_sigma = st.sidebar.slider("True lognormal sigma", 0.1, 1.5, 0.5)
    extreme_mult = st.sidebar.slider("Extreme value = extreme_mult x max(sample)", 2.0, 15.0, 8.0)
    seed = st.sidebar.number_input("Random seed", value=7, step=1)
    rng = np.random.default_rng(int(seed))
    x = stats.lognorm.rvs(s=true_sigma, scale=1.0, size=n, random_state=rng)
    x = np.append(x, extreme_mult * x.max())

if x is not None:
    x = x[~np.isnan(x)]

st.sidebar.header("2. Options")
show_mle_comparison = st.sidebar.checkbox("Compare against conventional-moment (MLE) fit", value=True)
top_k = st.sidebar.slider("Distribution families to show in ranking", 1, 9, 5)

# ---------------------------------------------------------------- Main ----
if x is None or len(x) < 4:
    st.info("Enter at least 4 sample values to begin (see the sidebar).")
    st.stop()

col_data, col_stats = st.columns([2, 1])
with col_data:
    st.subheader("Sample")
    st.write(f"n = {len(x)}")
    st.dataframe(pd.DataFrame({"value": x}).T, hide_index=True)

result = identify_dist(x)
L1, L2, T3, T4 = result["L_sample"]
fitted_dist = result["best"]

with col_stats:
    st.subheader("Sample L-moments")
    st.table(pd.DataFrame(
        {"value": [L1, L2, T3, T4]},
        index=["L1 (mean)", "L2", "T3 (L-skewness)", "T4 (L-kurtosis)"],
    ))

st.subheader("Distribution identification")
st.write(
    f"Closest match on the L-moment ratio diagram: **{fitted_dist}**"
)
ranking_df = pd.DataFrame(result["ranking"][:top_k], columns=["distribution", "distance"])
st.dataframe(ranking_df, hide_index=True)

chosen_dist = st.selectbox(
    "Distribution to fit (defaults to the closest match above)",
    options=[name for name, _ in result["ranking"]],
    index=0,
)

param_L = parameter_estimation(x, chosen_dist, L1, L2, T3, T4)
st.write(f"L-moment parameter estimate for **{chosen_dist}**: `{np.round(param_L, 4).tolist()}`")

lo = max(float(x.min()) - 1, 1e-6)
hi = float(x.max()) * 1.05
xgrid = np.linspace(lo, hi, 400)
pdf_L = pdf_l(xgrid, chosen_dist, param_L)
cdf_L = cdf_l(xgrid, chosen_dist, param_L)

mle_params = None
if show_mle_comparison and chosen_dist in _MLE_DIST:
    mle_family = _MLE_DIST[chosen_dist]
    try:
        mle_params = mle_family.fit(x)
        pdf_conv = mle_family.pdf(xgrid, *mle_params)
        cdf_conv = mle_family.cdf(xgrid, *mle_params)
        st.write(f"Conventional-moment (MLE) parameter estimate: `{np.round(mle_params, 4).tolist()}`")
    except Exception as exc:  # scipy's MLE fit can fail to converge on tiny/degenerate samples
        st.warning(f"MLE fit failed for comparison: {exc}")
        show_mle_comparison = False

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].hist(x, density=True, alpha=0.3, label="Sample histogram")
axes[0].plot(xgrid, pdf_L, lw=2, label="L-moment fit")
if show_mle_comparison and mle_params is not None:
    axes[0].plot(xgrid, pdf_conv, "--", lw=2, label="Conventional-moment (MLE) fit")
axes[0].set_xlabel("X"); axes[0].set_ylabel("PDF")
axes[0].set_title(f"PDF fit: {chosen_dist}")
axes[0].legend()

x_sorted = np.sort(x)
f_emp = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
axes[1].step(x_sorted, f_emp, where="post", color="k", label="Empirical CDF")
axes[1].plot(xgrid, cdf_L, lw=2, label="L-moment fit")
if show_mle_comparison and mle_params is not None:
    axes[1].plot(xgrid, cdf_conv, "--", lw=2, label="Conventional-moment (MLE) fit")
axes[1].set_xlabel("X"); axes[1].set_ylabel("CDF")
axes[1].set_title(f"CDF fit: {chosen_dist}")
axes[1].legend()
fig.tight_layout()
st.pyplot(fig)

if show_mle_comparison and mle_params is not None:
    st.subheader("Fit-quality comparison (Jensen-Shannon divergence)")
    edges = np.linspace(lo, hi, min(30, max(5, len(x))))
    p_emp, _ = np.histogram(x, bins=edges)
    p_L = np.diff(cdf_l(edges, chosen_dist, param_L))
    p_conv = np.diff(mle_family.cdf(edges, *mle_params))
    jsd_L = js_div(p_emp + 1e-12, p_L + 1e-12)
    jsd_conv = js_div(p_emp + 1e-12, p_conv + 1e-12)
    c1, c2 = st.columns(2)
    c1.metric("Empirical vs L-moment fit", f"{jsd_L:.4f}")
    c2.metric("Empirical vs MLE fit", f"{jsd_conv:.4f}", delta=f"{jsd_conv - jsd_L:+.4f}",
              delta_color="inverse")
    st.caption(
        "Lower divergence = closer match to the empirical sample. A positive "
        "delta means the conventional-moment fit is more distorted by any "
        "extreme values in the sample than the L-moment fit."
    )

st.divider()
st.caption(
    "Known limitations (inherited from the underlying method/toolbox): Weibull is "
    "not offered for automatic identification; Gumbel vs. GEV(k≈0) and Uniform vs. "
    "GP(k≈1) can be ambiguous since they are geometrically coincident on the "
    "L-moment ratio diagram."
)
