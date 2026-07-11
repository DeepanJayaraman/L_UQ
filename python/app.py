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

from lmoments_uq import identify_dist, parameter_estimation, pdf_l, cdf_l, js_div

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
    "Python port of [DeepanJayaraman/L-UQ](https://github.com/DeepanJayaraman/L-UQ)."
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
    extreme_kind = st.sidebar.radio(
        "Extreme type",
        ["Population extreme (max of 1e5 parent draws)",
         "Gross outlier (multiplier x sample max)"],
        help="A population extreme is a genuine rare event from the same "
             "distribution (the scheme of Jayaraman & Ramu 2021); a gross "
             "outlier models a defect or measurement error.",
    )
    if extreme_kind.startswith("Gross"):
        extreme_mult = st.sidebar.slider(
            "Extreme value = multiplier x max(sample)", 2.0, 15.0, 8.0)
    seed = st.sidebar.number_input("Random seed", value=7, step=1)
    rng = np.random.default_rng(int(seed))
    x = stats.lognorm.rvs(s=true_sigma, scale=1.0, size=n, random_state=rng)
    if extreme_kind.startswith("Gross"):
        x = np.append(x, extreme_mult * x.max())
    else:
        x = np.append(x, stats.lognorm.rvs(
            s=true_sigma, scale=1.0, size=100_000, random_state=rng).max())

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
    with st.expander("What are L-moments?"):
        st.markdown(
            """
L-moments are summary statistics built from **linear combinations of the
sorted sample** (order statistics), instead of squaring/cubing deviations
like conventional moments do:

- **L1** — the mean, identical to the ordinary mean.
- **L2** — a measure of spread, playing the role of the standard deviation.
- **T3 = L3/L2 (L-skewness)** — asymmetry, playing the role of skewness.
- **T4 = L4/L2 (L-kurtosis)** — tail weight, playing the role of kurtosis.

Because no observation is ever squared or cubed, a single extreme value
influences L-moments far less than it influences conventional moments —
which is exactly why they are preferred for scarce samples that contain
extremes. (Hosking, 1990)
            """
        )

st.subheader("Distribution identification")
st.write(
    f"Closest match on the L-moment ratio diagram: **{fitted_dist}**"
)
ranking_df = pd.DataFrame(result["ranking"][:top_k], columns=["distribution", "distance"])
st.dataframe(ranking_df, hide_index=True)
with st.expander("How does identification work? What does 'distance' mean?"):
    st.markdown(
        """
Every distribution family occupies a characteristic place on the
**L-moment ratio diagram**, a plot of L-kurtosis (T4) against L-skewness
(T3):

- **Two-parameter families** (uniform, normal, exponential, gumbel,
  logistic) always have the same shape, so each is a **single point** —
  e.g. the normal sits at (0, 0.123).
- **Three-parameter families** (GEV, generalized Pareto, lognormal,
  gamma) change shape with their third parameter, so each traces a
  **curve** of possible (T3, T4) pairs.

Your sample's (T3, T4) is one point on this diagram. The **distance**
column is the Euclidean distance from that point to each family's point
or curve — *smaller is better*. The closest family is proposed as the
best fit, but when the top distances are nearly tied the data supports
several families about equally; the dropdown below lets you fit any of
them.

Two known ambiguities: Gumbel is the shape=0 special case of the GEV
curve, and Uniform is a boundary case of the generalized Pareto curve, so
those pairs sit on top of each other on the diagram and either may rank
first for such data.
        """
    )

chosen_dist = st.selectbox(
    "Distribution to fit (defaults to the closest match above)",
    options=[name for name, _ in result["ranking"]],
    index=0,
)

# Human-readable name for each entry of the parameter array, matching the
# layouts produced by parameter_estimation() / consumed by pdf_l()/cdf_l().
_PARAM_NAMES = {
    "uniform": ["lower bound a", "upper bound b"],
    "normal": ["mean μ", "standard deviation σ"],
    "exponential": ["scale (mean) α"],
    "gumbel": ["scale α", "location η"],
    "logistic": ["location η", "scale α"],
    "generalized extreme value": ["shape (−k, Hosking sign)", "scale α", "location η"],
    "generalized pareto": ["shape (−k, Hosking sign)", "scale α", "location η"],
    "lognormal": ["log-scale μ", "shape σ", "location shift η"],
    "gamma": ["shape α", "scale β", "location shift (sample min)"],
}

param_L = parameter_estimation(x, chosen_dist, L1, L2, T3, T4)
st.subheader("Parameter estimates")
st.markdown(f"**L-moment fit of the {chosen_dist} distribution:**")
st.table(pd.DataFrame(
    {"estimate": np.round(param_L, 4)},
    index=_PARAM_NAMES.get(chosen_dist, [f"parameter {i+1}" for i in range(len(param_L))]),
))
with st.expander("How are these parameters estimated?"):
    st.markdown(
        """
Each family has **closed-form expressions relating its parameters to its
L-moments** (Hosking & Wallis, 1997). The sample's L1, L2, T3, T4 are
plugged directly into those expressions — no iterative optimization and
no likelihood maximization, so the estimate exists even for very small
samples and is not dragged around by a single extreme observation the way
a variance-based or maximum-likelihood estimate is.

Simple examples: for the normal, mean = L1 and σ = √π·L2; for the
uniform, the bounds are L1 ∓ 3·L2. The three-parameter families use
Hosking's rational-polynomial approximations of the same idea.
        """
    )

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
with st.expander("How to read these plots"):
    st.markdown(
        """
- **Left (PDF):** the fitted probability density over the sample's
  histogram. A good fit tracks where the histogram bars concentrate. If
  your sample contains an extreme value, watch how each fit reacts: a
  conventional/MLE fit often flattens and widens to "reach" the extreme,
  while the L-moment fit stays anchored to the bulk of the data yet still
  assigns tail probability to the extreme.
- **Right (CDF):** the fitted cumulative distribution against the
  empirical CDF (black staircase, which jumps by 1/n at each observed
  value). This view weights every observation equally, so it is the
  fairer visual test with scarce data — look for the fitted curve
  hugging the staircase, especially in the tail region near the extreme.
        """
    )

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
    with st.expander("What is Jensen-Shannon divergence?"):
        st.markdown(
            """
The Jensen-Shannon divergence (JSD) measures how different two
probability distributions are. Here, the sample and each fitted
distribution are binned over the same grid, and the JSD is computed
between the empirical bin masses and the fitted ones.

Properties that make it a good fit-quality score:

- **Bounded 0 to 1** (with the log base 2 used here): 0 means the
  binned distributions are identical, 1 means they share no overlap at
  all. The two metrics above are therefore directly comparable.
- **Symmetric**: unlike raw Kullback-Leibler divergence, it doesn't
  matter which distribution is "reference" and which is "model".
- **Always finite**: empty bins (common with scarce samples spread over
  a wide range) are handled by the standard 0·log 0 = 0 convention, so
  a sample with one far-out extreme still gets a meaningful score.

This implementation was verified to machine precision against SciPy's
independent `jensenshannon` implementation (see the repository's test
suite for the full set of verified properties).
            """
        )

st.divider()
st.caption(
    "Known limitations (inherited from the underlying method/toolbox): Weibull is "
    "not offered for automatic identification; Gumbel vs. GEV(k≈0) and Uniform vs. "
    "GP(k≈1) can be ambiguous since they are geometrically coincident on the "
    "L-moment ratio diagram."
)
