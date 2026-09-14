"""
Lab-4: Applied Statistical Modeling & Interactive Web Dashboard
Dataset: Medical Insurance Costs
Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera, omni_normtest

st.set_page_config(page_title="Medical Insurance Statistical Dashboard", layout="wide")

ALPHA = 0.05


@st.cache_data
def load_data():
    df = pd.read_csv("data/insurance.csv")
    return df


@st.cache_resource
def fit_model(df: pd.DataFrame):
    model = smf.ols(
        "charges ~ age + bmi + children + C(smoker) + C(sex) + C(region) + bmi:C(smoker)",
        data=df,
    ).fit()
    return model


df = load_data()
model = fit_model(df)

st.title("🏥 Medical Insurance Costs - Statistical Dashboard")
st.caption(
    "Lab-4 | Statistical Modeling with Python — EDA, Hypothesis Testing, OLS Regression & Diagnostics"
)

tab1, tab2, tab3 = st.tabs(
    ["📊 Data Exploration", "🧪 Hypothesis Testing Lab", "🔮 Live Prediction & Diagnostics"]
)

# TAB 1 — DATA EXPLORATION

with tab1:
    st.header("Data Exploration")

    with st.sidebar:
        st.subheader("Filters")
        age_range = st.slider(
            "Age range", int(df.age.min()), int(df.age.max()),
            (int(df.age.min()), int(df.age.max()))
        )
        bmi_range = st.slider(
            "BMI range", float(df.bmi.min()), float(df.bmi.max()),
            (float(df.bmi.min()), float(df.bmi.max()))
        )
        regions_sel = st.multiselect(
            "Region(s)", options=sorted(df.region.unique()), default=sorted(df.region.unique())
        )
        smoker_sel = st.multiselect(
            "Smoker status", options=sorted(df.smoker.unique()), default=sorted(df.smoker.unique())
        )
        sex_sel = st.multiselect(
            "Sex", options=sorted(df.sex.unique()), default=sorted(df.sex.unique())
        )

    fdf = df[
        (df.age.between(*age_range))
        & (df.bmi.between(*bmi_range))
        & (df.region.isin(regions_sel))
        & (df.smoker.isin(smoker_sel))
        & (df.sex.isin(sex_sel))
    ]

    st.markdown(f"**Filtered rows:** {len(fdf)} / {len(df)}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg. Charges", f"${fdf.charges.mean():,.0f}")
    c2.metric("Median Charges", f"${fdf.charges.median():,.0f}")
    c3.metric("Avg. BMI", f"{fdf.bmi.mean():.1f}")
    c4.metric("Avg. Age", f"{fdf.age.mean():.1f}")

    st.subheader("Summary statistics")
    st.dataframe(fdf.describe(include="all").T, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            fdf, x="charges", color="smoker", nbins=40, marginal="box",
            title="Distribution of Charges (by Smoker Status)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            fdf, x="bmi", y="charges", color="smoker", trendline="ols",
            title="Charges vs BMI (colored by Smoker Status)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.box(fdf, x="region", y="charges", color="region", title="Charges by Region")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        corr = fdf[["age", "bmi", "children", "charges"]].corr()
        fig4 = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlation Matrix"
        )
        st.plotly_chart(fig4, use_container_width=True)

# TAB 2 — HYPOTHESIS TESTING LAB
with tab2:
    st.header("Hypothesis Testing Lab")

    test_choice = st.radio(
        "Choose a test",
        ["Two-Group Comparison (t-test / Mann-Whitney U)", "Chi-Square (categorical vs categorical)",
         "One-Way ANOVA (numeric across 3+ groups)"],
        horizontal=False,
    )

    st.divider()

    if test_choice.startswith("Two-Group"):
        st.subheader("Two-Group Comparison")
        cat_col = st.selectbox("Grouping (categorical, 2 levels)", ["smoker", "sex"])
        num_col = st.selectbox("Metric (numeric)", ["charges", "bmi", "age"])

        levels = df[cat_col].unique()
        if len(levels) != 2:
            st.warning("Selected column does not have exactly 2 levels.")
        else:
            g1 = df.loc[df[cat_col] == levels[0], num_col]
            g2 = df.loc[df[cat_col] == levels[1], num_col]

            sw1 = stats.shapiro(g1.sample(min(len(g1), 5000), random_state=1))
            sw2 = stats.shapiro(g2.sample(min(len(g2), 5000), random_state=1))
            lev_stat, lev_p = stats.levene(g1, g2)

            colA, colB = st.columns(2)
            colA.metric(f"Shapiro-Wilk p ({levels[0]})", f"{sw1.pvalue:.4e}")
            colB.metric(f"Shapiro-Wilk p ({levels[1]})", f"{sw2.pvalue:.4e}")
            st.metric("Levene's test p-value (equal variance)", f"{lev_p:.4e}")

            normal = (sw1.pvalue > ALPHA) and (sw2.pvalue > ALPHA)

            if normal:
                equal_var = lev_p > ALPHA
                t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)
                test_name = "Two-Sample t-test"
                stat_val = t_stat
            else:
                u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                test_name = "Mann-Whitney U test"
                stat_val = u_stat

            st.markdown(f"**Test used:** {test_name} *(auto-selected based on normality check)*")
            st.metric("Test statistic", f"{stat_val:.4f}")
            st.metric("p-value", f"{p_val:.4e}")

            if p_val < ALPHA:
                st.success(f"**Reject H0** (α={ALPHA}) — significant difference in {num_col} between {cat_col} groups.")
            else:
                st.info(f"**Fail to Reject H0** (α={ALPHA}) — no significant difference detected.")

            fig = px.box(df, x=cat_col, y=num_col, color=cat_col, points="all",
                         title=f"{num_col} by {cat_col}")
            st.plotly_chart(fig, use_container_width=True)

    elif test_choice.startswith("Chi-Square"):
        st.subheader("Chi-Square Test of Independence")
        cat1 = st.selectbox("Categorical variable 1", ["smoker", "sex", "region"], index=0)
        cat2 = st.selectbox("Categorical variable 2", ["region", "sex", "smoker"], index=0)

        if cat1 == cat2:
            st.warning("Choose two different variables.")
        else:
            contingency = pd.crosstab(df[cat1], df[cat2])
            chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

            st.write("Contingency table:")
            st.dataframe(contingency, use_container_width=True)

            st.metric("Chi-square statistic", f"{chi2:.4f}")
            st.metric("Degrees of freedom", dof)
            st.metric("p-value", f"{p_val:.4e}")

            if p_val < ALPHA:
                st.success(f"**Reject H0** (α={ALPHA}) — {cat1} and {cat2} appear to be linked.")
            else:
                st.info(f"**Fail to Reject H0** (α={ALPHA}) — no significant association detected.")

            fig = px.bar(contingency, barmode="group", title=f"{cat1} vs {cat2} counts")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.subheader("One-Way ANOVA")
        cat_col = st.selectbox("Grouping variable (3+ groups)", ["region"])
        num_col = st.selectbox("Numeric metric", ["charges", "bmi", "age"], key="anova_num")

        groups = [df.loc[df[cat_col] == lvl, num_col] for lvl in df[cat_col].unique()]
        f_stat, p_val = stats.f_oneway(*groups)

        st.metric("F-statistic", f"{f_stat:.4f}")
        st.metric("p-value", f"{p_val:.4e}")

        if p_val < ALPHA:
            st.success(f"**Reject H0** (α={ALPHA}) — mean {num_col} differs across {cat_col} groups.")
        else:
            st.info(f"**Fail to Reject H0** (α={ALPHA}) — no significant difference across groups.")

        means = df.groupby(cat_col)[num_col].mean().reset_index()
        fig = px.bar(means, x=cat_col, y=num_col, title=f"Mean {num_col} by {cat_col}")
        st.plotly_chart(fig, use_container_width=True)

# TAB 3 — LIVE PREDICTION & DIAGNOSTICS
with tab3:
    st.header("Live Prediction & Model Diagnostics")

    st.subheader("Predict charges for a new individual")
    c1, c2, c3 = st.columns(3)
    with c1:
        in_age = st.slider("Age", 18, 64, 35)
        in_bmi = st.slider("BMI", 15.0, 55.0, 27.0, step=0.1)
    with c2:
        in_children = st.number_input("Children", 0, 5, 0)
        in_sex = st.selectbox("Sex", sorted(df.sex.unique()))
    with c3:
        in_smoker = st.selectbox("Smoker", sorted(df.smoker.unique()))
        in_region = st.selectbox("Region", sorted(df.region.unique()))

    new_point = pd.DataFrame([{
        "age": in_age, "bmi": in_bmi, "children": in_children,
        "sex": in_sex, "smoker": in_smoker, "region": in_region,
    }])

    pred = model.get_prediction(new_point)
    pred_summary = pred.summary_frame(alpha=ALPHA)

    st.markdown("#### Prediction Result")
    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted charges", f"${pred_summary['mean'][0]:,.2f}")
    p2.metric(f"{int((1-ALPHA)*100)}% CI (mean)",
              f"${pred_summary['mean_ci_lower'][0]:,.0f} – ${pred_summary['mean_ci_upper'][0]:,.0f}")
    p3.metric(f"{int((1-ALPHA)*100)}% Prediction Interval",
              f"${pred_summary['obs_ci_lower'][0]:,.0f} – ${pred_summary['obs_ci_upper'][0]:,.0f}")

    st.divider()
    st.subheader("Residual Diagnostics (Gauss-Markov checks)")

    fitted = model.fittedvalues
    resid = model.resid

    d1, d2 = st.columns(2)
    with d1:
        fig_rf = px.scatter(x=fitted, y=resid, labels={"x": "Fitted values", "y": "Residuals"},
                             title="Residuals vs Fitted (Linearity & Homoscedasticity)")
        fig_rf.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_rf, use_container_width=True)

    with d2:
        qq = stats.probplot(resid, dist="norm")
        theo_q = np.array(qq[0][0])
        samp_q = np.array(qq[0][1])
        slope, intercept = qq[1][0], qq[1][1]
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=theo_q, y=samp_q, mode="markers", name="Residuals"))
        fig_qq.add_trace(go.Scatter(x=theo_q, y=slope * theo_q + intercept, mode="lines",
                                     name="Reference line", line=dict(color="red", dash="dash")))
        fig_qq.update_layout(title="Q-Q Plot of Residuals (Normality)",
                              xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)

    jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(resid)
    omni_stat, omni_p = omni_normtest(resid)

    e1, e2 = st.columns(2)
    e1.metric("Jarque-Bera p-value", f"{jb_p:.4e}")
    e2.metric("Omnibus normality p-value", f"{omni_p:.4e}")
    st.caption("p < 0.05 in either test indicates residuals depart from normality.")

    st.subheader("Multicollinearity — Variance Inflation Factor (VIF)")
    X = df[["age", "bmi", "children"]].copy()
    X["smoker_num"] = (df["smoker"] == "yes").astype(int)
    X = sm.add_constant(X)
    vif_df = pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    st.dataframe(vif_df, use_container_width=True)
    st.caption("VIF > 5 (commonly > 10) suggests problematic multicollinearity.")

    with st.expander("Full OLS Regression Summary"):
        st.text(str(model.summary()))