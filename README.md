# Lab-4: Applied Statistical Modeling & Interactive Web Dashboard

**Course:** Statistical Modeling with Python | **Dataset:** Option A — Medical Insurance Costs

## Repository Structure
```
.
├── app.py                  # Streamlit dashboard (3 tabs — Part 3)
├── requirements.txt
├── data/
│   └── insurance.csv       # Medical insurance dataset (1338 rows)
├── notebooks/
│   ├── analysis.ipynb      # Part 1 & Part 2: EDA, hypothesis tests, OLS, diagnostics (executed)
│   └── analysis.html       # Static HTML export of the notebook, for quick viewing
└── README.md
```

## How to Run

### Local IDE (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py            # opens at http://localhost:8501
```

### Google Colab (fallback)
```python
!pip install streamlit
# upload app.py and data/insurance.csv, then:
!streamlit run app.py & npx localtunnel --port 8501
```

To view the analysis notebook: open `notebooks/analysis.ipynb` in Jupyter/VS Code, or open
`notebooks/analysis.html` directly in a browser (no setup needed).

## Dataset Summary

**Medical Insurance Costs** — 1,338 records, 7 features:

| Column | Type | Description |
|---|---|---|
| age | numeric | Age of primary beneficiary |
| sex | categorical | male / female |
| bmi | numeric | Body Mass Index |
| children | numeric | Number of dependents covered |
| smoker | categorical | yes / no |
| region | categorical | northeast / northwest / southeast / southwest |
| charges | numeric (target) | Individual medical costs billed by insurance |

Source: [stedy/Machine-Learning-with-R-datasets](https://github.com/stedy/Machine-Learning-with-R-datasets) (public, commonly used teaching dataset).

## Synthesis of Statistical Findings

1. **Charges are heavily right-skewed** (skew ≈ 1.5+) and dominated by smoking status — smokers'
   median charges are roughly 3–4× those of non-smokers.
2. **Hypothesis Test 1 (Smokers vs. Non-Smokers, charges):** Shapiro-Wilk rejected normality in
   both groups, so a **Mann-Whitney U test** was used instead of a t-test. Result: **p ≈ 5.3e-130
   → Reject H0** — charges differ significantly by smoking status.
3. **Hypothesis Test 2 (One-Way ANOVA, charges across 4 regions):** F ≈ 2.97, **p ≈ 0.031 →
   Reject H0** at α = 0.05 — mean charges differ across regions (southeast highest), though the
   effect is modest compared to the smoking effect.
4. **OLS Regression** (`charges ~ age + bmi + children + smoker + sex + region + bmi:smoker`):
   `age`, `bmi`, `smoker`, and the `bmi × smoker` interaction are strong, significant predictors;
   `sex` and `region` contribute comparatively little once smoking/BMI/age are controlled for.
   The significant interaction term confirms BMI's effect on cost is far steeper for smokers.
5. **Gauss-Markov diagnostics:** VIF values are all low (~1.0–1.4) — no multicollinearity
   concern. However, residuals show **mild heteroscedasticity** (fan-shaped spread vs. fitted
   values) and **non-normality** (Jarque-Bera/Omnibus both reject normality), consistent with the
   right-skewed nature of raw medical cost data. A `log(charges)` transformation is the natural
   next step to improve model fit and satisfy assumptions more closely.

## Dashboard Tabs (`app.py`)
- **Tab 1 — Data Exploration:** Sidebar filters (age/BMI sliders, region/smoker/sex multi-select),
  reactive Plotly charts, live summary statistics.
- **Tab 2 — Hypothesis Testing Lab:** Choose a test (two-group comparison, Chi-square, or ANOVA),
  pick variables via dropdowns — the app auto-selects t-test vs. Mann-Whitney based on a live
  Shapiro-Wilk check, and reports the Reject/Fail-to-Reject conclusion at α = 0.05.
- **Tab 3 — Live Prediction & Diagnostics:** Enter a hypothetical individual's details to get a
  real-time charge prediction with 95% confidence and prediction intervals, plus residual
  diagnostic plots (Residuals vs. Fitted, Q-Q plot), normality tests, and VIF table.

## Optional Bonus
To claim the +5 mark bonus, deploy to [Streamlit Community Cloud](https://share.streamlit.io)
and add the live URL here: `<your-deployed-url>`
