import streamlit as st


def render(tab):
    with tab:
        st.subheader("Glossary & Methods Reference")
        st.caption(
            "A comprehensive reference for every model, metric, and concept used in this application"
        )

        # ── Models ──────────────────────────────────────────────────────────────
        st.markdown("## Models")

        with st.expander("**SARIMAX — Frequency Forecasting**", expanded=False):
            st.markdown("""
**What it does:** Projects future monthly claim counts (frequency) for each of the 40 portfolio segments (4 product lines × 10 provinces).

**Business context:** Claim frequency is the primary driver of reserve exposure. Accurate frequency forecasts tell actuaries how many new claims to expect, which feeds into reserve adequacy planning, staffing, and reinsurance purchasing decisions.

**Input data:**
- `gold_claims_monthly` — 84 months of historical claims (Jan 2019 – Dec 2025) per segment
- `features_segment_monthly` — Macro-economic features (unemployment, housing price index, housing starts) from StatCan

**Statistical method:**
- **SARIMA(1,1,1)(1,1,1)₁₂** — Seasonal AutoRegressive Integrated Moving Average
  - `(1,1,1)` = 1 AR lag, 1 difference, 1 MA lag (non-seasonal component)
  - `(1,1,1)₁₂` = seasonal component with 12-month period (captures annual patterns)
  - The "X" in SARIMAX = exogenous regressors (macro-economic variables)
- Fitted per segment using `statsmodels.SARIMAX`
- Produces point forecasts with 95% prediction intervals
- Registered in Unity Catalog Model Registry and served via a REST endpoint

**Where it appears:** Claims Forecast tab (segment-level), On-Demand Forecast tab (aggregate)
""")

        with st.expander("**GARCH(1,1) — Volatility Modelling**", expanded=False):
            st.markdown("""
**What it does:** Measures how the *variability* of claim frequency changes over time. When GARCH volatility rises, the confidence bands around forecasts widen.

**Business context:** Constant-variance models assume uncertainty is the same every month. In reality, insurance claims exhibit *volatility clustering* — periods of high variability tend to follow other high-variability periods (e.g., after a catastrophe or regulatory change). GARCH captures this, giving more realistic uncertainty estimates.

**Input data:**
- Residuals from the SARIMAX model (the part of claims the trend model couldn't explain)

**Statistical method:**
- **GARCH(1,1)** — Generalized AutoRegressive Conditional Heteroskedasticity
  - σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}
  - **α** (alpha): How quickly new shocks affect volatility — high α means volatility reacts strongly to surprises
  - **β** (beta): How long high-volatility periods persist — high β means slow mean-reversion
  - **α + β** close to 1.0 indicates highly persistent volatility regimes
- An **ARCH-LM test** checks whether GARCH effects are statistically significant for each segment
- Only applied when the ARCH-LM p-value < 0.10

**Where it appears:** Claims Forecast tab (orange volatility area on right axis, GARCH Diagnostics expander)
""")

        with st.expander("**Chain Ladder — Deterministic Reserve Estimation**", expanded=False):
            st.markdown("""
**What it does:** Estimates total IBNR (Incurred But Not Reported) reserves using the industry-standard chain ladder method. Produces a single "best estimate" — no distribution.

**Business context:** The chain ladder is the most widely used actuarial reserving technique. It answers: "Based on how past claims have developed, how much more do we expect to pay on existing claims?" Every P&C insurer uses some form of chain ladder for regulatory filings.

**Input data:**
- `gold_reserve_triangle` — Loss development triangle showing cumulative paid claims by accident month and development lag

**Statistical method:**
- **Weighted Link Ratios (LDFs):** For each development lag k, compute f_k = Σ C(i,k+1) / Σ C(i,k)
  - These "development factors" capture how much claims grow from one lag to the next
  - Weighted by volume (larger accident periods have more influence)
- **Ultimate projection:** For each accident month, multiply the latest observed cumulative value by all remaining LDFs to project to ultimate
- **IBNR** = Projected ultimate − Current cumulative paid
- **Mack's variance:** σ²_k estimates the uncertainty in each LDF, used to derive the standard error of the reserve estimate

**Where it appears:** Scenario Analysis tab (triangle exhibit, development factors expander)
""")

        with st.expander("**Bootstrap Chain Ladder — Stochastic Reserve Distribution**", expanded=False):
            st.markdown("""
**What it does:** Extends the chain ladder into a full probability distribution of reserve outcomes. Instead of one best estimate, it produces thousands of possible reserve outcomes, enabling risk quantiles like VaR and CVaR.

**Business context:** Regulators (OSFI, Solvency II, IFRS 17) require insurers to quantify *reserve uncertainty*, not just provide point estimates. The bootstrap answers: "What's the range of outcomes, and how bad could it get?" This directly feeds capital adequacy calculations.

**Input data:**
- `gold_reserve_triangle` — Same triangle as deterministic chain ladder
- Chain ladder fitted LDFs and Mack standard errors
- Per-line coefficients of variation (CV) for reserve volatility

**Statistical method:**
1. **Fit chain ladder** — Compute weighted LDFs and Mack standard errors se(f_k) = σ_k / √(Σ C(i,k))
2. **Perturb LDFs** — For each replication, draw f_k^boot ~ N(f_k, se(f_k)²), floored at 1.0
3. **Project per accident month** — Each accident month has a `max_obs_lag`; only LDFs beyond that lag apply
4. **Add process variance** — Lognormal noise calibrated to per-line CVs and correlated across product lines
5. **Aggregate** — Sum across all accident months and product lines for total IBNR
6. **Repeat** 40,000+ times to build the full distribution
- Implemented as distributed Ray tasks across 4 workers (24 concurrent tasks)

**Where it appears:** Reserve Adequacy tab (IBNR distribution chart), Scenario Analysis tab (custom stress tests)
""")

        with st.expander("**Bootstrap Reserve Serving Endpoint**", expanded=False):
            st.markdown("""
**What it does:** An on-demand version of the bootstrap simulation, served as a REST API via Databricks Model Serving. Allows real-time "what if" scenario analysis without re-running the full pipeline.

**Business context:** Actuaries and risk managers need to quickly test reserve scenarios — "What if development factors deteriorate by 20%?" or "What's the impact of 3% superimposed inflation?" This endpoint answers those questions in seconds.

**Input data (request parameters):**
- `scenario` — Scenario type (baseline, adverse_development, judicial_inflation, pandemic_tail, superimposed_inflation)
- `ldf_multiplier` — Scales development factors (>1.0 = worse development)
- `inflation_adj` — Calendar-year superimposed inflation rate
- `cv_*` — Per-line reserve volatility (Personal Auto, Commercial Auto, Homeowners, Commercial Property)
- `n_replications` — Number of bootstrap samples (5,000–50,000)

**Output:**
- best_estimate_M, var_99_M, var_995_M, cvar_99_M, max_ibnr_M, reserve_risk_capital_M

**Where it appears:** Scenario Analysis tab (custom scenarios), Reserve Adequacy tab (Mack vs Bootstrap comparison)
""")

        st.divider()

        # ── Risk Metrics ────────────────────────────────────────────────────────
        st.markdown("## Risk Metrics")

        with st.expander("**IBNR — Incurred But Not Reported**", expanded=False):
            st.markdown("""
**Definition:** The estimated total amount of claims that have already occurred but have not yet been reported to the insurer, or have been reported but not yet fully developed to their ultimate value.

**Why it matters:** IBNR is typically the largest liability on a P&C insurer's balance sheet. Underestimating IBNR means the insurer doesn't hold enough reserves; overestimating ties up capital unnecessarily.

**How it's calculated here:** Chain ladder projection to ultimate minus current cumulative paid, summed across all accident months and product lines.
""")

        with st.expander("**VaR — Value at Risk**", expanded=False):
            st.markdown("""
**Definition:** The threshold value such that the probability of the reserve exceeding it is a specified percentage.

**Key levels used in this app:**
- **VaR 99% (1-in-100 year):** The reserve level exceeded in only 1% of bootstrap replications. You'd expect to need this much reserve only once a century.
- **VaR 99.5% (1-in-200 year):** The reserve level exceeded in only 0.5% of replications. This is the threshold used by Solvency II for reserve risk capital and approximated by OSFI's MCT.

**Example:** If VaR 99.5% = $21.1B, that means in 99.5% of simulated scenarios, total IBNR stays below $21.1B. Only in the worst 0.5% of outcomes would reserves exceed this level.
""")

        with st.expander("**CVaR — Conditional Value at Risk (Expected Shortfall)**", expanded=False):
            st.markdown("""
**Definition:** The average reserve across all scenarios *worse* than the VaR threshold. Also called Expected Shortfall or Tail VaR.

**Why it matters:** VaR tells you the boundary of the tail; CVaR tells you *how bad things get* once you're in the tail. Two distributions can have the same VaR but very different CVaRs — CVaR captures the severity of extreme outcomes.

**How it's calculated:** CVaR 99% = average of all bootstrap replications above the 99th percentile.
""")

        with st.expander("**Reserve Risk Capital**", expanded=False):
            st.markdown("""
**Definition:** The amount of capital an insurer must hold specifically to cover adverse reserve development. In this app, it's defined as VaR 99.5% − Best Estimate IBNR.

**Regulatory context:**
- **Solvency II:** Reserve risk is a component of the Solvency Capital Requirement (SCR), calibrated to VaR 99.5%
- **OSFI MCT:** Uses prescribed risk factors by line of business (simpler than simulation)
- **IFRS 17:** Requires a Risk Adjustment that reflects reserve uncertainty — often informed by bootstrap distributions
""")

        with st.expander("**LDF — Loss Development Factor**", expanded=False):
            st.markdown("""
**Definition:** The ratio of cumulative claims at development lag k+1 to cumulative claims at lag k. Also called link ratios or age-to-age factors.

**Formula:** f_k = Σ C(i,k+1) / Σ C(i,k) — weighted across accident periods

**Example:** An LDF of 1.15 from lag 12 to lag 24 means that, on average, cumulative paid claims grow by 15% between 12 and 24 months of development.

**LDF Volatility:** The standard deviation of individual link ratios around the weighted average. Higher volatility means less predictable development, which increases reserve risk.

**Where it appears:** Scenario Analysis tab (development factors expander), Reserve Adequacy tab (LDF volatility section)
""")

        with st.expander("**Coefficient of Variation (CV)**", expanded=False):
            st.markdown("""
**Definition:** The ratio of standard deviation to mean (CV = σ/μ). Measures relative variability — a CV of 0.15 means the standard deviation is 15% of the mean.

**In this app:** Per-line CVs control how much noise is added during the bootstrap process variance step. Higher CV = wider IBNR distribution for that product line.

**Default values:**
| Product Line | CV | Interpretation |
|---|---|---|
| Personal Auto | 0.15 | Most predictable — high volume, short tail |
| Homeowners | 0.12 | Low volatility — weather events are seasonal but predictable in aggregate |
| Commercial Auto | 0.18 | Moderate — includes bodily injury with longer development |
| Commercial Property | 0.20 | Most volatile — large individual losses, longer tail |
""")

        st.divider()

        # ── Regulatory Frameworks ───────────────────────────────────────────────
        st.markdown("## Regulatory Frameworks")

        with st.expander("**OSFI MCT — Minimum Capital Test**", expanded=False):
            st.markdown("""
**What it is:** OSFI's (Office of the Superintendent of Financial Institutions) capital adequacy framework for Canadian P&C insurers. It's a **factor-based** approach — prescribed risk factors are applied to balance sheet items.

**MCT Ratio = Capital Available / Minimum Capital Required × 100%**

**Key thresholds:**
- **100%** — Minimum regulatory requirement
- **150%** — OSFI supervisory target
- **170–220%** — Typical range for well-capitalized Canadian insurers

**Components (simplified in this app):**
- **Claims Reserve Risk:** OSFI risk margins (5–10%) × outstanding case reserves by line
- **Premium Risk:** OSFI risk factors (12–22%) × net earned premium by line
- **Diversification Credit:** ρ=0.50 correlation across lines (square-root formula)

**Not included here:** Market risk, credit risk, operational risk (~30–40% of total MCT capital)

**Contrast with bootstrap:** The MCT is a standardized formula; the bootstrap is an internal model. Both aim to quantify capital adequacy, but the bootstrap captures the actual tail shape of the insurer's reserve distribution.
""")

        with st.expander("**Solvency II**", expanded=False):
            st.markdown("""
**What it is:** The EU's risk-based regulatory framework for insurers. More sophisticated than MCT — allows internal models (like bootstrap) as alternatives to the standard formula.

**Relevance to this app:**
- The **99.5% VaR** threshold used throughout this app is the Solvency II calibration standard for the Solvency Capital Requirement (SCR)
- Reserve risk is one component of the non-life underwriting risk module
- Internal models (like the bootstrap chain ladder) can be approved by regulators to replace the standard formula

**Not implemented here:** Full Solvency II standard formula (which includes market risk, counterparty default risk, etc.)
""")

        with st.expander("**IFRS 17 — Risk Adjustment**", expanded=False):
            st.markdown("""
**What it is:** The international accounting standard for insurance contracts, effective 2023. Requires a **Risk Adjustment** that reflects the uncertainty in future cash flows.

**Relevance to this app:**
- The bootstrap reserve distribution directly supports IFRS 17 Risk Adjustment calculations
- The distribution quantifies the range of possible outcomes — the Risk Adjustment is typically set at a confidence level (e.g., 75th–85th percentile)
- Unlike Solvency II (which prescribes 99.5%), IFRS 17 lets the insurer choose the confidence level, disclosed in financial statements
""")

        st.divider()

        # ── Scenario Types ──────────────────────────────────────────────────────
        st.markdown("## Reserve Scenario Types")

        with st.expander("**Scenario definitions**", expanded=False):
            st.markdown("""
| Scenario | What it models | Key parameters |
|---|---|---|
| **Baseline** | Normal reserve development — no stress applied | LDF multiplier = 1.0, no inflation |
| **Adverse Development** | LDFs inflated — reserves develop worse than expected | LDF multiplier > 1.0 |
| **Judicial Inflation** | Social inflation / nuclear verdicts increasing Auto severity at later lags | Primarily affects Auto lines at lags 24+ |
| **Pandemic Tail** | Delayed settlements extending the development period | LDF multiplier + extended tail |
| **Superimposed Inflation** | Calendar-year trend (CPI + X%) applied across all lines and development periods | inflation_adj > 0 |
| **Combined Stress** | Multiple adverse factors simultaneously | LDF + inflation + elevated CV |

**Severity levels** scale the scenario parameters:
- **Mild** (0.7×) — Minor deterioration
- **Moderate** (1.0×) — Central stress estimate
- **Severe** (1.5×) — Significant adverse movement
- **Extreme** (2.5×) — Tail event / worst-case planning
""")

        st.divider()

        # ── Data Pipeline ───────────────────────────────────────────────────────
        st.markdown("## Data Pipeline")

        with st.expander("**End-to-end data flow**", expanded=False):
            st.markdown("""
```
Raw CDC events (synthetic — Weibull development, CY inflation)
  │
  ├─ bronze_claims          ← Spark Declarative Pipeline (streaming append)
  │   └─ gold_claims_monthly    ← Monthly aggregation by segment
  │       └─ silver_rolling_features  ← Rolling 3m/6m stats, YoY change
  │           └─ features_segment_monthly  ← + macro features (Feature Store)
  │               └─ SARIMAX frequency model
  │                   └─ predictions_frequency_forecast
  │
  ├─ bronze_reserve_cdc     ← Streaming reserve development
  │   └─ silver_reserves (SCD Type 2)  ← Tracks reserve revisions over time
  │       └─ gold_reserve_triangle     ← Standard loss development triangle
  │           ├─ Chain Ladder + Bootstrap (Ray)
  │           │   ├─ predictions_bootstrap_reserves  ← VaR/CVaR distribution
  │           │   ├─ predictions_reserve_scenarios   ← Pre-computed stress scenarios
  │           │   ├─ predictions_reserve_evolution   ← 12-month adequacy outlook
  │           │   ├─ predictions_runoff_projection   ← Surplus trajectory
  │           │   └─ predictions_ldf_volatility      ← Development factor risk
  │           │
  │           └─ Bootstrap Reserve endpoint (Model Serving)
  │
  └─ StatCan macro data (unemployment, housing)
      └─ features_segment_monthly (exogenous regressors)
```

**Table naming convention:**
- `raw_*` — Ingested source data
- `bronze_*` — Streaming append-only (SDP)
- `silver_*` — Cleaned / SCD Type 2 (SDP)
- `gold_*` — Business-ready aggregations (SDP)
- `features_*` — Feature Store tables (point-in-time joins)
- `predictions_*` — Model output tables
""")

        with st.expander("**Key tables reference**", expanded=False):
            st.markdown("""
| Table | Description | Key columns |
|---|---|---|
| `gold_claims_monthly` | Monthly claims by segment | segment_id, product_line, region, month, claims_count, total_incurred, avg_severity, earned_premium |
| `gold_reserve_triangle` | Loss development triangle | segment_id, product_line, accident_month, dev_lag, cumulative_paid, cumulative_incurred, case_reserve |
| `predictions_frequency_forecast` | SARIMAX forecasts | segment_id, month, record_type, claims_count, forecast_mean, forecast_lo95, forecast_hi95, cond_volatility |
| `predictions_bootstrap_reserves` | Reserve distribution | best_estimate_M, var_99_M, var_995_M, cvar_99_M, reserve_risk_capital_M, max_ibnr_M |
| `predictions_reserve_scenarios` | Pre-computed stress | scenario_label, best_estimate_M, var_99_M, var_995_M, cvar_99_M, var_995_vs_baseline |
| `predictions_reserve_evolution` | 12-month outlook | forecast_month, best_estimate_M, var_99_M, var_995_M, reserve_risk_capital_M |
| `predictions_runoff_projection` | Surplus trajectory | month, surplus_p05/p25/p50/p75/p95, ruin_probability |
| `predictions_ldf_volatility` | LDF risk by line | product_line, avg_ldf, std_ldf, n_factors |
| `features_segment_monthly` | Feature Store | segment_id, month, claims_count, rolling stats, macro features |
""")
