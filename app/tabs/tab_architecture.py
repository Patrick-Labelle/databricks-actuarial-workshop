"""Architecture tab — visual reference architecture of the workshop platform."""

import streamlit as st


def render(tab):
    with tab:
        st.html(_ARCHITECTURE_HTML)


# ─── Architecture diagram HTML ──────────────────────────────────────────────

_ARCHITECTURE_HTML = """
<style>
/* ── Architecture wrapper ── */
.arch-root {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    padding: 4px 0 0 0;
}
.arch-root h2 {
    margin: 0 0 2px 0;
    font-size: 1.4rem;
    color: #1B3A5C;
}
.arch-root .subtitle {
    color: #64748B;
    font-size: 0.88rem;
    margin: 0 0 18px 0;
}

/* ── Three-column layout ── */
.arch-flow {
    display: flex;
    align-items: stretch;
    gap: 0;
    min-height: 520px;
}

/* ── Left: Data Sources ── */
.arch-sources {
    width: 165px;
    min-width: 165px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-right: 6px;
}
.arch-sources .col-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748B;
    margin-bottom: 4px;
}
.source-card {
    background: #fff;
    border: 1.5px solid #E2E8F0;
    border-radius: 8px;
    padding: 10px 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.source-card:hover {
    border-color: #CBD5E1;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.source-card .s-icon {
    font-size: 1.3rem;
    flex-shrink: 0;
}
.source-card .s-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #1B3A5C;
    line-height: 1.2;
}
.source-card .s-detail {
    font-size: 0.68rem;
    color: #94A3B8;
    line-height: 1.25;
    margin-top: 1px;
}

/* ── Arrow columns ── */
.arch-arrow-col {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    min-width: 36px;
}

/* ── Center: Databricks Platform ── */
.arch-platform {
    flex: 1;
    border: 2.5px solid #E8590C;
    border-radius: 14px;
    background: linear-gradient(180deg, #FFF7ED 0%, #FFFFFF 40%);
    padding: 0;
    position: relative;
    overflow: hidden;
}
.platform-header {
    background: linear-gradient(135deg, #E8590C, #C2410C);
    color: #fff;
    padding: 10px 18px;
    font-weight: 700;
    font-size: 0.88rem;
    letter-spacing: 0.02em;
    display: flex;
    align-items: center;
    gap: 8px;
}
.platform-header .db-icon {
    font-size: 1.1rem;
}
.uc-banner {
    background: #FEF3C7;
    border-bottom: 1px solid #FDE68A;
    padding: 6px 18px;
    font-size: 0.73rem;
    color: #92400E;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
}
.platform-body {
    padding: 14px 16px 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

/* ── Medallion row ── */
.medallion-row {
    display: flex;
    align-items: center;
    gap: 0;
}
.medallion-box {
    flex: 1;
    border-radius: 8px;
    padding: 8px 10px;
    text-align: center;
}
.medallion-box .m-title {
    font-weight: 700;
    font-size: 0.78rem;
    margin-bottom: 2px;
}
.medallion-box .m-tables {
    font-size: 0.65rem;
    opacity: 0.8;
    line-height: 1.3;
}
.m-bronze { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
.m-silver { background: #E0E7FF; color: #3730A3; border: 1px solid #C7D2FE; }
.m-gold   { background: #FEF9C3; color: #854D0E; border: 1px solid #FDE047; }
.m-arrow {
    font-size: 1.1rem;
    color: #94A3B8;
    padding: 0 4px;
    flex-shrink: 0;
}

/* ── Inner component grid ── */
.inner-grid {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.inner-card {
    flex: 1;
    min-width: 120px;
    background: #fff;
    border: 1.5px solid #E2E8F0;
    border-radius: 8px;
    padding: 10px 12px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.inner-card:hover {
    border-color: #CBD5E1;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.inner-card .ic-icon { font-size: 1.1rem; margin-bottom: 4px; }
.inner-card .ic-title {
    font-weight: 700;
    font-size: 0.76rem;
    color: #1B3A5C;
    line-height: 1.2;
    margin-bottom: 2px;
}
.inner-card .ic-desc {
    font-size: 0.67rem;
    color: #64748B;
    line-height: 1.35;
}
.inner-card .ic-tag {
    display: inline-block;
    margin-top: 5px;
    padding: 1px 7px;
    border-radius: 4px;
    font-size: 0.62rem;
    font-weight: 600;
}
.tag-purple { background: #F3E8FF; color: #6B21A8; }
.tag-green  { background: #DCFCE7; color: #166534; }
.tag-amber  { background: #FEF3C7; color: #92400E; }
.tag-blue   { background: #DBEAFE; color: #1E40AF; }
.tag-red    { background: #FEE2E2; color: #991B1B; }

/* Section labels inside platform */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94A3B8;
    margin-top: 2px;
}

/* ── Platform footer badges ── */
.platform-footer {
    display: flex;
    gap: 8px;
    padding: 10px 16px;
    border-top: 1px solid #FDE68A;
    background: #FFFBEB;
    flex-wrap: wrap;
}
.pf-badge {
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    background: #fff;
    border: 1px solid #E2E8F0;
    color: #475569;
}

/* ── Right: Outputs ── */
.arch-outputs {
    width: 155px;
    min-width: 155px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-left: 6px;
}
.arch-outputs .col-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748B;
    margin-bottom: 4px;
}
.output-card {
    background: #fff;
    border: 1.5px solid #E2E8F0;
    border-radius: 8px;
    padding: 8px 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.output-card:hover {
    border-color: #CBD5E1;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.output-card .o-icon { font-size: 1.1rem; flex-shrink: 0; }
.output-card .o-label {
    font-size: 0.76rem;
    font-weight: 600;
    color: #1B3A5C;
    line-height: 1.2;
}

/* ── Bottom feature cards ── */
.arch-highlights {
    display: flex;
    gap: 14px;
    margin-top: 20px;
    flex-wrap: wrap;
}
.highlight-card {
    flex: 1;
    min-width: 180px;
    background: #fff;
    border: 1.5px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
    transition: box-shadow 0.2s, border-color 0.2s;
}
.highlight-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    border-color: #CBD5E1;
}
.highlight-card .h-icon {
    font-size: 1.8rem;
    margin-bottom: 8px;
}
.highlight-card .h-title {
    font-weight: 700;
    font-size: 0.88rem;
    color: #1B3A5C;
    margin-bottom: 4px;
}
.highlight-card .h-desc {
    font-size: 0.75rem;
    color: #64748B;
    line-height: 1.4;
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    .source-card, .inner-card, .output-card, .highlight-card {
        background: #1E293B; border-color: #334155;
    }
    .source-card:hover, .inner-card:hover, .output-card:hover, .highlight-card:hover {
        border-color: #475569; box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    .source-card .s-label, .inner-card .ic-title, .output-card .o-label, .highlight-card .h-title {
        color: #E2E8F0;
    }
    .source-card .s-detail, .inner-card .ic-desc, .highlight-card .h-desc {
        color: #94A3B8;
    }
    .arch-platform { background: linear-gradient(180deg, #1C1917 0%, #0F172A 40%); }
    .uc-banner { background: #422006; border-color: #854D0E; }
    .platform-footer { background: #1C1917; border-color: #854D0E; }
    .pf-badge { background: #1E293B; border-color: #334155; color: #CBD5E1; }
}
</style>

<div class="arch-root">
<h2>Reference Architecture</h2>
<p class="subtitle">End-to-end data, ML, and serving flow on the Databricks Intelligence Platform</p>

<!-- ═══ THREE-COLUMN FLOW ═══ -->
<div class="arch-flow">

<!-- ── LEFT: DATA SOURCES ── -->
<div class="arch-sources">
    <div class="col-title">Data Sources</div>

    <div class="source-card">
        <div class="s-icon">🎲</div>
        <div>
            <div class="s-label">Claims Generator</div>
            <div class="s-detail">Weibull CDF patterns, CY inflation, 40 segments</div>
        </div>
    </div>

    <div class="source-card">
        <div class="s-icon">📊</div>
        <div>
            <div class="s-label">Reserve Development</div>
            <div class="s-detail">Incremental paid & incurred CDC events</div>
        </div>
    </div>

    <div class="source-card">
        <div class="s-icon">🏛️</div>
        <div>
            <div class="s-label">StatCan API</div>
            <div class="s-detail">Unemployment, housing prices, housing starts</div>
        </div>
    </div>

    <div class="source-card" style="margin-top:auto">
        <div class="s-icon">📋</div>
        <div>
            <div class="s-label">84 Months</div>
            <div class="s-detail">Jan 2019 – Dec 2025</div>
        </div>
    </div>
</div>

<!-- ── LEFT ARROW ── -->
<div class="arch-arrow-col">
    <svg width="28" height="80" viewBox="0 0 28 80">
        <path d="M2 40 L20 40 M14 32 L22 40 L14 48" stroke="#E8590C" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
</div>

<!-- ── CENTER: DATABRICKS PLATFORM ── -->
<div class="arch-platform">
    <div class="platform-header">
        <span class="db-icon">◆</span>
        Databricks Data Intelligence Platform
    </div>
    <div class="uc-banner">
        🔒 Unity Catalog — Governance, Lineage &amp; Security &nbsp;|&nbsp; 3 Schemas: actuarial_data · actuarial_models · actuarial_app
    </div>

    <div class="platform-body">

        <!-- Medallion Architecture -->
        <div class="section-label">Spark Declarative Pipelines — Medallion Architecture</div>
        <div class="medallion-row">
            <div class="medallion-box m-bronze">
                <div class="m-title">Bronze</div>
                <div class="m-tables">bronze_claims<br>bronze_reserve_cdc</div>
            </div>
            <span class="m-arrow">→</span>
            <div class="medallion-box m-silver">
                <div class="m-title">Silver</div>
                <div class="m-tables">silver_reserves (SCD2)<br>silver_rolling_features</div>
            </div>
            <span class="m-arrow">→</span>
            <div class="medallion-box m-gold">
                <div class="m-title">Gold</div>
                <div class="m-tables">gold_claims_monthly<br>gold_reserve_triangle</div>
            </div>
        </div>

        <!-- Feature Store + ML Training -->
        <div class="section-label">Feature Engineering &amp; ML Training (Ray on Spark)</div>
        <div class="inner-grid">
            <div class="inner-card">
                <div class="ic-icon">🧮</div>
                <div class="ic-title">Feature Store</div>
                <div class="ic-desc">Point-in-time joins, rolling stats, macro features</div>
                <span class="ic-tag tag-green">features_segment_monthly</span>
            </div>
            <div class="inner-card">
                <div class="ic-icon">📉</div>
                <div class="ic-title">SARIMAX + GARCH</div>
                <div class="ic-desc">Per-segment frequency forecasting with volatility modelling</div>
                <span class="ic-tag tag-purple">Frequency Forecaster</span>
            </div>
            <div class="inner-card">
                <div class="ic-icon">🔺</div>
                <div class="ic-title">Chain Ladder</div>
                <div class="ic-desc">Weighted LDFs, Mack variance, deterministic IBNR</div>
                <span class="ic-tag tag-purple">Reserve Estimation</span>
            </div>
            <div class="inner-card">
                <div class="ic-icon">🎰</div>
                <div class="ic-title">Bootstrap Chain Ladder</div>
                <div class="ic-desc">40K+ replications → full IBNR distribution, VaR, CVaR</div>
                <span class="ic-tag tag-purple">Stochastic Reserve</span>
            </div>
        </div>

        <!-- Serving & Intelligence -->
        <div class="section-label">Serving &amp; Intelligence</div>
        <div class="inner-grid">
            <div class="inner-card">
                <div class="ic-icon">🚀</div>
                <div class="ic-title">Model Serving</div>
                <div class="ic-desc">Frequency Forecaster + Bootstrap Reserve endpoints</div>
                <span class="ic-tag tag-amber">REST API</span>
            </div>
            <div class="inner-card">
                <div class="ic-icon">🐘</div>
                <div class="ic-title">Lakebase</div>
                <div class="ic-desc">Scale-to-zero Postgres — synced tables, annotations</div>
                <span class="ic-tag tag-green">Sub-100ms reads</span>
            </div>
            <div class="inner-card">
                <div class="ic-icon">💬</div>
                <div class="ic-title">Genie AI</div>
                <div class="ic-desc">Natural language SQL with actuarial domain instructions</div>
                <span class="ic-tag tag-blue">Text-to-SQL</span>
            </div>
            <div class="inner-card">
                <div class="ic-icon">🤖</div>
                <div class="ic-title">AI Gateway</div>
                <div class="ic-desc">Llama 3.3 70B with rate limits &amp; usage tracking</div>
                <span class="ic-tag tag-blue">Foundation Models</span>
            </div>
        </div>

    </div>

    <!-- Platform footer -->
    <div class="platform-footer">
        <span class="pf-badge">Databricks Apps</span>
        <span class="pf-badge">Serverless Compute</span>
        <span class="pf-badge">MLflow Tracing</span>
        <span class="pf-badge">UC Model Registry</span>
        <span class="pf-badge">Asset Bundles (IaC)</span>
    </div>
</div>

<!-- ── RIGHT ARROW ── -->
<div class="arch-arrow-col">
    <svg width="28" height="80" viewBox="0 0 28 80">
        <path d="M2 40 L20 40 M14 32 L22 40 L14 48" stroke="#E8590C" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
</div>

<!-- ── RIGHT: APP OUTPUTS ── -->
<div class="arch-outputs">
    <div class="col-title">App Outputs</div>

    <div class="output-card">
        <div class="o-icon">💬</div>
        <div class="o-label">Risk Assistant</div>
    </div>
    <div class="output-card">
        <div class="o-icon">📈</div>
        <div class="o-label">Claims Forecast</div>
    </div>
    <div class="output-card">
        <div class="o-icon">💰</div>
        <div class="o-label">Reserve Adequacy</div>
    </div>
    <div class="output-card">
        <div class="o-icon">📋</div>
        <div class="o-label">Scenario Analysis</div>
    </div>
    <div class="output-card">
        <div class="o-icon">⚡</div>
        <div class="o-label">On-Demand Forecast</div>
    </div>
    <div class="output-card">
        <div class="o-icon">🗺️</div>
        <div class="o-label">Geography</div>
    </div>
    <div class="output-card" style="margin-top:auto">
        <div class="o-icon">🔍</div>
        <div class="o-label">MLflow Traces</div>
    </div>
</div>

</div>

<!-- ═══ BOTTOM HIGHLIGHT CARDS ═══ -->
<div class="arch-highlights">
    <div class="highlight-card">
        <div class="h-icon">🎰</div>
        <div class="h-title">Stochastic Reserving</div>
        <div class="h-desc">Bootstrap Chain Ladder with 40,000+ replications for full reserve risk distribution</div>
    </div>
    <div class="highlight-card">
        <div class="h-icon">⚡</div>
        <div class="h-title">Real-time Scoring</div>
        <div class="h-desc">Model Serving endpoints for on-demand frequency forecasts and reserve simulations</div>
    </div>
    <div class="highlight-card">
        <div class="h-icon">🤖</div>
        <div class="h-title">AI-Powered</div>
        <div class="h-desc">Genie text-to-SQL and multi-tool LLM chatbot for natural language analytics</div>
    </div>
    <div class="highlight-card">
        <div class="h-icon">📦</div>
        <div class="h-title">Infrastructure as Code</div>
        <div class="h-desc">Databricks Asset Bundles — all resources version-controlled and reproducible</div>
    </div>
</div>

</div>
"""
