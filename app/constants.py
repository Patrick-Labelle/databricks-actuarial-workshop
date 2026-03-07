def fmt_dollars(value_m: float, precision: int = 1) -> str:
    """Format a value in $M for display. Auto-scales to $B for large values.

    Args:
        value_m: Value in millions of dollars.
        precision: Decimal places (default 1).
    """
    if abs(value_m) >= 1000:
        return f"${value_m / 1000:.{precision}f}B"
    return f"${value_m:.{precision}f}M"


SCENARIO_TYPES = [
    "Assumption Override",
    "Reserve Deterioration",
    "External Event",
    "Review Comment",
    "Judgment Adjustment",
    "Model Calibration",
]

APPROVAL_STATUSES = ["Draft", "Pending Review", "Approved"]

# ─── Reserve scenario presets ────────────────────────────────────────────────
# Each preset defines LDF multipliers, inflation adjustments, and scenario
# parameters for the Bootstrap Chain Ladder reserve simulator.
RESERVE_SCENARIO_PRESETS = {
    "Adverse Development": {
        "ldf_multiplier": 1.2, "inflation_adj": 0.0,
        "cv_mult": 1.3, "scenario": "adverse_development",
        "desc": "Late-lag LDFs inflated by 20%. Reserves develop worse than expected.",
    },
    "Judicial Inflation": {
        "ldf_multiplier": 1.0, "inflation_adj": 0.0,
        "cv_mult": 1.2, "scenario": "judicial_inflation",
        "desc": "Social inflation / nuclear verdicts: 1.3x on Auto lines at lags 24+.",
    },
    "Pandemic Tail": {
        "ldf_multiplier": 1.1, "inflation_adj": 0.0,
        "cv_mult": 1.4, "scenario": "pandemic_tail",
        "desc": "Extended development period (+6 months) due to delayed settlements.",
    },
    "Superimposed Inflation": {
        "ldf_multiplier": 1.0, "inflation_adj": 0.03,
        "cv_mult": 1.1, "scenario": "superimposed_inflation",
        "desc": "Accelerating calendar-year trend (CPI + 3%) across all lines.",
    },
    "Combined Stress": {
        "ldf_multiplier": 1.15, "inflation_adj": 0.02,
        "cv_mult": 1.3, "scenario": "adverse_development",
        "desc": "Combined adverse development + moderate inflation. Moderate but broad impact.",
    },
    "Extreme Tail": {
        "ldf_multiplier": 1.5, "inflation_adj": 0.05,
        "cv_mult": 2.0, "scenario": "adverse_development",
        "desc": "Extreme adverse scenario: 50% LDF inflation + 5% superimposed inflation.",
    },
}

SEVERITY_LEVELS = {
    "Mild":     0.70,
    "Moderate": 1.00,
    "Severe":   1.50,
    "Extreme":  2.50,
}

CANADIAN_PROVINCES = [
    "Ontario", "Quebec", "British Columbia", "Alberta", "Atlantic",
    "Manitoba", "Saskatchewan", "Nova Scotia", "New Brunswick",
    "Newfoundland", "PEI", "NWT", "Yukon",
]
