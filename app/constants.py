SCENARIO_TYPES = [
    "Assumption Override",
    "Catastrophe Event",
    "External Event",
    "Review Comment",
    "Judgment Adjustment",
    "Model Calibration",
]

APPROVAL_STATUSES = ["Draft", "Pending Review", "Approved"]

# ─── CAT scenario presets ─────────────────────────────────────────────────────
# Each preset defines severity multipliers (relative to baseline) applied to
# means, CVs, and correlation offsets.  Return period further scales means.
CAT_PRESETS = {
    "Hurricane":           {
        "means_mult": [2.8, 1.9, 1.3], "cv_mult": [1.4, 1.3, 1.2], "corr_add": 0.15,
        "desc": "Severe wind + storm surge. Property and Auto heavily impacted.",
    },
    "Earthquake":          {
        "means_mult": [3.8, 1.6, 1.4], "cv_mult": [1.6, 1.2, 1.3], "corr_add": 0.20,
        "desc": "Structural damage + fires. Property losses dominate.",
    },
    "Flood":               {
        "means_mult": [2.3, 1.7, 1.1], "cv_mult": [1.3, 1.2, 1.1], "corr_add": 0.12,
        "desc": "Widespread inundation. Both Property and Auto impacted.",
    },
    "Wildfire":            {
        "means_mult": [3.2, 1.4, 1.2], "cv_mult": [1.5, 1.1, 1.2], "corr_add": 0.15,
        "desc": "Structure and vehicle losses across a broad geographic area.",
    },
    "Pandemic":            {
        "means_mult": [1.1, 0.8, 2.8], "cv_mult": [1.2, 1.1, 1.5], "corr_add": 0.25,
        "desc": "Liability and healthcare claims surge; Auto frequency drops.",
    },
    "Market Crash":        {
        "means_mult": [1.2, 1.1, 1.8], "cv_mult": [1.3, 1.2, 1.4], "corr_add": 0.30,
        "desc": "Systemic financial stress — correlated across all lines.",
    },
    "Industrial Accident": {
        "means_mult": [2.0, 1.5, 2.2], "cv_mult": [1.3, 1.2, 1.4], "corr_add": 0.18,
        "desc": "Explosion, chemical spill, or structural collapse. High liability.",
    },
    "Zombies": {
        "means_mult": [144.0, 217.0, 316.0], "cv_mult": [7.71, 9.64, 6.43], "corr_add": 0.50,
        "desc": "Societal collapse — 90% of maximum possible loss across all lines.",
    },
    "Asteroid": {
        "means_mult": [200.0, 300.0, 500.0], "cv_mult": [9.0, 11.0, 8.0], "corr_add": 1.0,
        "desc": "Extinction-level event — total portfolio loss (100% of maximum) across all lines.",
    },
}

RETURN_PERIOD_MULT = {
    "1-in-50yr":  0.70,
    "1-in-100yr": 1.00,
    "1-in-250yr": 1.80,
    "1-in-500yr": 3.00,
}

CANADIAN_PROVINCES = [
    "Ontario", "Quebec", "British Columbia", "Alberta", "Atlantic",
    "Manitoba", "Saskatchewan", "Nova Scotia", "New Brunswick",
    "Newfoundland", "PEI", "NWT", "Yukon",
]
