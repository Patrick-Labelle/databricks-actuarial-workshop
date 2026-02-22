# Databricks Actuarial Workshop

Actuarial workshop materials: a Databricks App (Streamlit dashboard) and demo content for workshops.

## Repository structure

```
.
├── app/                 # Databricks App (Streamlit)
│   ├── app.py           # Main application
│   ├── app.yaml         # App configuration
│   └── requirements.txt # App dependencies
├── demos/               # Workshop demo content
│   └── README.md        # Notebooks, scripts, sample data
└── README.md
```

## App (`app/`)

Run the Actuarial Risk Dashboard as a Databricks App. From the `app/` directory:

- **Local:** `streamlit run app.py`
- **Databricks:** Deploy via the Apps UI or CLI using `app.yaml`.

See `app/requirements.txt` for Python dependencies.

## Demos (`demos/`)

Workshop demos, notebooks, and sample scripts live in `demos/`. Use this folder for:

- Jupyter/Databricks notebooks
- Sample SQL or Python scripts
- Small reference datasets or links to data

## Development

- App code: `app/`
- Demo content: `demos/`
- Root-level config (e.g. `.gitignore`) applies to the whole repo.
