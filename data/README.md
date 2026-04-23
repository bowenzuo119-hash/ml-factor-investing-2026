# data/

This folder holds all raw and processed datasets used by the project.

**Nothing in here is committed to git.** Data files (parquet, csv, feather, h5, pkl) are gitignored because they are large and because the source of truth should be the *code that produces them*, not a snapshot.

## How to populate this folder

Run the download / preparation scripts in `src/` (e.g., `python -m src.download_data`). Those scripts pull the raw data from its original source and save it here as parquet files.

## Expected layout (once populated)

Raw pulls go into `data/raw/` (e.g., `data/raw/prices.parquet`, `data/raw/fundamentals.parquet`). Cleaned and feature-engineered datasets go into `data/processed/` (e.g., `data/processed/panel.parquet`). Any small reference tables (ticker lists, sector maps) live directly under `data/`.

## What is committed

Only this `README.md` and a `.gitkeep` file, so the folder exists in a fresh clone. Every other file pattern under `data/` is excluded via `.gitignore`.
