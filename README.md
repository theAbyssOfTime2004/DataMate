# DataMate

Offline question-answering agent over messy Vietnamese shop tabular data (CSV, Excel, SQLite) for the Zalo AI Challenge 2026 — DataMate track.

## Data Layout

Competition data is **read-only** and must reside outside the repository under `DATA_ROOT` (default `D:/zaloaic`):
- `D:/zaloaic/train/`
- `D:/zaloaic/public_test/`

Can be configured via `.env` (see `.env.example`) and accessed through `datamate.config.settings`.

## Installation

Create a virtual environment (Python 3.11+) and install in editable mode with dev dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
```

## Running Tests & Linter

```powershell
.\.venv\Scripts\pytest -q
.\.venv\Scripts\ruff check src tests evals
```

## Architecture & Evaluation

TBD.
