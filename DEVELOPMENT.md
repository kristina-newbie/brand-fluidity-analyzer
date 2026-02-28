# Development Setup

Use **one virtual environment at the project root** so you don’t end up with multiple venvs (e.g. in both root and `backend/`).

**Quick path:** From project root run `./setup.sh` — it creates `.venv`, installs deps, sets up NLTK and `backend/.env`, and creates upload dirs. No directory changes required.

## 1. Create and use the venv (project root)

From the **project root** (`brand-fluidity-analyzer/`):

```bash
# Create venv at project root (recommended: .venv)
python3 -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# Or on Windows
# .venv\Scripts\activate
```

## 2. Install dependencies

With the venv activated, install the backend (and dev) requirements:

```bash
# From project root
pip install -r backend/requirements-dev.txt
```

This installs everything in `backend/requirements.txt` plus dev tools (e.g. pylint).

Optional one-time setup:

```bash
# NLTK data (if not already done)
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

# .env (copy from example and add ANTHROPIC_API_KEY)
cp backend/.env.example backend/.env
```

## 3. Run the backend

```bash
# From project root, with venv activated
cd backend && python main.py
```

Or from project root without `cd`: `python backend/main.py` (ensure you run from root so paths work).

## 4. Run Pylint on the backend

With the same venv activated (you’ve installed `requirements-dev.txt`):

```bash
# From project root
pylint backend
```

To lint only Python files (no `__pycache__`, etc.):

```bash
pylint backend/*.py
```

Config is in the project root `.pylintrc`; adjust there if you want to enable/disable messages or change line length.

## Summary

| Step              | Command |
|-------------------|---------|
| Create venv       | `python3 -m venv .venv` (at project root) |
| Activate          | `source .venv/bin/activate` |
| Install deps      | `pip install -r backend/requirements-dev.txt` |
| Run backend       | `cd backend && python main.py` |
| Lint backend      | `pylint backend` |
