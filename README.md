# ⚽ Football Match Predictor

Machine learning app that predicts football match outcomes using XGBoost. Trained on 50K+ historical matches.

## Installation
Downlaod everything from the Match-Predictor1 folder.


**Setup environment:**

python -m venv .venv

**Activate (Windows):**
.venv\Scripts\activate

**Activate (macOS/Linux):**
source .venv/Scripts/activate

**Install and run:**
python scripts/train_model.py
python src/web/app.py


Visit http://localhost:5000

## Features

- Match outcome predictions (Win/Draw/Loss)
- Historical head-to-head statistics
- Interactive web interface
- 54.2% prediction accuracy

## Tech Stack

Python • Flask • XGBoost • scikit-learn • pandas • matplotlib
