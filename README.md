# ⚽ Football Match Predictor

Machine learning app that predicts football match outcomes using XGBoost. Trained on 50K+ historical matches.

## Installation
**Clone and navigate:**
git clone https://github.com/yourusername/match-predictor.git
cd match-predictor


**Setup environment:**

python -m venv .venv

**Activate (Windows):**
.venv\Scripts\activate

**Activate (macOS/Linux):**
source .venv/bin/activate

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

## License

MIT
