import os
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

df = pd.read_csv("data/results.csv")

with open("models/predictor.pkl", "rb") as f:
    saved_data = pickle.load(f)
    teams = saved_data["teams"]
    team_to_idx = saved_data["team_to_idx"]

df["home_encoded"] = df["home_team"].map(team_to_idx).fillna(-1).astype(int)
df["away_encoded"] = df["away_team"].map(team_to_idx).fillna(-1).astype(int)

df["home_avg_scored"] = df.groupby("home_team")["home_score"].transform("mean")
df["home_avg_conceded"] = df.groupby("home_team")["away_score"].transform("mean")
df["away_avg_scored"] = df.groupby("away_team")["away_score"].transform("mean")
df["away_avg_conceded"] = df.groupby("away_team")["home_score"].transform("mean")

df["home_advantage"] = (df["home_team"] != df["away_team"]).astype(int)

df = df.dropna()

def get_result(r):
    if r["home_score"] > r["away_score"]:
        return "HomeWin"
    elif r["home_score"] < r["away_score"]:
        return "AwayWin"
    else: return "Draw"

df["result"] = df.apply(get_result, axis=1)

X = df[[
    "home_encoded", "away_encoded",
    "home_avg_scored", "home_avg_conceded",
    "away_avg_scored", "away_avg_conceded",
    "home_advantage"
]]

y = df["result"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, multi_class="multinomial", class_weight="balanced"),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200, class_weight="balanced_subsample"),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}")
    if name == "XGBoost":
        model.fit(X_train, le.transform(y_train))
        preds = model.predict(X_test)
        preds = le.inverse_transform(preds)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)


    acc = accuracy_score(y_test, preds)
    print(f"{name}: Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    if name == "XGBoost":
        cv_score = cross_val_score(model, X, y_encoded, cv=5, scoring="accuracy")
    else:
        cv_score = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print(f"f{name} - CV Accuracy: {np.mean(cv_score):.4f} + {np.std(cv_score):.4f}")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "CV Accuracy": np.mean(cv_score),
        "CV std": np.std(cv_score)
    })

print("\nSummary")
Summary_df = pd.DataFrame(results)
print(Summary_df)