from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

app = Flask(__name__)

df = pd.read_csv("data/results.csv")

df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce",)
df = df[df["date"].notna()].copy()

start_date = pd.Timestamp("2022-11-20")  # Start of 2022 Wordl Cup
df = df[df["date"] >= start_date].copy()

with open("models/predictor.pkl", "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    teams = saved_data["teams"]
    team_to_idx = saved_data["team_to_idx"]
    label_encoder = saved_data.get("label_encoder")

def encode_team(team):
    try:
        return team_to_idx.get(team, -1)
    except ValueError:
        raise ValueError(f"Team '{team}' not in trained model's team list.")

def build_features(team1, team2):
    home_encoded = encode_team(team1)
    away_encoded = encode_team(team2)

    home_matches = df[(df["home_team"] == team1) | (df["away_team"] == team1)]
    away_matches = df[(df["home_team"] == team2) | (df["away_team"] == team2)]

    home_avg_scored = home_matches["home_score"].mean() if not home_matches.empty else 0
    home_avg_conceded = home_matches["away_score"].mean() if not home_matches.empty else 0

    away_avg_scored = away_matches["away_score"].mean() if not away_matches.empty else 0
    away_avg_conceded = away_matches["home_score"].mean() if not away_matches.empty else 0

    home_advantage = 1

    return pd.DataFrame([{
        "home_encoded": home_encoded,
        "away_encoded": away_encoded,
        "home_avg_scored": home_avg_scored,
        "home_avg_conceded": home_avg_conceded,
        "away_avg_scored": away_avg_scored,
        "away_avg_conceded": away_avg_conceded,
        "home_advantage": home_advantage,
        "home_form": 0,
        "away_form": 0
    }])

def teamStats(team1, team2):
    filtered = ((df["home_team"] == team1) & (df["away_team"] == team2)) | \
               ((df["home_team"] == team2) & (df["away_team"] == team1))
    matches = df[filtered]

    if matches.empty:
        team1_wins = team2_wins = draws = 0
    else:
        winners = np.where(matches["home_score"] > matches["away_score"], matches["home_team"],
                  np.where(matches["home_score"] < matches["away_score"], matches["away_team"], "Draw"))

        team1_wins = int((winners == team1).sum())
        team2_wins = int((winners == team2).sum())
        draws = int((winners == "Draw").sum())

    X_new = build_features(team1, team2)

    probs = model.predict_proba(X_new)[0]

    if label_encoder:
        labels = label_encoder.inverse_transform(np.arange(len(probs)))
    else:
        labels = model.classes_

    probs_dict = {labels[i]: probs[i] for i in range(len(labels))}
    team1_prob = probs_dict.get("HomeWin", 0)
    team2_prob = probs_dict.get("AwayWin", 0)
    draws_prob = probs_dict.get("Draw", 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].bar([team1, team2, "Draw"], [team1_wins, team2_wins, draws], color=["blue", "red", "gray"])
    axes[0].set_title(f"Home Team {team1} vs {team2}")
    axes[0].set_ylabel("Matches")

    axes[1].bar([team1, team2, "Draw"], [team1_prob, team2_prob, draws_prob], color=["blue", "red", "gray"])
    axes[1].set_title("Prediction Probabilities")
    axes[1].set_ylabel("Probability")

    plt.tight_layout()
    img_path = "static/result.png"
    plt.savefig(img_path)
    plt.close(fig)

    print("X_new:", X_new)
    print("Probs:", probs)

    return {
        "team1": team1,
        "team2": team2,
        "draws": draws,
        "team1_wins": team1_wins,
        "team2_wins": team2_wins,
        "team1_prob": round(team1_prob * 100, 1),
        "team2_prob": round(team2_prob * 100, 1),
        "draw_prob": round(draws_prob * 100, 1),
        "img_path": img_path
    }

@app.route("/", methods=["GET", "POST"])

def index():
    if request.method == "POST":
        team1 = request.form["team1"]
        team2 = request.form["team2"]
        stats = teamStats(team1, team2)
        return render_template("result.html", stats=stats)
    return render_template("index.html", teams=teams)

if __name__ == "__main__":
    app.run(debug=True)

