import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def main():
    try:
        data_path = "data/results.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found. Put your CSV in the data/ folder.")

        df = pd.read_csv(data_path)

        df["home_team"] = df["home_team"].astype(str).str.strip()
        df["away_team"] = df["away_team"].astype(str).str.strip()
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
        df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])

        teams = sorted(set(df["home_team"]).union(set(df["away_team"])))
        team_to_idx = {team: i for i, team in enumerate(teams)}
        df["home_encoded"] = df["home_team"].map(team_to_idx)
        df["away_encoded"] = df["away_team"].map(team_to_idx)

        def get_result(r):
            if r["home_score"] > r["away_score"]:
                return "HomeWin"
            elif r["home_score"] < r["away_score"]:
                return "AwayWin"
            else:
                return "Draw"

        df["result"] = df.apply(get_result, axis=1)

        df = df.sort_values("date")

        df["home_avg_scored"] = (
            df.groupby("home_team")["home_score"].transform(lambda x: x.shift().rolling(10, min_periods=1).mean())
        )
        df["home_avg_conceded"] = (
            df.groupby("home_team")["away_score"].transform(lambda x: x.shift().rolling(10, min_periods=1).mean())
        )

        df["away_avg_scored"] = (
            df.groupby("away_team")["away_score"].transform(lambda x: x.shift().rolling(10, min_periods=1).mean())
        )
        df["away_avg_conceded"] = (
            df.groupby("away_team")["home_score"].transform(lambda x: x.shift().rolling(10, min_periods=1).mean())
        )
        df["home_advantage"] = df["neutral"].apply(lambda x: 0 if x else 1)

        df["home_is_win"] = (df["result"] == "HomeWin").astype(int)
        df["away_is_win"] = (df["result"] == "AwayWin").astype(int)

        df["home_form"] = (df.groupby("home_team")["home_is_win"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()))
        df["away_form"] = (df.groupby("away_team")["away_is_win"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()))


        X = df[["home_encoded", "away_encoded",
                "home_avg_scored", "home_avg_conceded",
                "away_avg_scored", "away_avg_conceded",
                "home_advantage", "home_form", "away_form"]].fillna(0)

        y = df["result"].copy()

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        print("\nX dtypes:")
        print(X.dtypes)
        print("\nSample X row:")
        print(X.iloc[0])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
        )
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        print("Model trained. Test accuracy:", round(acc, 4))

        os.makedirs("models", exist_ok=True)
        out_path = os.path.abspath("models/predictor.pkl")
        with open(out_path, "wb") as f:
            pickle.dump({"model": model, "teams": teams, "team_to_idx": team_to_idx, "label_encoder": le}, f)


    except Exception as exc:
        print("=== TRAINING ERROR ===")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
