import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def extract_seed_number(seed_str):
    return int(re.sub("[^0-9]", "", seed_str))

def main():
    # 데이터 로드
    m_seeds = pd.read_csv("MNCAATourneySeeds.csv")
    m_tourn_compact = pd.read_csv("MNCAATourneyCompactResults.csv")
    m_regular_compact = pd.read_csv("MRegularSeasonCompactResults.csv")

    # 시드 처리
    m_seeds["SeedNum"] = m_seeds["Seed"].apply(extract_seed_number)

    # 정규 시즌 승률 계산
    regular_wins = m_regular_compact.groupby(["Season", "WTeamID"]).size().reset_index(name="win_count")
    regular_losses = m_regular_compact.groupby(["Season", "LTeamID"]).size().reset_index(name="loss_count")
    
    reg_wl = pd.merge(regular_wins, regular_losses, how="outer",
                      left_on=["Season", "WTeamID"],
                      right_on=["Season", "LTeamID"])
    reg_wl["TeamID"] = reg_wl["WTeamID"].fillna(reg_wl["LTeamID"])
    reg_wl["win_count"] = reg_wl["win_count"].fillna(0)
    reg_wl["loss_count"] = reg_wl["loss_count"].fillna(0)
    reg_wl.drop(columns=["WTeamID", "LTeamID"], inplace=True)
    reg_wl["total_games"] = reg_wl["win_count"] + reg_wl["loss_count"]
    reg_wl["win_rate"] = reg_wl["win_count"] / reg_wl["total_games"]
    reg_wl["win_rate"].fillna(0, inplace=True)

    # 토너먼트 데이터와 정규 시즌 승률 병합
    df = pd.merge(m_tourn_compact, m_seeds[["Season", "TeamID", "SeedNum"]],
                  how="left", left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"])
    df.rename(columns={"SeedNum": "WSeed"}, inplace=True)
    df.drop(columns=["TeamID"], inplace=True)
    
    df = pd.merge(df, m_seeds[["Season", "TeamID", "SeedNum"]],
                  how="left", left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"])
    df.rename(columns={"SeedNum": "LSeed"}, inplace=True)
    df.drop(columns=["TeamID"], inplace=True)

    df = pd.merge(df, reg_wl[["Season", "TeamID", "win_rate"]],
                  how="left", left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"])
    df.rename(columns={"win_rate": "WTeamWinRate"}, inplace=True)
    df.drop(columns=["TeamID"], inplace=True)
    
    df = pd.merge(df, reg_wl[["Season", "TeamID", "win_rate"]],
                  how="left", left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"])
    df.rename(columns={"win_rate": "LTeamWinRate"}, inplace=True)
    df.drop(columns=["TeamID"], inplace=True)

    # 피처 엔지니어링
    df["SeedDiff"] = df["WSeed"] - df["LSeed"]
    df["WinRateDiff"] = df["WTeamWinRate"] - df["LTeamWinRate"]
    
    # 데이터셋 분할 (1985~2023 -> Train, 2024 -> Test)
    train_df = df[df["Season"] <= 2023].copy()
    test_df = df[df["Season"] == 2024].copy()

    # XGBoost 회귀 모델 학습
    features = ["SeedDiff", "WinRateDiff", "WSeed", "LSeed", "WTeamWinRate", "LTeamWinRate"]
    target = "WTeamWinRate"

    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values

    model = XGBRegressor(n_estimators=100, learning_rate=0.2, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"[INFO] MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # 실제 vs 예측 승률 시각화
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([0, 1], [0, 1], linestyle='--', color='red')
    plt.xlabel("Actual Win Rate")
    plt.ylabel("Predicted Win Rate")
    plt.title("Win Rate Prediction: Actual vs Predicted")
    plt.show()

if __name__ == "__main__":
    main()