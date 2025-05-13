import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

def extract_seed_number(seed_str):
    return int(re.sub("[^0-9]", "", seed_str))

def main():
    # 데이터 로드
    m_seeds = pd.read_csv("MNCAATourneySeeds.csv")
    m_tourn_compact = pd.read_csv("MNCAATourneyCompactResults.csv")
    m_regular_compact = pd.read_csv("MRegularSeasonCompactResults.csv")

    # 시드 처리
    m_seeds["SeedNum"] = m_seeds["Seed"].apply(extract_seed_number)

    # 시드와 승률의 상관관계 분석
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=m_seeds["SeedNum"], y=m_seeds.index, alpha=0.6)
    plt.xlabel("Seed Number")
    plt.ylabel("Index")
    plt.title("Seed Number Distribution")
    plt.show()

    # 정규 시즌 전체 승률 계산
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
    
    # 시드와 승률의 상관관계 분석
    seed_winrate_corr = pd.merge(m_seeds, reg_wl, on=["Season", "TeamID"], how="left")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=seed_winrate_corr["SeedNum"], y=seed_winrate_corr["win_rate"], alpha=0.6)
    plt.xlabel("Seed Number")
    plt.ylabel("Win Rate")
    plt.title("Seed Number vs. Win Rate")
    plt.show()

    # 최근 경기(예: 마지막 10경기) 승률 계산
    recent_games = m_regular_compact.sort_values(by=["Season", "DayNum"], ascending=[True, False])
    recent_wins = recent_games.groupby(["Season", "WTeamID"]).head(10).groupby(["Season", "WTeamID"]).size().reset_index(name="recent_win_count")
    recent_losses = recent_games.groupby(["Season", "LTeamID"]).head(10).groupby(["Season", "LTeamID"]).size().reset_index(name="recent_loss_count")
    
    recent_wl = pd.merge(recent_wins, recent_losses, how="outer",
                         left_on=["Season", "WTeamID"],
                         right_on=["Season", "LTeamID"])
    recent_wl["TeamID"] = recent_wl["WTeamID"].fillna(recent_wl["LTeamID"])
    recent_wl["recent_win_count"] = recent_wl["recent_win_count"].fillna(0)
    recent_wl["recent_loss_count"] = recent_wl["recent_loss_count"].fillna(0)
    recent_wl.drop(columns=["WTeamID", "LTeamID"], inplace=True)
    recent_wl["recent_total_games"] = recent_wl["recent_win_count"] + recent_wl["recent_loss_count"]
    recent_wl["recent_win_rate"] = recent_wl["recent_win_count"] / recent_wl["recent_total_games"]
    recent_wl["recent_win_rate"].fillna(0, inplace=True)

    # XGBoost 모델 학습을 위한 데이터 준비
    df = pd.merge(seed_winrate_corr, recent_wl, on=["Season", "TeamID"], how="left")
    df["RecentWinRateDiff"] = df["recent_win_rate"] - df["win_rate"]
    
    X = df[["SeedNum", "win_rate", "RecentWinRateDiff"]]
    y = (df["win_rate"] > 0.5).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results_df = pd.DataFrame({
        "Metric": ["Accuracy"],
        "Value": [accuracy]
    })
    
    print(results_df)

if __name__ == "__main__":
    main()
