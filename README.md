# 🏀 NCAA Tournament Seed & Win Rate Prediction Project

NCAA 남자 농구 토너먼트 데이터를 활용하여 시드, 승률, 최근 경기 승률 등을 기반으로  
XGBoost 모델을 통해 팀 승률 예측과 시드 간 승률 차이를 분석하는 프로젝트입니다.

---

## 📂 프로젝트 구성

| 구분       | 주요 기능                                              | 스크립트              |
|----------|-----------------------------------------------------|---------------------|
| 📈 승률 예측    | 팀의 시드, 전체 승률, 최근 승률 차이를 활용한 승리 예측 (이진 분류)          | `final.py`          |
| 📊 승률 회귀 예측 | 시드 차이, 승률 차이를 활용한 경기 승률 예측 (회귀)                        | `final22.py`        |

---

## 🔧 사용 기술

- Python 3.10
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- XGBoost

---

## ▶ 실행 방법

### 1. 승률 이진 분류 (XGBoost Classifier)
```bash
python final.py
주요 Feature

Seed Number

전체 시즌 승률 (win_rate)

최근 승률과 전체 승률 차이 (RecentWinRateDiff)

평가 지표: Accuracy

2. 승률 회귀 예측 (XGBoost Regressor)
bash
복사
편집
python final22.py
주요 Feature

SeedDiff (WSeed - LSeed)

WinRateDiff (WTeamWinRate - LTeamWinRate)

각 팀의 시드와 승률

평가 지표: MSE, R²

결과: 실제 vs 예측 승률 시각화

📊 분석 흐름
시드 처리

문자열 시드 -> 숫자 시드 변환

정규 시즌 승률 계산

전체 시즌, 최근 10경기 승률

XGBoost 모델 학습

Classifier (final.py) → 승리 여부 분류

Regressor (final22.py) → 승률 예측

시각화

Seed vs Win Rate 관계 시각화

승률 예측 Scatter Plot

💡 주의 사항
사용 데이터셋: Kaggle NCAA Tournament 데이터 (MNCAATourneySeeds.csv, MNCAATourneyCompactResults.csv, MRegularSeasonCompactResults.csv)

데이터 파일 동일 경로 필요

최신 데이터에 대한 예측 정확도는 데이터의 최신성, 팀 변화에 따라 달라질 수 있음
