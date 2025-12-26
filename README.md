# 📉 Telco Customer Churn Prediction & Monitoring Pipeline

## 📖 프로젝트 개요 (Project Overview)
이 프로젝트는 통신사 고객 데이터를 분석하여 **고객 이탈(Churn)을 예측**하는 머신러닝 모델을 개발하고, 실제 운영 환경을 가정하여 **데이터 드리프트(Data Drift)를 모니터링하고 자동으로 모델을 재학습**시키는 파이프라인을 구축한 프로젝트입니다.

단순한 예측 모델링을 넘어, 시간의 흐름에 따라 데이터 분포가 변할 때 모델의 성능을 유지하기 위한 **MLOps(Machine Learning Operations)** 관점의 접근을 시도했습니다.

## 🎯 목표 (Objectives)
* **이탈 예측**: 고객의 인구통계학적 정보와 서비스 가입 정보를 바탕으로 이탈 여부 분류
* **모델 최적화**: XGBoost 알고리즘을 사용하여 예측 성능(특히 Recall, F1-score) 최적화
* **지속적 모니터링**: `Evidently AI`를 활용하여 데이터 분포 변화(Drift) 감지 및 성능 저하 시 자동 재학습 로직 구현

## 🛠️ 기술 스택 (Tech Stack)
* **Language**: Python 3.x
* **Data Analysis**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Machine Learning**: Scikit-learn, XGBoost
* **MLOps & Monitoring**: Evidently AI, Joblib

## 📂 파일 구성 (Project Structure)
* `Telco-Customer-Churn.ipynb`: 데이터 전처리(EDA), 파이프라인 구축, 모델 학습 및 평가 과정을 담은 주피터 노트북
* `monitor.py`: 운영 환경을 시뮬레이션하여 데이터 드리프트를 감지하고, 필요 시 모델을 재학습시키는 자동화 스크립트
* `WA_Fn-UseC_-Telco-Customer-Churn.csv`: 학습에 사용된 데이터셋 (Kaggle)
* `models/`: 학습된 모델(`.pkl`)이 저장되는 디렉토리
* `reports/`: 데이터 드리프트에 대한 report가 저장되는 디렉토리

## 🚀 주요 기능 및 과정 (Key Features)

### 1. 데이터 전처리 및 분석 (EDA)
* 결측치 처리 및 `TotalCharges` 수치형 변환
* 범주형 변수(One-Hot Encoding)와 수치형 변수(Standard Scaling) 처리를 위한 `Sklearn Pipeline` 구축
* 불균형 데이터 처리를 고려한 학습

### 2. 모델링 (Modeling)
* **Model**: XGBoost Classifier
* **Evaluation Metric**:
    * **F1-Score**: [0.63]
    * **Recall**: [0.75] (이탈 고객을 놓치지 않는 것이 중요하므로 Recall을 중요 지표로 선정)

### 3. 모니터링 및 재학습 (Monitoring & Retraining)
* **Drift Detection**: `monitor.py`를 실행하면 `Evidently` 라이브러리가 기준 데이터(Reference)와 현재 데이터(Current) 간의 분포 차이를 분석합니다. 이를 `reports/` 폴더에 타임스탬프와 함께 저장합니다.
* **Auto-Retraining**: 데이터 드리프트가 감지되면 자동으로 파이프라인이 전체 데이터에 대해 모델을 재학습하고, 새로운 모델을 `models/` 폴더에 타임스탬프와 함께 저장합니다.
