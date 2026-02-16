# 🛒 Retail Demand Forecasting & Inventory Optimization

## 📖 프로젝트 개요 (Project Overview)
이 프로젝트는 소매 기업의 2년 치 과거 판매 데이터와 프로모션, 날씨, 경쟁사 가격 등 다양한 내·외부 요인을 분석하여 **미래 30일간의 수요를 예측(Demand Forecasting)**하는 딥러닝 모델을 개발한 프로젝트입니다.

단순한 판매량 추정이 아닌, 시계열 데이터의 장기 의존성과 복합적인 상호작용을 학습할 수 있는 **Temporal Fusion Transformer(TFT)** 모델을 도입했습니다. 이를 통해 단순한 점 예측(Point Prediction)이 아닌 **구간 예측(Quantile Prediction)**을 수행하여 수요의 불확실성을 정량화하고, 재고 최적화를 통한 비즈니스 비용 절감을 목표로 했습니다.

## 🎯 목표 (Objectives)
* **중·단기 수요 예측**: 향후 **30일(Short-to-Mid term)** 간의 일별 판매량을 시계열적으로 예측하여 재고 운영 계획 지원
* **예측 성능 확보**: MAPE(오차율) 10% 수준 달성 및 RMSE 기준 Baseline(단순 이동평균) 대비 **15% 이상 성능 개선**
* **불확실성 관리**: **Quantile Regression(분위수 회귀)**을 통해 95% 신뢰 구간을 제공하여 과학적인 안전 재고(Safety Stock) 산정 근거 마련

## 🛠️ 기술 스택 (Tech Stack)
* **Language**: Python 3.9+
* **Data Analysis**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn, TensorBoard
* **Deep Learning**: PyTorch, PyTorch Lightning
* **Time Series Modeling**: PyTorch Forecasting (TFT)
* **Optimization**: Optuna (Bayesian Optimization)

## 📂 파일 구성 (Project Structure)
* `retail-store-inventoty-and-demand-forecasting-using-TFT.ipynb`: 데이터 전처리, TimeSeriesDataSet 구축, TFT 모델 학습, Optuna 튜닝 및 결과 시각화 전 과정을 담은 주피터 노트북
* `sales_data`: 학습에 사용된 데이터셋
* `lightning_logs/`: 학습 로그 및 체크포인트 저장 디렉토리
* `saved_models/`: 최적화된 모델이 저장되는 디렉토리

## 🚀 주요 기능 및 과정 (Key Features)

### 1. 데이터 전처리 및 분석 (EDA)
* **Data Leakage 방지**: TFT 입력 구조에 맞춰 변수 유형 세분화 (Static, Known Inputs, Unknown Inputs)
* **파생 변수 생성**: 시계열 인덱스(`time_idx`), 경쟁사 대비 가격 비율(`Price_Ratio`), 가격 세그먼트(`Price_Segment`) 생성
* **데이터 변환**: 매장/제품별 스케일 조정을 위한 `GroupNormalizer` 및 이분산성 완화를 위한 로그 변환(`np.log1p`) 적용

### 2. 모델링 (Modeling)
* **Model**: Temporal Fusion Transformer (TFT)
* **Optimization**:
    * **Optuna**: 베이지안 최적화를 통해 `Hidden Size`, `Attention Heads`, `Learning Rate` 등 하이퍼파라미터 튜닝
    * **Loss Function**: `QuantileLoss`를 사용하여 예측 구간(Prediction Interval) 학습

### 3. 성과 및 해석 (Performance & Interpretation)
* **Performance Improvement**:
    * **RMSE**: [9.14] (Baseline 61.06 대비 **85.03% 성능 향상**)
    * **MAPE**: [10.15%] (목표치인 10%에 근접한 높은 정확도 달성)
* **Interpretability**:
    * **Static**: `Product ID`(35%)가 카테고리보다 높은 중요도를 가짐
    * **Encoder**: `Competitor Pricing` & `Price`가 과거 판매량보다 높은 중요도를 보이며, **가격 경쟁력**이 수요의 핵심 동인임을 규명

## 📖 링크
* Link: https://kunho192.tistory.com/15
