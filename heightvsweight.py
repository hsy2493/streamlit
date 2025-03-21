#!/usr/bin/env python
# coding: utf-8

# Height and Weight

# 라이브러리 
import numpy as np  # type: ignore # 계산
import pandas as pd  # 배열
import matplotlib.pyplot as plt  # 그래프 시각화
import seaborn as sns  # 데이터 시각화
from sklearn.model_selection import train_test_split  # 데이터 분할 (훈련용/테스트용)
from sklearn.linear_model import LinearRegression  # 선형 회귀 모델
from sklearn.ensemble import RandomForestRegressor  # 랜덤 포레스트 모델
from sklearn.metrics import mean_squared_error  # 모델 성과지표
import streamlit as st  # UI

# In[20]:

# streamlit 
@st.cache_data
# @st.cache_data : 캐시로 저장된 값을 화면에 출력함

# streamlit UI 함수
def load_data():
    
    # 데이터 로드
    df = pd.read_csv('dataset/weight-height.csv')  # csv 파일 불러오기
    
    # 전처리 
    df = df.dropna()  # 결측치 완전 제거
    
    # 성별 변환 (Male, 남성 : 0, Female, 여성 : 1)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})  # 성별 변환
    
    return df  # 데이터 반환

# 캐시 값으로 UI에 데이터 출력
df = load_data()

# In[26]:

# 폰트 지정 (한글 깨짐 방지)
plt.rcParams['font.family'] = 'AppleGothic' # 한글 깨짐 방지 (mac os 전용)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 📌 Streamlit UI 시작
st.title("🔍 적정 몸무게 예측")
st.markdown("### 📊 입력된 데이터의 키 (cm)와 성별을 기반으로 적정 몸무게 (kg)를 예측합니다.")

# 2️⃣ **샘플 데이터 표시**
st.write("### 📋 1. 샘플 데이터")
st.dataframe(df.head())  # 표 스타일 개선


# 3️⃣ **주요 변수별 분포 (히스토그램)**
st.write("### 📈 2. 주요 변수별 분포")

# 그래프 그리기
fig, sub = plt.subplots(1, 3, figsize=(20, 5))

# 1) 성별 (Gender)
sub[0].hist(df['Gender'], bins=2, color='lightgreen', edgecolor='black')
sub[0].set_title('성별 (Gender) 분포')
sub[0].set_xlabel('성별 (0: 남성, 1: 여성)')
sub[0].set_ylabel('빈도수')

# 2) 키 (Height)
sub[1].hist(df['Height'], bins=20, color='skyblue', edgecolor='black')
sub[1].set_title('키 (Height) 분포')
sub[1].set_xlabel('신장 (cm)')
sub[1].set_ylabel('빈도수')

# 3) 몸무게 (Weight)
sub[2].hist(df['Weight'], bins=20, color='salmon', edgecolor='black')
sub[2].set_title('몸무게 (Weight) 분포')
sub[2].set_xlabel('몸무게 (kg)')
sub[2].set_ylabel('빈도수')

st.pyplot(fig)  # 그래프 출력


# 4️⃣ **모델 예측값 시각화**
st.write("### 🤖 3. 머신러닝 모델 학습")

# 독립변수(X)와 종속변수(y) 분리
X = df[['Height', 'Gender']]  # 원인(X) : 신장, 성별
y = df['Weight']  # 결과(y) : 몸무게

# In[27]:

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#  훈련용 80%, 테스트용 20%

# 📌 **선형 회귀 모델 학습**
linear_model = LinearRegression()  # 선형회귀 모델 생성
linear_model.fit(X_train, y_train)  # 훈련용 데이터로 학습

# 데이터 예측
y_pred_linear = linear_model.predict(X_test)  # 학습한 모델로 테스트 데이터 예측

# 성능 평가 (MSE)
linear_mse = mean_squared_error(y_test, y_pred_linear)  # 평균제곱오차 (MSE)


# 📌 **랜덤 포레스트 모델 학습**
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # 랜덤 포레스트 모델 생성
rf_model.fit(X_train, y_train)  # 훈련용 데이터로 학습

# 테스트용 데이터로 예측
y_pred_rf = rf_model.predict(X_test)

# 성능 평가 (MSE)
rf_mse = mean_squared_error(y_test, y_pred_rf)  # 랜덤 포레스트 모델


# 📊 **예측값 vs 실제값 비교 그래프**
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 선형 회귀 모델 비교
axes[0].scatter(y_test, y_pred_linear, color='blue', alpha=0.5)
axes[0].set_title('Linear Regression(선형 회귀): 실제값 vs 예측값')
axes[0].set_xlabel('실제값')
axes[0].set_ylabel('예측값')

# 랜덤 포레스트 모델 비교
axes[1].scatter(y_test, y_pred_rf, color='red', alpha=0.5)
axes[1].set_title('Random Forest(랜덤 포레스트): 실제값 vs 예측값')
axes[1].set_xlabel('실제값')
axes[1].set_ylabel('예측값')

st.pyplot(fig)  # 그래프 출력

# ✅ MSE 결과 출력
st.info(f"📉 **선형 회귀 MSE:** {linear_mse:.2f}")
st.info(f"🌳 **랜덤 포레스트 MSE:** {rf_mse:.2f}")


# 5️⃣ **사용자 입력을 통한 몸무게 예측**
st.write("### 🎯 4. 사용자 정보 입력")

# 📌 입력 UI 정렬
col1, col2 = st.columns(2)

with col1:
    height = st.number_input("📏 키 (cm)", min_value=100, max_value=250, value=170)

with col2:
    gender = st.radio("🚻 성별 선택", ["남", "여"])  # 0: 남, 1: 여
    gender = 0 if gender == "남" else 1  # 성별 숫자로 변환

# 📌 버튼을 눌러 예측 실행
if st.button("🔍 몸무게 예측하기"):
    input_data = np.array([[height, gender]])  # 입력 데이터 배열

    # 🔵 선형 회귀 모델 예측
    weight_pred_linear = linear_model.predict(input_data)[0]

    # 🟢 랜덤 포레스트 모델 예측
    weight_pred_rf = rf_model.predict(input_data)[0]

    # ✅ 예측 결과 출력
    st.success(f"📌 **적정 몸무게 (kg): :** {weight_pred_linear:.2f} kg")
