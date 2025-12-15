import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# Orange에서 저장한 전처리 완료 CSV (파일명 그대로 사용)
CSV_PATH = "전처리완료 data (수율파악)_승인완료.csv"


# -----------------------------
# 1. 데이터 로딩
# -----------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """전처리 완료 CSV 데이터를 불러온다."""
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    # Orange도 target이 NaN인 행은 학습에 못 쓰므로 제거
    df = df.dropna(subset=["수율"])

    return df


# -----------------------------
# 2. Orange 방식에 최대한 맞춘 모델 학습
# -----------------------------
@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Orange Data Sampler(80:20) + AdaBoost 설정을 최대한 비슷하게 구현.
    - 80%: 학습 데이터 (Data Sample)
    - 20%: 테스트 데이터 (Remaining Data)
    - 수치형: 결측치 보정 + 0~1 정규화
    - 범주형: 결측치 보정 + One-Hot Encoding
    """

    # Orange에서 Select Columns 후 사용하는 피처
    feature_cols = [
        "형태",
        "생산중량",
        "제품길이",
        "제품두께",
        "품종",
        "완제형태",
        "Billet 규격",
        "Hole수",
    ]
    target_col = "수율"

    # 필요한 컬럼만 사용 (결측치는 아래 Imputer에서 처리)
    data = df[feature_cols + [target_col]]

    X = data[feature_cols]
    y = data[target_col]

    # 범주형 / 수치형 컬럼 구분 (Orange의 Discrete vs Continuous)
    cat_cols = ["형태", "품종", "완제형태", "Billet 규격"]
    num_cols = ["생산중량", "제품길이", "제품두께", "Hole수"]

    # 숫자형: 결측치는 중앙값으로, 이후 0~1 정규화 (Orange 기본 Normalize 흉내)
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    # 범주형: 결측치는 최빈값으로, 이후 One-Hot Encoding (Orange Continuize + Impute)
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, cat_cols),
            ("num", num_transformer, num_cols),
        ]
    )

    # Orange AdaBoost 기본값과 비슷한 설정
    # (DecisionTreeRegressor(max_depth=3)가 기본 base estimator)
    ada = AdaBoostRegressor(
        n_estimators=50,     # Orange 기본 50에 근사
        learning_rate=1.0,   # 기본값
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("regressor", ada),
        ]
    )

    # ★ Orange Data Sampler 80:20 재현 (shuffle + seed 고정)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,      # 20% Remaining Data
        shuffle=True,
        random_state=42,    # Data Sampler Seed와 맞추는 용도
    )

    # 학습 (Orange에서 Data Sample → AdaBoost 학습)
    model.fit(X_train, y_train)

    # Remaining Data에 대한 성능 (Orange Predictions(1)과 비슷한 개념)
    y_pred_test = model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # 전체 데이터(in-sample) 기준 R² (Orange에서 같은 데이터 예측할 때 값과 비슷)
    y_pred_all = model.predict(X)
    r2_all = r2_score(y, y_pred_all)

    # 학습 데이터 기준 R² (참고용)
    y_pred_train = model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)

    return model, r2_test, mae_test, r2_all, r2_train


# -----------------------------
# 3. Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="압출 수율 예측 시스템", layout="centered")

    st.title("압출 수율 예측 시스템 (AdaBoost / Orange 방식 근사)")
    st.write("전처리 완료 데이터를 기반으로, Orange Data Sampler(80:20)와 유사한 방식으로 학습한 모델입니다.")

    # 데이터 로딩 + 모델 학습
    with st.spinner("데이터를 불러오고 모델을 학습하는 중입니다... (최초 1회만 소요)"):
        df = load_data()
        model, r2_test, mae_test, r2_all, r2_train = train_model(df)

    st.success(
        f"모델 학습 완료\n"
        f"- Test R² (Remaining 20%) = {r2_test:.3f}, MAE = {mae_test:.3f}\n"
        f"- Train R² (Sample 80%) = {r2_train:.3f}\n"
        f"- 전체 데이터 R² (참고용, Orange Predictions와 유사) = {r2_all:.3f}"
    )

    st.markdown("---")
    st.subheader("입력 조건")

    # 입력 옵션 준비
    형태_options = sorted(df["형태"].dropna().unique().tolist())
    품종_options = sorted(df["품종"].dropna().unique().tolist())
    완제형태_options = sorted(df["완제형태"].dropna().unique().tolist())
    billet_options = sorted(df["Billet 규격"].dropna().unique().tolist())

    생산중량_default = float(df["생산중량"].median())
    제품길이_default = float(df["제품길이"].median())
    제품두께_default = float(df["제품두께"].median())
    hole_default = int(df["Hole수"].median())

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            형태 = st.selectbox("형태", 형태_options)
            품종 = st.selectbox("품종", 품종_options)
            완제형태 = st.selectbox("완제형태", 완제형태_options)
            Billet규격 = st.selectbox("Billet 규격", billet_options)

        with col2:
            생산중량 = st.number_input(
                "생산중량 (kg)", min_value=0.0, value=생산중량_default, step=10.0
            )
            제품길이 = st.number_input(
                "제품길이 (mm)", min_value=0.0, value=제품길이_default, step=100.0
            )
            제품두께 = st.number_input(
                "제품두께 (mm)", min_value=0.0, value=제품두께_default, step=0.01
            )
            Hole수 = st.number_input(
                "Hole 수", min_value=1, value=hole_default, step=1
            )

        submitted = st.form_submit_button("수율 예측하기")

    if submitted:
        input_df = pd.DataFrame(
            [{
                "형태": 형태,
                "생산중량": 생산중량,
                "제품길이": 제품길이,
                "제품두께": 제품두께,
                "품종": 품종,
                "완제형태": 완제형태,
                "Billet 규격": Billet규격,
                "Hole수": Hole수,
            }]
        )

        pred = model.predict(input_df)[0]
        yield_percent = float(pred) * 100.0

        st.subheader("예측 결과")
        st.write(f"예상 수율: **{yield_percent:.2f}%**")

        # 참고용: 같은 형태/품종 데이터 일부 보여주기
        st.markdown("#### 참고: 유사 조건 실제 데이터 (상위 10개)")
        mask = (df["형태"] == 형태) & (df["품종"] == 품종)
        similar = df[mask].copy()
        if similar.empty:
            st.write("같은 형태/품종 데이터가 없습니다. 전체 데이터 중 상위 10개를 표시합니다.")
            st.dataframe(df.head(10))
        else:
            st.dataframe(similar.head(10))


if __name__ == "__main__":
    main()
