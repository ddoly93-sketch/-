import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# CSV 파일명 (DRM 때문에 파일명 그대로 사용)
CSV_PATH = "전처리완료 data (수율파악)_승인완료.csv"


# -------------------------------------------------------
# 1. 데이터 로딩
# -------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    # 타깃 결측치는 제거
    df = df.dropna(subset=["수율"])
    return df


# -------------------------------------------------------
# 2. Gradient Boosting 모델 학습
#    (모델 이름은 화면에 노출하지 않음)
# -------------------------------------------------------
@st.cache_resource
def train_model(df: pd.DataFrame):
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

    data = df[feature_cols + [target_col]].dropna()

    X = data[feature_cols]
    y = data[target_col]

    # 범주형 / 수치형 구분
    cat_cols = ["형태", "품종", "완제형태", "Billet 규격"]
    num_cols = ["생산중량", "제품길이", "제품두께", "Hole수"]

    # 전처리 정의
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    # 내부적으로 사용할 Gradient Boosting 모델
    model_core = GradientBoostingRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("regressor", model_core),
        ]
    )

    # 80:20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    # 학습
    model.fit(X_train, y_train)

    # 성능 측정
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    return model, r2_test, mae_test, r2_train


# -------------------------------------------------------
# 3. Streamlit UI
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="압출 수율 예측 시스템", layout="centered")
    st.title("압출 수율 예측 시스템 (머신러닝 기반)")
    st.write("전처리 완료 데이터를 기반으로, 입력 조건에 따라 예상 수율을 예측합니다.")

    # 모델 준비
    with st.spinner("모델 학습 중... (최초 1회)"):
        df = load_data()
        model, r2_test, mae_test, r2_train = train_model(df)

    # 성능 표시 (모델명 숨김)
    st.success(
        f"모델 학습 완료\n"
        f"- Test R² = {r2_test:.3f}, MAE = {mae_test:.3f}\n"
        f"- Train R² = {r2_train:.3f}"
    )

    st.markdown("---")
    st.subheader("입력 조건")

    # 옵션 값 자동 추출
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
                "생산중량 (kg)",
                min_value=0.0,
                value=생산중량_default,
                step=10.0,
            )
            제품길이 = st.number_input(
                "제품길이 (mm)",
                min_value=0.0,
                value=제품길이_default,
                step=100.0,
            )
            제품두께 = st.number_input(
                "제품두께 (mm)",
                min_value=0.0,
                value=제품두께_default,
                step=0.01,
            )
            Hole수 = st.number_input(
                "Hole 수",
                min_value=1,
                value=hole_default,
                step=1,
            )

        submit = st.form_submit_button("수율 예측하기")

    if submit:
        input_data = pd.DataFrame(
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

        # 예측 (수율을 0~1로 학습했다고 가정하고 %로 변환)
        pred = model.predict(input_data)[0] * 100.0

        st.subheader("예측 결과")
        st.write(f"예상 수율: **{pred:.2f}%**")


if __name__ == "__main__":
    main()
