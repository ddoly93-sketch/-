import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# CSV 파일명 (그대로 사용)
CSV_PATH = "전처리완료 data (수율파악)_승인완료.csv"


# -------------------------------------------------------
# 1. 데이터 로딩
# -------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """전처리 완료 CSV 로드."""
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    # 타깃 결측치는 제거
    df = df.dropna(subset=["수율"])

    return df


# -------------------------------------------------------
# 2. 모델 학습 (Gradient Boosting 기반 성능 최적화 버전)
# -------------------------------------------------------
@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    전처리 완료 데이터를 이용해 GradientBoosting 회귀 모델 학습.
    - 수치형: 결측치 → 중앙값, 이후 StandardScaler
    - 범주형: 결측치 → 최빈값, 이후 One-Hot Encoding
    - 80:20 Train/Test 분할
    - 5-fold 교차검증 R², Train/Test R², MAE 계산
    """

    # 피처/타깃 컬럼 정의
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

    data = df[feature_cols + [target_col]].copy()
    # (추가적인 결측치 처리) — 전체에서 NaN 제거
    data = data.dropna(subset=feature_cols + [target_col])

    X = data[feature_cols]
    y = data[target_col]

    # 범주형 / 수치형 구분
    cat_cols = ["형태", "품종", "완제형태", "Billet 규격"]
    num_cols = ["생산중량", "제품길이", "제품두께", "Hole수"]

    # 수치형 전처리: 중앙값 대체 + 표준화
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 범주형 전처리: 최빈값 대체 + 원-핫 인코딩
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

    # Gradient Boosting 회귀 모델 (AdaBoost보다 일반적으로 안정적)
    gbr = GradientBoostingRegressor(
        n_estimators=400,      # 트리 개수
        learning_rate=0.05,   # 학습률
        max_depth=3,          # 개별 트리 깊이
        subsample=0.9,        # Stochastic Gradient Boosting
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("regressor", gbr),
        ]
    )

    # 80:20 Train/Test 분할 (Orange Data Sampler 비율과 동일)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    # 학습
    model.fit(X_train, y_train)

    # Test 성능
    y_pred_test = model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Train 성능
    y_pred_train = model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)

    # 5-fold 교차검증 (전체 데이터 기준)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    r2_cv_mean = float(cv_scores.mean())
    r2_cv_std = float(cv_scores.std())

    return model, r2_test, mae_test, r2_train, r2_cv_mean, r2_cv_std


# -------------------------------------------------------
# 3. Streamlit UI
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="압출 수율 예측 시스템", layout="centered")

    st.title("압출 수율 예측 시스템 (Gradient Boosting 기반)")
    st.write("전처리 완료 데이터를 기반으로, 공정 조건을 입력하면 예상 수율을 예측합니다.")

    # 데이터 로딩 + 모델 학습
    with st.spinner("데이터를 불러오고 모델을 학습하는 중입니다... (최초 1회만 수행)"):
        df = load_data()
        model, r2_test, mae_test, r2_train, r2_cv_mean, r2_cv_std = train_model(df)

    st.success(
        f"모델 학습 완료\n"
        f"- Test R² (20% 홀드아웃) = {r2_test:.3f}, MAE = {mae_test:.3f}\n"
        f"- Train R² (80% 학습 데이터) = {r2_train:.3f}\n"
        f"- 5-fold CV R² 평균 = {r2_cv_mean:.3f} (±{r2_cv_std:.3f})"
    )

    st.markdown("---")
    st.subheader("입력 조건")

    # 입력 옵션 생성 (실제 데이터 기반)
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
        # 입력값을 DataFrame으로 구성
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

        # 예측
        pred = model.predict(input_df)[0]
        yield_percent = float(pred) * 100.0

        st.subheader("예측 결과")
        st.write(f"예상 수율: **{yield_percent:.2f}%**")

        # 참고용: 유사 조건 실제 데이터 표시
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
