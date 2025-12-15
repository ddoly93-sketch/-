import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# GitHub 리포지토리에 올라가 있는 CSV 파일 이름과 동일해야 함
CSV_PATH = "전처리완료 data (수율파악)_승인완료.csv"


# -----------------------------
# 데이터 로딩
# -----------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """전처리 완료 CSV 데이터를 불러온다."""
    # UTF-8-SIG 인코딩으로 저장된 파일을 읽음
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    # 수율 결측치는 제거
    df = df.dropna(subset=["수율"])

    return df


# -----------------------------
# 모델 학습 (AdaBoost + 간단 튜닝)
# -----------------------------
@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    전처리 완료 데이터를 이용해 AdaBoost 회귀모델을 학습하고 성능을 반환.
    - 5-fold 교차검증으로 AdaBoost 하이퍼파라미터 튜닝
    - Train/Test 분할 후 Test R² / MAE 계산
    - 참고용으로 전체(in-sample) R²도 함께 계산
    """
    # Orange에서 Select Columns 후 사용한 특징 컬럼
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

    # 필요한 컬럼만 사용하고 결측치 제거
    data = df[feature_cols + [target_col]].dropna()

    X = data[feature_cols]
    y = data[target_col]

    # 범주형 / 수치형 컬럼 구분
    cat_cols = ["형태", "품종", "완제형태", "Billet 규격"]
    num_cols = ["생산중량", "제품길이", "제품두께", "Hole수"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # 간단한 하이퍼파라미터 후보 (AdaBoost 깊게 튜닝하면 느려지므로 적당한 수준만 탐색)
    param_grid = [
        {"n_estimators": 100, "learning_rate": 0.1},
        {"n_estimators": 200, "learning_rate": 0.1},
        {"n_estimators": 300, "learning_rate": 0.1},
        {"n_estimators": 300, "learning_rate": 0.05},
    ]

    best_score = -np.inf
    best_params = None

    # 5-fold 교차검증으로 R²가 가장 좋은 조합 선택
    for params in param_grid:
        ada = AdaBoostRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=42,
        )

        model = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("regressor", ada),
            ]
        )

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        mean_r2 = scores.mean()

        if mean_r2 > best_score:
            best_score = mean_r2
            best_params = params

    # 선택된 최적 파라미터로 최종 모델 구성
    ada_best = AdaBoostRegressor(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        random_state=42,
    )

    best_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("regressor", ada_best),
        ]
    )

    # Train/Test 분할 (대략 70:30, Orange Data Sampler 구조와 유사)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 학습
    best_model.fit(X_train, y_train)

    # Test 성능
    y_pred_test = best_model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # 전체(in-sample) 성능 (Orange Predictions 창의 R²와 비교용)
    y_pred_all = best_model.predict(X)
    r2_all = r2_score(y, y_pred_all)

    # 반환 값:
    # - best_model: 예측에 사용할 최종 모델
    # - r2_test, mae_test: 새로운 데이터에 대한 성능
    # - best_score: 5-fold CV 평균 R²
    # - r2_all: 전체 데이터 기준 R² (참고용)
    return best_model, r2_test, mae_test, best_score, r2_all


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="압출 수율 예측 시스템", layout="centered")

    st.title("압출 수율 예측 시스템 (AdaBoost 기반)")
    st.write("전처리 완료 데이터를 기반으로, 형태/조건을 입력하면 예상 수율을 예측합니다.")

    # 데이터 로딩 + 모델 학습
    with st.spinner("데이터를 불러오고 모델을 학습하는 중입니다... (최초 1회만 소요)"):
        df = load_data()
        model, r2_test, mae_test, r2_cv, r2_all = train_model(df)

    st.success(
        f"모델 학습 완료\n"
        f"- 5-fold CV R² = {r2_cv:.3f}\n"
        f"- Test R² = {r2_test:.3f}, MAE = {mae_test:.3f}\n"
        f"- 전체 데이터 R²(참고용) = {r2_all:.3f}"
    )

    st.markdown("---")
    st.subheader("입력 조건")

    # 입력 위젯용 옵션/기본값 설정
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

        # 참고용: 같은 형태/품종 조건의 실제 데이터 일부 표시
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
