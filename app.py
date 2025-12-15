import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
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
    df = df.dropna(subset=["수율"])
    return df


# -------------------------------------------------------
# 2. 모델 학습
#    - 생산중량은 사용하지 않고, 나머지 인자만 사용
#    - 여러 모델 중 교차검증 R²가 가장 좋은 모델 자동 선택
#    - 모델 이름은 UI에 노출하지 않음
# -------------------------------------------------------
@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    전처리 완료 데이터를 이용해 회귀 모델 학습.
    - 사용 피처: 형태, 제품길이, 제품두께, 품종, 완제형태, Billet 규격, Hole수
    - 수치형: 결측치 → 중앙값, 이후 StandardScaler
    - 범주형: 결측치 → 최빈값, 이후 One-Hot Encoding
    - 80:20 Train/Test 분할
    - Train 데이터에서 5-fold 교차검증으로 최적 모델 선택
    """

    # 피처/타깃 컬럼 정의 (생산중량 제외)
    feature_cols = [
        "형태",
        "제품길이",
        "제품두께",
        "품종",
        "완제형태",
        "Billet 규격",
        "Hole수",
    ]
    target_col = "수율"

    data = df[feature_cols + [target_col]].copy()
    data = data.dropna(subset=feature_cols + [target_col])

    X = data[feature_cols]
    y = data[target_col]

    # 범주형 / 수치형 구분
    cat_cols = ["형태", "품종", "완제형태", "Billet 규격"]
    num_cols = ["제품길이", "제품두께", "Hole수"]

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

    # 80:20 Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    # 후보 모델들 정의 (UI에는 이름 노출 안 함)
    candidate_models = {
        "gbr": GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            random_state=42,
        ),
        "extratrees": ExtraTreesRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        ),
    }

    best_name = None
    best_score = -np.inf
    best_estimator = None

    # Train 데이터에서 5-fold CV로 가장 좋은 모델 선택
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, reg in candidate_models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("regressor", reg),
            ]
        )
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_estimator = reg

    # 선택된 최적 모델로 최종 파이프라인 구성 후 학습
    best_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("regressor", best_estimator),
        ]
    )
    best_model.fit(X_train, y_train)

    # 성능 측정
    y_pred_test = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)

    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    # best_name은 내부 확인용으로만 사용 가능 (UI에는 안 씀)
    return best_model, r2_test, mae_test, r2_train


# -------------------------------------------------------
# 3. Streamlit UI
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="압출 수율 예측 시스템", layout="centered")

    st.title("압출 수율 예측 시스템 (머신러닝 기반)")
    st.write("생산중량을 제외한 공정 인자를 기반으로 예상 수율을 예측합니다.")

    # 데이터 로딩 + 모델 학습
    with st.spinner("데이터를 불러오고 모델을 학습하는 중입니다... (최초 1회만 수행)"):
        df = load_data()
        model, r2_test, mae_test, r2_train = train_model(df)

    # 모델명 노출 없이 성능만 표시
    st.success(
        f"모델 학습 완료\n"
        f"- Test R² (20% 홀드아웃) = {r2_test:.3f}, MAE = {mae_test:.3f}\n"
        f"- Train R² (80% 학습 데이터) = {r2_train:.3f}"
    )

    st.markdown("---")
    st.subheader("입력 조건")

    # 입력 옵션 생성 (실제 데이터 기반)
    형태_options = sorted(df["형태"].dropna().unique().tolist())
    품종_options = sorted(df["품종"].dropna().unique().tolist())
    완제형태_options = sorted(df["완제형태"].dropna().unique().tolist())
    billet_options = sorted(df["Billet 규격"].dropna().unique().tolist())

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
        # 입력값을 DataFrame으로 구성 (생산중량은 아예 사용하지 않음)
        input_df = pd.DataFrame(
            [{
                "형태": 형태,
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


if __name__ == "__main__":
    main()
