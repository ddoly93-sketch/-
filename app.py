import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

CSV_PATH = "전처리완료 data (수율파악)_승인완료.csv"

@st.cache_resource
def load_model():
    # 1) 데이터 로드
    df = pd.read_csv(CSV_PATH, encoding="cp949")

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

    # 2) 전처리
    cat_cols = ["형태", "품종", "완제형태", "Billet 규격"]
    num_cols = ["생산중량", "제품길이", "제품두께", "Hole수"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # 3) AdaBoost 회귀 모델
    ada = AdaBoostRegressor(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("regressor", ada),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, r2, mae


# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="압출 수율 예측 시스템", layout="centered")

st.title("압출 수율 예측 시스템 (AdaBoost 기반)")
st.write("형태, 생산조건, 규격을 입력하면 예상 수율을 예측합니다.")

with st.spinner("모델 학습 중입니다. 처음 한 번만 시간이 조금 걸립니다..."):
    model, r2, mae = load_model()

st.success(f"모델 로딩 완료 (R²={r2:.3f}, MAE={mae:.3f})")

st.markdown("---")
st.subheader("입력 정보")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        형태 = st.selectbox("형태", ["환봉", "육각", "Coil", "기타"])
        품종 = st.text_input("품종 (예: 3772, 3604 등)", value="3772")
        완제형태 = st.selectbox("완제형태", ["ROD", "Coil", "기타"])
        Billet규격 = st.text_input("Billet 규격 (예: 1200, 1600 등)", value="1600")

    with col2:
        생산중량 = st.number_input("생산중량 (kg)", min_value=0.0, step=10.0, value=3000.0)
        제품길이 = st.number_input("제품길이 (mm)", min_value=0.0, step=100.0, value=3000.0)
        제품두께 = st.number_input("제품두께 (mm)", min_value=0.0, step=0.01, value=16.0)
        Hole수 = st.number_input("Hole 수", min_value=1, step=1, value=1)

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
    yield_percent = float(pred) * 100.0  # 0~1 스케일을 %로 변환

    st.subheader("예측 결과")
    st.write(f"예상 수율: **{yield_percent:.2f} %**")
