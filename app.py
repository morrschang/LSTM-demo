import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd

st.set_page_config(page_title="LSTM 模型展示", page_icon="📊", layout="centered")

st.title("📊 LSTM 模型互動展示網站")
st.caption("這是由 Kaggle Notebook 自動轉換的互動式展示網站。")

# =============================
# 🚀 自動載入或建立 LSTM 模型（帶錯誤防護）
# =============================
@st.cache_resource
def load_lstm_model_safe():
    try:
        model_dir = "model"
        model_path = os.path.join(model_dir, "lstm_model.h5")

        # --- 防呆修正：確保 model 是資料夾 ---
        if os.path.exists(model_dir) and not os.path.isdir(model_dir):
            os.remove(model_dir)
        os.makedirs(model_dir, exist_ok=True)

        # --- 載入或建立模型 ---
        if os.path.exists(model_path):
            model = load_model(model_path)
            st.success("✅ 模型已成功載入！")
        else:
            st.warning("⚠️ 找不到模型，正在建立新模型中...（約 3 秒）")

            model = Sequential([
                LSTM(32, input_shape=(10, 1), activation="tanh"),
                Dense(1)
            ])
            model.compile(optimizer=Adam(0.001), loss="mse")

            X = np.random.random((30, 10, 1))
            y = np.random.random((30, 1))
            model.fit(X, y, epochs=2, verbose=0)

            model.save(model_path)
            st.info("✅ 已自動建立並儲存新的模型於 model/lstm_model.h5")

        return model

    except Exception as e:
        st.error(f"❌ 模型載入錯誤：{str(e)}")
        return None


# =============================
# 🧩 主應用介面
# =============================
try:
    model = load_lstm_model_safe()
    if model is None:
        st.stop()

    st.divider()
    st.header("🔢 輸入資料進行預測")

    user_input = st.number_input("請輸入單一數值 (例如：0.5)", value=0.5)
    sequence_length = st.slider("時間步長 (Time Steps)", 1, 10, 10)

    X_input = np.full((1, sequence_length, 1), user_input)
    prediction = model.predict(X_input)
    st.success(f"📈 模型預測結果：{float(prediction[0][0]):.5f}")

    # =============================
    # 📂 測試資料上傳區
    # =============================
    st.divider()
    st.header("📂 使用測試資料進行展示")
    uploaded_file = st.file_uploader("上傳 CSV 檔（單一欄位）", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if data.shape[1] == 1:
            X_test = np.expand_dims(data.values, axis=2)
            pred = model.predict(X_test)
            st.write("✅ 預測結果：")
            st.dataframe(pd.DataFrame(pred, columns=["prediction"]))
        else:
            st.error("CSV 檔必須只有一個欄位！")

    st.info("🧩 模型檔會自動建立於 model/lstm_model.h5。")

except Exception as err:
    st.error(f"🚨 應用程式執行時發生錯誤：{str(err)}")
