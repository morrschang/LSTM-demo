import os
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import streamlit as st

# =============================
# 🚀 自動載入或建立 LSTM 模型
# =============================
def load_lstm_model():
    model_path = "model/lstm_model.h5"
    os.makedirs("model", exist_ok=True)

    # 若模型存在就直接載入
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("✅ 模型已載入：", model_path)
    else:
        print("⚠️ 找不到模型，正在建立新模型...")

        # 建立一個簡單的 LSTM 模型
        model = Sequential([
            LSTM(32, input_shape=(10, 1), activation="tanh"),
            Dense(1)
        ])
        model.compile(optimizer=Adam(0.001), loss="mse")

        # 用隨機資料訓練少量樣本（初始化權重）
        X = np.random.random((50, 10, 1))
        y = np.random.random((50, 1))
        model.fit(X, y, epochs=3, batch_size=8, verbose=0)

        # 儲存模型以供下次使用
        model.save(model_path)
        print("✅ 已建立並儲存新的模型於：", model_path)
    return model


# =============================
# 🧩 Streamlit 介面
# =============================
st.title("📊 LSTM 模型互動展示網站")
st.markdown("這是由 Kaggle Notebook 自動轉換的互動式展示網站。")

# 載入模型（自動建立或讀取）
model = load_lstm_model()

# 使用者輸入
st.header("🔢 輸入資料進行預測")
user_input = st.number_input("請輸入單一數值 (例如：0.5)", value=0.5)

# 模擬一個輸入時間序列
sequence_length = st.slider("時間步長 (Time Steps)", 1, 10, 10)
X_input = np.full((1, sequence_length, 1), user_input)

# 模型預測
prediction = model.predict(X_input)
st.success(f"✅ 模型預測輸出： {float(prediction[0][0]):.5f}")

# 測試資料上傳功能
st.header("📂 使用測試資料進行展示")
uploaded_file = st.file_uploader("上傳 CSV 檔（單一欄位）", type=["csv"])
if uploaded_file:
    import pandas as pd
    data = pd.read_csv(uploaded_file)
    if data.shape[1] == 1:
        X_test = np.expand_dims(data.values, axis=2)
        pred = model.predict(X_test)
        st.write("📈 預測結果：")
        st.write(pred.flatten())
    else:
        st.error("CSV 檔必須只包含一個欄位！")

st.info("🧩 模型檔會自動建立於 model/lstm_model.h5。")
