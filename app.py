
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="LSTM 模型展示", layout="wide")
st.title("🧠 LSTM 時間序列預測展示平台")
st.markdown("這是由 Kaggle Notebook 自動轉換而成的互動式展示網站。您可以輸入數值或使用測試資料來觀察模型的預測結果。")

@st.cache_resource
def load_lstm_model():
    try:
        model = load_model("model/lstm_model.h5")
        return model
    except Exception as e:
        st.error("❌ 無法載入模型，請確認 model/lstm_model.h5 是否存在。")
        st.exception(e)
        return None

model = load_lstm_model()

st.header("🔢 輸入資料進行預測")
col1, col2 = st.columns(2)

with col1:
    input_value = st.number_input("請輸入單一數值 (例如：0.5)", value=0.0, format="%.4f")

with col2:
    num_steps = st.slider("時間步長 (Time Steps)", 1, 10, 3)

if st.button("🚀 開始預測"):
    if model:
        x_input = np.array([[[input_value]] * num_steps])
        y_pred = model.predict(x_input)
        pred_value = y_pred[0][0]

        st.success(f"✅ 模型預測結果：**{pred_value:.4f}**")

        st.subheader("📈 模擬預測曲線")
        plt.figure(figsize=(8, 4))
        plt.plot(range(num_steps), [input_value]*num_steps, label="Input Sequence")
        plt.plot(range(num_steps, num_steps+1), [pred_value], "ro-", label="Predicted Next Value")
        plt.legend()
        st.pyplot(plt)

st.header("📂 使用測試資料進行展示")
uploaded_file = st.file_uploader("上傳測試資料（CSV 格式，單一欄位）", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("✅ 成功載入資料。前幾筆資料：")
    st.dataframe(data.head())

    if st.button("📊 進行批量預測"):
        values = data.iloc[:, 0].values
        preds = []
        for v in values:
            x_input = np.array([[[v]] * num_steps])
            y_pred = model.predict(x_input)
            preds.append(y_pred[0][0])

        result_df = pd.DataFrame({"Input": values, "Predicted": preds})
        st.line_chart(result_df)
        st.write("📈 批量預測結果：")
        st.dataframe(result_df.head())

st.markdown("---\n🧩 **說明：**\n- 模型檔請放於 `model/lstm_model.h5`\n- 可在 `data/` 放置範例 CSV 測試檔\n- 修改程式可調整輸入維度或顯示更多資訊\n---")
