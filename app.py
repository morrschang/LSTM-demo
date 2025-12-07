def load_lstm_model():
    model_path = "model/lstm_model.h5"

    # --- 🧱 防呆修正：確保 'model' 一定是資料夾 ---
    if os.path.exists("model") and not os.path.isdir("model"):
        print("⚠️ 偵測到 'model' 是檔案，將刪除並改為資料夾...")
        os.remove("model")
    os.makedirs("model", exist_ok=True)
    # -------------------------------------------------

    if os.path.exists(model_path):
        model = load_model(model_path)
        print("✅ 模型已載入：", model_path)
    else:
        print("⚠️ 找不到模型，正在建立新模型...")

        model = Sequential([
            LSTM(32, input_shape=(10, 1), activation="tanh"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")

        # 用隨機資料初始化
        X = np.random.random((30, 10, 1))
        y = np.random.random((30, 1))
        model.fit(X, y, epochs=3, verbose=0)

        model.save(model_path)
        print("✅ 已建立並儲存新的模型於：", model_path)

    return model
