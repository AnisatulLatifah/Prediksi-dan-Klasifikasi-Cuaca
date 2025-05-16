import os
import numpy as np
import pandas as pd
import mysql.connector
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.losses import MeanSquaredError


# ====== Fungsi bantu ======
def get_combined_data():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='cuaca_db'
    )
    cuaca_lama = pd.read_sql("SELECT * FROM cuaca_lama", conn)
    input_user = pd.read_sql("SELECT * FROM cuaca_input_user", conn)
    conn.close()
    return pd.concat([cuaca_lama, input_user], ignore_index=True)

def get_next_model_folder():
    base_dir = "model/model_retrain"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    existing = [d for d in os.listdir(base_dir) if d.startswith("model_")]
    next_id = len(existing) + 1
    folder = os.path.join(base_dir, f"model_{next_id}")
    os.makedirs(folder, exist_ok=True)
    return folder

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=MeanSquaredError())  
    return model

def prepare_lstm_data(data, feature):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[[feature]])
    X, y = [], []
    for i in range(30, len(scaled)):
        X.append(scaled[i-30:i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaler

# ====== Main training ======
def train_and_save_all():
    df = get_combined_data()
    df = df.sort_values('Tanggal')

    folder = get_next_model_folder()
    print("🔁 Retrain model dan simpan ke:", folder)

    # Fitur LSTM
    lstm_features = ['RH_avg', 'Tavg', 'RR', 'ss']
    for fitur in lstm_features:
        X, y, scaler = prepare_lstm_data(df, fitur)
        model = build_lstm_model((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=10, verbose=0)
        model.save(os.path.join(folder, f"{fitur.lower()}_model.h5"))
        joblib.dump(scaler, os.path.join(folder, f"scaler_{fitur.lower()}.pkl"))
        print(f"✅ {fitur} LSTM model & scaler disimpan.")

    # Klasifikasi RF
    X_rf = df[['RH_avg', 'Tavg', 'RR', 'ss']]
    y_rf = df['Label'] if 'Label' in df.columns else (np.random.choice(['Baik', 'Buruk'], len(X_rf)))  # dummy
    rf_model = RandomForestClassifier()
    rf_model.fit(X_rf, y_rf)
    joblib.dump(rf_model, os.path.join(folder, "rf_model.pkl"))
    print("✅ Random Forest klasifikasi disimpan.")

    # Simpan tanggal retrain
    with open(os.path.join(folder, "tanggal_retrain.txt"), 'w') as f:
        f.write(datetime.now().strftime("%Y-%m-%d"))

    # Jadikan model ini aktif
    with open("model/model_aktif.txt", 'w') as f:
        f.write(folder.replace("\\", "/"))  # konsisten path

    print("🎉 Retrain selesai! Model ini otomatis menjadi aktif.")

# Jalankan saat file ini dieksekusi langsung
if __name__ == "__main__":
    train_and_save_all()
