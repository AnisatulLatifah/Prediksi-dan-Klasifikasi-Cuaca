import os
import numpy as np
import pandas as pd
import mysql.connector
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import joblib
from keras.models import load_model
import tensorflow as tf
tf.config.run_functions_eagerly(True)

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

def prepare_lstm_data(data, feature, scaler_path):
    scaler = joblib.load(scaler_path)  # Pakai scaler dari model_utama
    data_clean = data[[feature]].dropna()
    if len(data_clean) < 30:
        raise ValueError(f"⚠️ Data {feature} terlalu sedikit setelah dropna: hanya {len(data_clean)}")
    scaled = scaler.transform(data_clean)
    X, y = [], []
    for i in range(30, len(scaled)):
        X.append(scaled[i-30:i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaler

# ====== Main Training ======
def train_and_save_all(fine_tune=True):
    df = get_combined_data()
    df = df.sort_values('Tanggal')

    # Rolling smoothing 7 hari
    for col in ['RH_avg', 'Tavg', 'RR', 'ss']:
        df[col] = df[col].rolling(window=7, center=True, min_periods=1).mean()
        print(f"📊 Data valid {col} setelah rolling:", df[col].dropna().shape[0])

    folder = get_next_model_folder()
    print("🔁 Fine-tuning dari model_utama dan simpan ke:", folder)

    # Konfigurasi spesifik per fitur
    epoch_dict = {
        'RH_avg': 10,
        'Tavg': 10,
        'RR': 10,
        'ss': 10
    }

    batch_dict = {
        'RH_avg': 32,
        'Tavg': 32,
        'RR': 32,
        'ss': 32
    }

    fitur_list = ['RH_avg', 'Tavg', 'RR', 'ss']
    for fitur in fitur_list:
        fitur_lower = fitur.lower()
        path_model_utama = f"model/model_utama/{fitur_lower}_model.h5"
        path_scaler_utama = f"model/model_utama/scaler_{fitur_lower}.pkl"

        if fine_tune and os.path.exists(path_model_utama) and os.path.exists(path_scaler_utama):
            model = load_model(path_model_utama, compile=False)
            model.compile(optimizer='adam', loss='mse')
            print(f"🔧 Fine-tune model {fitur} dari model_utama")
        else:
            print(f"⚠️ File model_utama {fitur_lower} tidak ditemukan! Gagal retrain.")
            continue

        try:
            X, y, scaler = prepare_lstm_data(df, fitur, path_scaler_utama)
            model.fit(
            X, y,
            epochs=epoch_dict[fitur],
            batch_size=batch_dict[fitur],
            verbose=1,
            shuffle=False
            )
            
            model.save(os.path.join(folder, f"{fitur_lower}_model.h5"))
            joblib.dump(scaler, os.path.join(folder, f"scaler_{fitur_lower}.pkl"))
            print(f"✅ {fitur} retrain selesai dan disimpan.")
        except Exception as e:
            print(f"❌ Gagal retrain {fitur}: {e}")
            continue

    # Random Forest
    try:
        X_rf = df[['RH_avg', 'Tavg', 'RR', 'ss']].dropna()
        y_rf = df['Label'] if 'Label' in df.columns else (
            np.random.choice(['Baik', 'Buruk'], len(X_rf))
        )
        rf_model = RandomForestClassifier()
        rf_model.fit(X_rf, y_rf)
        joblib.dump(rf_model, os.path.join(folder, "rf_model.pkl"))
        print("✅ Random Forest klasifikasi disimpan.")
    except Exception as e:
        print(f"❌ Gagal retrain RandomForest: {e}")

    # Simpan metadata retrain
    with open(os.path.join(folder, "tanggal_retrain.txt"), 'w') as f:
        f.write(datetime.now().strftime("%Y-%m-%d"))

    with open("model/model_aktif.txt", 'w') as f:
        f.write(folder.replace("\\", "/"))

    print("🎉 Retrain selesai. Model ini sekarang menjadi aktif:", folder)

# Jalankan langsung
if __name__ == "__main__":
    train_and_save_all(fine_tune=True)