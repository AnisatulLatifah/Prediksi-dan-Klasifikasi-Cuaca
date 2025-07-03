import pandas as pd
import numpy as np
import os
import joblib
from datetime import date, timedelta
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier

# Ideal range untuk pertumbuhan optimal
ideal_ranges = {
    'RH_avg': (50, 85),
    'Tavg': (25, 30),
    'RR': (2.6, 8),
    'ss': (6, 8)
}
# Ambil tanggal terakhir dari tabel berdasarkan model aktif
def get_last_date_by_model():
    import mysql.connector
    import pandas as pd

    with open("model/model_aktif.txt", "r") as f:
        path_model = f.read().strip()

    if 'model_utama' in path_model:
        table = 'cuaca_lama'
    else:
        table = 'cuaca_input_user'

    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='cuaca_db'
    )
    cursor = conn.cursor()
    cursor.execute(f"SELECT MAX(tanggal) FROM {table}")
    result = cursor.fetchone()[0]
    conn.close()

    return pd.to_datetime(result) if result else pd.to_datetime(date.today())

# Fungsi untuk mengecek apakah model aktif adalah model utama
def is_model_utama():
    try:
        with open("model/model_aktif.txt", "r") as f:
            aktif = f.read().strip()
            return "model_utama" in aktif
    except:
        return True

# Fungsi untuk membaca model aktif yang dipilih pengguna
def load_model_aktif():
    path = 'model/model_utama'

    if not os.path.exists(path):
        raise FileNotFoundError(f"Folder model tidak ditemukan: {path}")

    model_rh = load_model(f"{path}/rh_avg_model.h5", compile=False)
    model_tavg = load_model(f"{path}/tavg_model.h5", compile=False)
    model_rr = load_model(f"{path}/rr_model.h5", compile=False)
    model_ss = load_model(f"{path}/ss_model.h5", compile=False)
    model_rf = joblib.load(f"{path}/rf_model.pkl")

    scaler_rh = joblib.load(f"{path}/scaler_rh_avg.pkl")
    scaler_tavg = joblib.load(f"{path}/scaler_tavg.pkl")
    scaler_rr = joblib.load(f"{path}/scaler_rr.pkl")
    scaler_ss = joblib.load(f"{path}/scaler_ss.pkl")

    return model_rh, model_tavg, model_rr, model_ss, model_rf, scaler_rh, scaler_tavg, scaler_rr, scaler_ss

# Fungsi ambil data dari database sesuai model aktif
def get_df_bersih_by_model():
    import mysql.connector

    with open("model/model_aktif.txt", "r") as f:
        path_model = f.read().strip()

    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='cuaca_db'
    )

    if 'model_utama' in path_model:
        df = pd.read_sql("SELECT * FROM cuaca_lama ORDER BY tanggal ASC", conn)
    else:
        df_lama = pd.read_sql("SELECT * FROM cuaca_lama ORDER BY tanggal ASC", conn)
        df_input = pd.read_sql("SELECT * FROM cuaca_input_user ORDER BY tanggal ASC", conn)
        df = pd.concat([df_lama, df_input], ignore_index=True)

    conn.close()
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    return df

# Fungsi prediksi LSTM 1 hari
def prediksi_lstm(data_series, model, scaler):
    data_30hari = np.array(data_series[-30:]).reshape(-1, 1)
    scaled = scaler.transform(data_30hari).reshape(1, 30, 1)
    pred_scaled = model.predict(scaled)
    pred_asli = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    return float(pred_asli[0][0])

# Fungsi deskripsi keterangan
def generate_keterangan(row):
    penjelasan = []
    hasil = row['Peluang Pertumbuhan']
    rh, tavg, rr, ss = row['RH_avg'], row['Tavg'], row['RR'], row['ss']
    rh_min, rh_max = ideal_ranges['RH_avg']
    tavg_min, tavg_max = ideal_ranges['Tavg']
    rr_min, rr_max = ideal_ranges['RR']
    ss_min, ss_max = ideal_ranges['ss']

    if hasil == 'Baik':
        if rh_min <= rh <= rh_max:
            penjelasan.append("Kelembapan udara baik, karena berada dalam kisaran optimal.")
        if tavg_min <= tavg <= tavg_max:
            penjelasan.append("Suhu udara baik, karena berada dalam kisaran optimal.")
        if rr_min <= rr <= rr_max:
            penjelasan.append("Curah hujan baik, karena berada dalam kisaran optimal.")
        if ss_min <= ss <= ss_max:
            penjelasan.append("Penyinaran matahari baik, karena berada dalam kisaran optimal.")
    else:
        if rh < rh_min:
            penjelasan.append("Kelembapan buruk, karena terlalu rendah dari kisaran optimal (akar bisa mengering dan menghambat penyerapan nutrisi).")
        elif rh > rh_max:
            penjelasan.append("Kelembapan buruk, karena melebihi kisaran optimal (akar rentan membusuk akibat tanah terlalu lembap).")

        if tavg < tavg_min:
            penjelasan.append("Suhu buruk, karena terlalu rendah dari kisaran optimal (pertumbuhan melambat dan buah sulit berkembang).")
        elif tavg > tavg_max:
            penjelasan.append("Suhu buruk, karena melebihi kisaran optimal (daun bisa layu dan tanaman mengalami stres panas).")

        if rr < rr_min:
            penjelasan.append("Curah hujan buruk, karena terlalu rendah dari kisaran optimal (tanaman kekurangan air, berisiko layu).")
        elif rr > rr_max:
            penjelasan.append("Curah hujan buruk, karena melebihi kisaran optimal (risiko jamur meningkat dan akar bisa membusuk).")

        if ss < ss_min:
            penjelasan.append("Penyinaran buruk, karena terlalu singkat (fotosintesis terganggu, pertumbuhan terhambat).")
        elif ss > ss_max:
            penjelasan.append("Penyinaran buruk, karena terlalu lama (tanaman bisa stres cahaya dan daun mengering).")

    return penjelasan if penjelasan else ["Kondisi cuaca tidak memenuhi kategori ideal maupun ekstrem."]

# Prediksi hari ini
def get_prediksi_klasifikasi_hari_ini(df_bersih, last_date):
    model_rh, model_tavg, model_rr, model_ss, model_rf, \
    scaler_rh, scaler_tavg, scaler_rr, scaler_ss = load_model_aktif()

    df_bersih['Tanggal'] = pd.to_datetime(df_bersih['Tanggal'])
    df_bersih = df_bersih[df_bersih['Tanggal'] <= pd.to_datetime(last_date)]
    df_bersih = df_bersih.sort_values('Tanggal')

    if df_bersih['RH_avg'].mean() < 50 and df_bersih['Tavg'].mean() > 50:
        temp = df_bersih['RH_avg'].copy()
        df_bersih['RH_avg'] = df_bersih['Tavg']
        df_bersih['Tavg'] = temp

    # ========== PREPROCESSING SEBELUM PREDIKSI HARI INI ==========
    df_bersih['RH_avg'] = df_bersih['RH_avg'].rolling(window=7, center=True, min_periods=1).mean()
    df_bersih['Tavg'] = df_bersih['Tavg'].rolling(window=7, center=True, min_periods=1).mean()
    df_bersih['RR'] = np.log1p(df_bersih['RR'])  
    df_bersih['RR'] = df_bersih['RR'].rolling(window=7, center=True, min_periods=1).mean()
    df_bersih['ss'] = df_bersih['ss'].rolling(window=7, center=True, min_periods=1).mean()


    if len(df_bersih) < 30:
        raise ValueError("Data kurang dari 30 baris sebelum hari ini!")

    rh = prediksi_lstm(df_bersih['RH_avg'], model_rh, scaler_rh)
    tavg = prediksi_lstm(df_bersih['Tavg'], model_tavg, scaler_tavg)
    rr_log = prediksi_lstm(df_bersih['RR'], model_rr, scaler_rr)
    rr = np.expm1(rr_log)
    ss = prediksi_lstm(df_bersih['ss'], model_ss, scaler_ss)

    # ========== PROSES KLASIFIKASI ==========
    df_pred = pd.DataFrame([[rh, tavg, rr, ss]], columns=['RH_avg', 'Tavg', 'RR', 'ss'])
    hasil_klasifikasi = model_rf.predict(df_pred)[0]

    row = {'RH_avg': rh, 'Tavg': tavg, 'RR': rr, 'ss': ss, 'Peluang Pertumbuhan': hasil_klasifikasi}
    keterangan = generate_keterangan(row)

    return {
        'tanggal': str(last_date + timedelta(days=1)),
        'RH_avg': round(rh, 2),
        'Tavg': round(tavg, 2),
        'RR': round(rr, 2),
        'ss': round(ss, 2),
        'klasifikasi': hasil_klasifikasi,
        'keterangan': keterangan
    }

# Konversi ke hari dan bulan Indonesia
def ubah_ke_hari_indo(nama_hari):
    mapping = {
        'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
        'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'
    }
    return mapping.get(nama_hari, nama_hari)

def ubah_ke_bulan_indo(nomor_bulan):
    mapping = {
        1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 5: 'Mei', 6: 'Juni',
        7: 'Juli', 8: 'Agustus', 9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
    }
    return mapping.get(nomor_bulan, str(nomor_bulan))

# Prediksi 7 hari ke depan (rekursif)
def get_prediksi_7_hari(df_bersih, last_date):
    model_rh, model_tavg, model_rr, model_ss, model_rf, \
    scaler_rh, scaler_tavg, scaler_rr, scaler_ss = load_model_aktif()

    hasil_list = []
    df_bersih['Tanggal'] = pd.to_datetime(df_bersih['Tanggal'])
    df_bersih = df_bersih.sort_values('Tanggal')

    if df_bersih['RH_avg'].mean() < 50 and df_bersih['Tavg'].mean() > 50:
        df_bersih['RH_avg'], df_bersih['Tavg'] = df_bersih['Tavg'], df_bersih['RH_avg']

    # ========== PISAHKAN DATA UNTUK HARI INI ==========
    df_hari_ini = df_bersih.copy()  # salin sebelum rolling
    hari_ini_result = get_prediksi_klasifikasi_hari_ini(df_hari_ini, last_date)

    # ========== ROLLING SETELAHNYA (untuk hari 2-7) ==========
    df_bersih['RH_avg'] = df_bersih['RH_avg'].rolling(window=7, center=True, min_periods=1).mean()
    df_bersih['Tavg'] = df_bersih['Tavg'].rolling(window=7, center=True, min_periods=1).mean()
    df_bersih['RR'] = np.log1p(df_bersih['RR'])
    df_bersih['RR']    = df_bersih['RR'].rolling(window=7, center=True, min_periods=1).mean()
    df_bersih['ss']    = df_bersih['ss'].rolling(window=7, center=True, min_periods=1).mean()

    if len(df_bersih) < 30:
        raise ValueError("Data tidak cukup untuk prediksi!")

    df_hist = df_bersih[df_bersih['Tanggal'] <= pd.to_datetime(last_date)]

    # ========== MASUKKAN HASIL HARI PERTAMA ==========
    rh1 = hari_ini_result['RH_avg']
    tavg1 = hari_ini_result['Tavg']
    rr1 = hari_ini_result['RR']
    ss1 = hari_ini_result['ss']
    klasifikasi1 = hari_ini_result['klasifikasi']
    ket1 = hari_ini_result['keterangan']
    tgl1 = pd.to_datetime(hari_ini_result['tanggal'])

    hasil_list.append({
        'hari': ubah_ke_hari_indo(tgl1.strftime('%A')),
        'tanggal': f"{tgl1.day:02d} {ubah_ke_bulan_indo(tgl1.month)} {tgl1.year}",
        'RH_avg': rh1, 'Tavg': tavg1, 'RR': rr1, 'ss': ss1,
        'klasifikasi': klasifikasi1,
        'keterangan': ket1
    })

    # ========== SIAPKAN INPUT UNTUK HARI 2-7 (RECURSIVE)==========
    rh_seq = scaler_rh.transform(np.array(df_hist['RH_avg'][-30:]).reshape(-1, 1)).reshape(1, 30, 1)
    tavg_seq = scaler_tavg.transform(np.array(df_hist['Tavg'][-30:]).reshape(-1, 1)).reshape(1, 30, 1)
    rr_seq = scaler_rr.transform(np.array(df_hist['RR'][-30:]).reshape(-1, 1)).reshape(1, 30, 1)
    ss_seq = scaler_ss.transform(np.array(df_hist['ss'][-30:]).reshape(-1, 1)).reshape(1, 30, 1)

    rh_seq = np.append(rh_seq[:, 1:, :], [[[scaler_rh.transform([[rh1]])[0][0]]]], axis=1)
    tavg_seq = np.append(tavg_seq[:, 1:, :], [[[scaler_tavg.transform([[tavg1]])[0][0]]]], axis=1)
    rr_seq = np.append(rr_seq[:, 1:, :], [[[scaler_rr.transform([[rr1]])[0][0]]]], axis=1)
    ss_seq = np.append(ss_seq[:, 1:, :], [[[scaler_ss.transform([[ss1]])[0][0]]]], axis=1)

    # ========== LOOP HARI 2 SAMPAI 7 ==========
    for i in range(2, 8):
        rh_pred = model_rh.predict(rh_seq)[0][0]
        tavg_pred = model_tavg.predict(tavg_seq)[0][0]
        rr_pred = model_rr.predict(rr_seq)[0][0]
        ss_pred = model_ss.predict(ss_seq)[0][0]

        rh_seq = np.append(rh_seq[:, 1:, :], [[[rh_pred]]], axis=1)
        tavg_seq = np.append(tavg_seq[:, 1:, :], [[[tavg_pred]]], axis=1)
        rr_seq = np.append(rr_seq[:, 1:, :], [[[rr_pred]]], axis=1)
        ss_seq = np.append(ss_seq[:, 1:, :], [[[ss_pred]]], axis=1)

        rh = float(scaler_rh.inverse_transform([[rh_pred]])[0][0])
        tavg = float(scaler_tavg.inverse_transform([[tavg_pred]])[0][0])
        rr = float(scaler_rr.inverse_transform([[rr_pred]])[0][0])
        ss = float(scaler_ss.inverse_transform([[ss_pred]])[0][0])

        df_pred = pd.DataFrame([[rh, tavg, rr, ss]], columns=['RH_avg', 'Tavg', 'RR', 'ss'])
        klasifikasi = model_rf.predict(df_pred)[0]
        row = {'RH_avg': rh, 'Tavg': tavg, 'RR': rr, 'ss': ss, 'Peluang Pertumbuhan': klasifikasi}
        keterangan = generate_keterangan(row)

        tgl_pred = last_date + timedelta(days=i)
        hasil_list.append({
            'hari': ubah_ke_hari_indo(tgl_pred.strftime('%A')),
            'tanggal': f"{tgl_pred.day:02d} {ubah_ke_bulan_indo(tgl_pred.month)} {tgl_pred.year}",
            'RH_avg': round(rh, 2), 'Tavg': round(tavg, 2),
            'RR': round(rr, 2), 'ss': round(ss, 2),
            'klasifikasi': klasifikasi,
            'keterangan': keterangan
        })

    return hasil_list
