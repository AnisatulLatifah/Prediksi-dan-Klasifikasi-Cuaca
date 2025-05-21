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

# Fungsi untuk membaca model aktif yang dipilih pengguna
def load_model_aktif():
    file_path = 'model/model_aktif.txt'

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, 'w') as f:
            f.write('model/model_utama')

    with open(file_path, 'r') as f:
        path = f.read().strip()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Folder model tidak ditemukan: {path}")

    model_rh = load_model(f"{path}/rh_avg_model.h5")
    model_tavg = load_model(f"{path}/tavg_model.h5")
    model_rr = load_model(f"{path}/rr_model.h5")
    model_ss = load_model(f"{path}/ss_model.h5")
    model_rf = joblib.load(f"{path}/rf_model.pkl")

    scaler_rh = joblib.load(f"{path}/scaler_rh_avg.pkl")
    scaler_tavg = joblib.load(f"{path}/scaler_tavg.pkl")
    scaler_rr = joblib.load(f"{path}/scaler_rr.pkl")
    scaler_ss = joblib.load(f"{path}/scaler_ss.pkl")

    return model_rh, model_tavg, model_rr, model_ss, model_rf, scaler_rh, scaler_tavg, scaler_rr, scaler_ss

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
    rh = row['RH_avg']
    tavg = row['Tavg']
    rr = row['RR']
    ss = row['ss']

    rh_min, rh_max = ideal_ranges['RH_avg']
    tavg_min, tavg_max = ideal_ranges['Tavg']
    rr_min, rr_max = ideal_ranges['RR']
    ss_min, ss_max = ideal_ranges['ss']

    if hasil == 'Baik':
        if rh_min <= rh <= rh_max:
            penjelasan.append("Kelembapan baik, karena berada dalam kisaran optimal (50–85%).")
        if tavg_min <= tavg <= tavg_max:
            penjelasan.append("Suhu baik, karena berada dalam kisaran optimal (25–30°C).")
        if rr_min <= rr <= rr_max:
            penjelasan.append("Curah hujan baik, karena memadai dalam batas ideal (2,6–8 mm).")
        if ss_min <= ss <= ss_max:
            penjelasan.append("Penyinaran baik, karena cukup lama dan mendukung fotosintesis (6–8 jam).")
    else:
        if rh < rh_min:
            penjelasan.append("Kelembapan buruk, karena terlalu rendah dari batas ideal.")
        elif rh > rh_max:
            penjelasan.append("Kelembapan buruk, karena melebihi batas ideal dan dapat mengganggu pertumbuhan.")
        if tavg < tavg_min:
            penjelasan.append("Suhu buruk, karena terlalu rendah dari kisaran optimal.")
        elif tavg > tavg_max:
            penjelasan.append("Suhu buruk, karena terlalu tinggi dari kisaran optimal.")
        if rr < rr_min:
            penjelasan.append("Curah hujan buruk, karena terlalu rendah dan berisiko kekeringan.")
        elif rr > rr_max:
            penjelasan.append("Curah hujan buruk, karena terlalu tinggi dan bisa menyebabkan kelebihan air.")
        if ss < ss_min:
            penjelasan.append("Penyinaran buruk, karena terlalu singkat untuk fotosintesis optimal.")
        elif ss > ss_max:
            penjelasan.append("Penyinaran buruk, karena terlalu lama dan bisa menyebabkan stres cahaya.")

    if not penjelasan:
        penjelasan.append("Kondisi cuaca tidak memenuhi kategori ideal maupun ekstrem.")

    return penjelasan

# Prediksi hari ini
def get_prediksi_klasifikasi_hari_ini(df_bersih):
    model_rh, model_tavg, model_rr, model_ss, model_rf, \
    scaler_rh, scaler_tavg, scaler_rr, scaler_ss = load_model_aktif()

    df_bersih['Tanggal'] = pd.to_datetime(df_bersih['Tanggal'])
    df_bersih = df_bersih[df_bersih['Tanggal'] < pd.to_datetime(date.today())]
    df_bersih = df_bersih.sort_values('Tanggal')

      # ✅ TARUH DI SINI:
    if df_bersih['RH_avg'].mean() < 50 and df_bersih['Tavg'].mean() > 50:
        print(">>> Kolom RH_avg dan Tavg tertukar, akan dibalik...")
        temp = df_bersih['RH_avg'].copy()
        df_bersih['RH_avg'] = df_bersih['Tavg']
        df_bersih['Tavg'] = temp

    # Tambahkan smoothing rolling 7 hari
    df_bersih['RH_avg'] = df_bersih['RH_avg'].rolling(window=7, center=True, min_periods=1).mean()
    df_bersih['Tavg'] = df_bersih['Tavg'].rolling(window=7, center=True, min_periods=1).mean()
    df_bersih['RR']    = df_bersih['RR'].rolling(window=7, center=True, min_periods=1).mean()
    df_bersih['ss']    = df_bersih['ss'].rolling(window=7, center=True, min_periods=1).mean()

    # print("===== 30 DATA RH_avg TERAKHIR YANG MASUK KE MODEL =====")
    # print(df_bersih['RH_avg'][-30:].values)

    # print("===== 30 DATA TAVG TERAKHIR YANG MASUK KE MODEL =====")
    # print(df_bersih['Tavg'][-30:].values)

    # print("===== 30 DATA RR TERAKHIR YANG MASUK KE MODEL =====")
    # print(df_bersih['RR'][-30:].values)

    # print("===== 30 DATA SS TERAKHIR YANG MASUK KE MODEL =====")
    # print(df_bersih['ss'][-30:].values)


    if len(df_bersih) < 30:
        raise ValueError("Data kurang dari 30 baris sebelum hari ini!")

    rh = prediksi_lstm(df_bersih['RH_avg'], model_rh, scaler_rh)
    tavg = prediksi_lstm(df_bersih['Tavg'], model_tavg, scaler_tavg)
    rr = prediksi_lstm(df_bersih['RR'], model_rr, scaler_rr)
    ss = prediksi_lstm(df_bersih['ss'], model_ss, scaler_ss)

    # # ====== Prediksi RH_avg ======
    # rh_scaled = model_rh.predict(scaler_rh.transform(np.array(df_bersih['RH_avg'][-30:]).reshape(-1, 1)).reshape(1, 30, 1))
    # rh = scaler_rh.inverse_transform(rh_scaled.reshape(-1, 1))[0][0]
    # print(">>> RH_avg Scaled :", rh_scaled)
    # print(">>> RH_avg Final  :", rh)

    # # ====== Prediksi Tavg ======
    # tavg_scaled = model_tavg.predict(scaler_tavg.transform(np.array(df_bersih['Tavg'][-30:]).reshape(-1, 1)).reshape(1, 30, 1))
    # tavg = scaler_tavg.inverse_transform(tavg_scaled.reshape(-1, 1))[0][0]
    # print(">>> Tavg Scaled   :", tavg_scaled)
    # print(">>> Tavg Final    :", tavg)

    # # ====== Prediksi RR ======
    # rr_scaled = model_rr.predict(scaler_rr.transform(np.array(df_bersih['RR'][-30:]).reshape(-1, 1)).reshape(1, 30, 1))
    # rr = scaler_rr.inverse_transform(rr_scaled.reshape(-1, 1))[0][0]
    # print(">>> RR Scaled     :", rr_scaled)
    # print(">>> RR Final      :", rr)

    # # ====== Prediksi SS ======
    # ss_scaled = model_ss.predict(scaler_ss.transform(np.array(df_bersih['ss'][-30:]).reshape(-1, 1)).reshape(1, 30, 1))
    # ss = scaler_ss.inverse_transform(ss_scaled.reshape(-1, 1))[0][0]
    # print(">>> SS Scaled     :", ss_scaled)
    # print(">>> SS Final      :", ss)


    df_pred = pd.DataFrame([[rh, tavg, rr, ss]], columns=['RH_avg', 'Tavg', 'RR', 'ss'])
    hasil_klasifikasi = model_rf.predict(df_pred)[0]

    row = {
        'RH_avg': rh, 'Tavg': tavg, 'RR': rr, 'ss': ss,
        'Peluang Pertumbuhan': hasil_klasifikasi
    }
    keterangan = generate_keterangan(row)

    return {
        'tanggal': str(date.today()),
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

    # ✅ TARUH DI SINI:
    if df_bersih['RH_avg'].mean() < 50 and df_bersih['Tavg'].mean() > 50:
        print(">>> Kolom RH_avg dan Tavg tertukar, akan dibalik...")
        temp = df_bersih['RH_avg'].copy()
        df_bersih['RH_avg'] = df_bersih['Tavg']
        df_bersih['Tavg'] = temp

    # Terapkan rolling smoothing window=7
    df_bersih['RH_avg'] = df_bersih['RH_avg'].rolling(window=7, center=True, min_periods=1).mean()
    df_bersih['Tavg'] = df_bersih['Tavg'].rolling(window=7, center=True, min_periods=1).mean()
    df_bersih['RR'] = df_bersih['RR'].rolling(window=7, center=True, min_periods=1).mean()
    df_bersih['ss'] = df_bersih['ss'].rolling(window=7, center=True, min_periods=1).mean()

    if len(df_bersih) < 30:
        raise ValueError("Data tidak cukup untuk prediksi 7 hari ke depan!")

    # Ambil 30 data terakhir sebelum last_date
    df_hist = df_bersih[df_bersih['Tanggal'] <= pd.to_datetime(last_date)]
    if len(df_hist) < 30:
        raise ValueError("Data historis sebelum tanggal terakhir tidak mencukupi 30 hari!")

    rh_seq = scaler_rh.transform(np.array(df_hist['RH_avg'][-30:]).reshape(-1, 1)).reshape(1, 30, 1)
    tavg_seq = scaler_tavg.transform(np.array(df_hist['Tavg'][-30:]).reshape(-1, 1)).reshape(1, 30, 1)
    rr_seq = scaler_rr.transform(np.array(df_hist['RR'][-30:]).reshape(-1, 1)).reshape(1, 30, 1)
    ss_seq = scaler_ss.transform(np.array(df_hist['ss'][-30:]).reshape(-1, 1)).reshape(1, 30, 1)

    for i in range(1, 8):
        rh_pred_scaled = model_rh.predict(rh_seq)[0][0]
        tavg_pred_scaled = model_tavg.predict(tavg_seq)[0][0]
        rr_pred_scaled = model_rr.predict(rr_seq)[0][0]
        ss_pred_scaled = model_ss.predict(ss_seq)[0][0]

        rh_seq = np.append(rh_seq[:, 1:, :], [[[rh_pred_scaled]]], axis=1)
        tavg_seq = np.append(tavg_seq[:, 1:, :], [[[tavg_pred_scaled]]], axis=1)
        rr_seq = np.append(rr_seq[:, 1:, :], [[[rr_pred_scaled]]], axis=1)
        ss_seq = np.append(ss_seq[:, 1:, :], [[[ss_pred_scaled]]], axis=1)

        rh = float(scaler_rh.inverse_transform([[rh_pred_scaled]])[0][0])
        tavg = float(scaler_tavg.inverse_transform([[tavg_pred_scaled]])[0][0])
        rr = float(scaler_rr.inverse_transform([[rr_pred_scaled]])[0][0])
        ss = float(scaler_ss.inverse_transform([[ss_pred_scaled]])[0][0])

        df_pred = pd.DataFrame([[rh, tavg, rr, ss]], columns=['RH_avg', 'Tavg', 'RR', 'ss'])
        klasifikasi = model_rf.predict(df_pred)[0]

        row = {
            'RH_avg': rh, 'Tavg': tavg, 'RR': rr, 'ss': ss,
            'Peluang Pertumbuhan': klasifikasi
        }
        keterangan = generate_keterangan(row)

        # ✅ Gunakan last_date yang dikirim sebagai acuan tanggal prediksi
        tanggal_pred = last_date + timedelta(days=i)
        hari = ubah_ke_hari_indo(tanggal_pred.strftime('%A'))
        bulan = ubah_ke_bulan_indo(tanggal_pred.month)
        tanggal_str = f"{tanggal_pred.day:02d} {bulan} {tanggal_pred.year}"

        hasil_list.append({
            'hari': hari,
            'tanggal': tanggal_str,
            'RH_avg': round(rh, 2),
            'Tavg': round(tavg, 2),
            'RR': round(rr, 2),
            'ss': round(ss, 2),
            'klasifikasi': klasifikasi,
            'keterangan': keterangan
        })

    return hasil_list
