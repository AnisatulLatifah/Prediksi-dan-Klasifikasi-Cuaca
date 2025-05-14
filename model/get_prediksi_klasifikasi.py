import pandas as pd
import numpy as np
from keras.models import load_model
import joblib
from sklearn.ensemble import RandomForestClassifier
from datetime import date, timedelta

# Load model & scaler
model_rh = load_model('model/rh_avg_model.h5')
model_tavg = load_model('model/tavg_model.h5')
model_rr = load_model('model/rr_model.h5')
model_ss = load_model('model/ss_model.h5')
model_rf = joblib.load('model/rf_model.pkl')

scaler_rh = joblib.load('model/scaler_rh_avg.pkl')
scaler_tavg = joblib.load('model/scaler_tavg.pkl')
scaler_rr = joblib.load('model/scaler_rr.pkl')
scaler_ss = joblib.load('model/scaler_ss.pkl')

# Ideal range
ideal_ranges = {
    'RH_avg': (50, 85),
    'Tavg': (25, 30),
    'RR': (2.6, 8),
    'ss': (6, 8)
}

def generate_keterangan(row):
    penjelasan = []
    hasil = row['Peluang Pertumbuhan']

    rh = row['RH_avg']
    tavg = row['Tavg']
    rr = row['RR']
    ss = row['ss']

    rh_min, rh_max = ideal_ranges['RH_avg']
    if hasil == 'Baik' and rh_min <= rh <= rh_max:
        penjelasan.append("Kelembapan baik, karena berada dalam kisaran optimal (50–85%).")
    elif hasil == 'Buruk':
        if rh < rh_min:
            penjelasan.append("Kelembapan buruk, karena terlalu rendah dari batas ideal.")
        elif rh > rh_max:
            penjelasan.append("Kelembapan buruk, karena melebihi batas ideal dan dapat mengganggu pertumbuhan.")

    tavg_min, tavg_max = ideal_ranges['Tavg']
    if hasil == 'Baik' and tavg_min <= tavg <= tavg_max:
        penjelasan.append("Suhu baik, karena berada dalam kisaran optimal (25–30°C).")
    elif hasil == 'Buruk':
        if tavg < tavg_min:
            penjelasan.append("Suhu buruk, karena terlalu rendah dari kisaran optimal.")
        elif tavg > tavg_max:
            penjelasan.append("Suhu buruk, karena terlalu tinggi dari kisaran optimal.")

    rr_min, rr_max = ideal_ranges['RR']
    if hasil == 'Baik' and rr_min <= rr <= rr_max:
        penjelasan.append("Curah hujan baik, karena memadai dalam batas ideal (2,6–8 mm).")
    elif hasil == 'Buruk':
        if rr < rr_min:
            penjelasan.append("Curah hujan buruk, karena terlalu rendah dan berisiko kekeringan.")
        elif rr > rr_max:
            penjelasan.append("Curah hujan buruk, karena terlalu tinggi dan bisa menyebabkan kelebihan air.")

    ss_min, ss_max = ideal_ranges['ss']
    if hasil == 'Baik' and ss_min <= ss <= ss_max:
        penjelasan.append("Penyinaran baik, karena cukup lama dan mendukung fotosintesis (6–8 jam).")
    elif hasil == 'Buruk':
        if ss < ss_min:
            penjelasan.append("Penyinaran buruk, karena terlalu singkat untuk fotosintesis optimal.")
        elif ss > ss_max:
            penjelasan.append("Penyinaran buruk, karena terlalu lama dan bisa menyebabkan stres cahaya.")

    if not penjelasan:
        penjelasan.append("Kondisi cuaca tidak memenuhi kategori ideal maupun ekstrem.")

    return penjelasan

def prediksi_lstm(data_series, model, scaler):
    data_30hari = np.array(data_series[-30:]).reshape(-1, 1)
    scaled = scaler.transform(data_30hari).reshape(1, 30, 1)
    pred_scaled = model.predict(scaled)
    pred_asli = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    return float(pred_asli[0][0])

# Prediksi hari ini berdasarkan tanggal sekarang
def get_prediksi_klasifikasi_hari_ini(df_bersih):
    df_bersih['Tanggal'] = pd.to_datetime(df_bersih['Tanggal'])
    df_bersih = df_bersih[df_bersih['Tanggal'] < pd.to_datetime(date.today())]
    df_bersih = df_bersih.sort_values('Tanggal')

    if len(df_bersih) < 30:
        raise ValueError("Data kurang dari 30 baris sebelum hari ini!")

    rh = prediksi_lstm(df_bersih['RH_avg'], model_rh, scaler_rh)
    tavg = prediksi_lstm(df_bersih['Tavg'], model_tavg, scaler_tavg)
    rr = prediksi_lstm(df_bersih['RR'], model_rr, scaler_rr)
    ss = prediksi_lstm(df_bersih['ss'], model_ss, scaler_ss)

    df_pred = pd.DataFrame([{
        'RH_avg': rh, 'Tavg': tavg, 'RR': rr, 'ss': ss
    }])
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

# Prediksi 7 Hari kedepan
def get_prediksi_7_hari(df_bersih):
    hasil_list = []
    df_bersih['Tanggal'] = pd.to_datetime(df_bersih['Tanggal'])
    df_bersih = df_bersih[df_bersih['Tanggal'] < pd.to_datetime(date.today())]
    df_bersih = df_bersih.sort_values('Tanggal')

    if len(df_bersih) < 37:
        raise ValueError("Data tidak cukup untuk prediksi 7 hari ke depan!")

    if len(df_bersih) < 37:
        raise ValueError("Data tidak cukup untuk prediksi 7 hari ke depan!")

    for i in range(1, 8):
        subset = df_bersih.iloc[-(30+i):-i]

        rh = prediksi_lstm(subset['RH_avg'], model_rh, scaler_rh)
        tavg = prediksi_lstm(subset['Tavg'], model_tavg, scaler_tavg)
        rr = prediksi_lstm(subset['RR'], model_rr, scaler_rr)
        ss = prediksi_lstm(subset['ss'], model_ss, scaler_ss)

        df_pred = pd.DataFrame([{
            'RH_avg': rh, 'Tavg': tavg, 'RR': rr, 'ss': ss
        }])
        klasifikasi = model_rf.predict(df_pred)[0]

        row = {
            'RH_avg': rh, 'Tavg': tavg, 'RR': rr, 'ss': ss,
            'Peluang Pertumbuhan': klasifikasi
        }
        keterangan = generate_keterangan(row)

        tanggal_pred = date.today() + timedelta(days=i)
        hari_indo = ubah_ke_hari_indo(tanggal_pred.strftime('%A'))
        bulan_indo = ubah_ke_bulan_indo(tanggal_pred.month)
        tanggal_str = f"{tanggal_pred.day:02d} {bulan_indo} {tanggal_pred.year}"

        hasil_list.append({
            'hari': hari_indo,
            'tanggal': tanggal_str,
            'RH_avg': round(rh, 2),
            'Tavg': round(tavg, 2),
            'RR': round(rr, 2),
            'ss': round(ss, 2),
            'klasifikasi': klasifikasi,
            'keterangan': keterangan
        })

    return hasil_list
