import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def bersihkan_data(df):
    # Validasi kolom 'Tanggal' ada
    if 'Tanggal' not in df.columns:
        raise ValueError("Kolom 'Tanggal' tidak ditemukan pada data!")

    # === 1. Parsing kolom tanggal ===
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%Y-%m-%d', errors='coerce')  

    # Drop baris yang gagal diparse
    df = df.dropna(subset=['Tanggal'])

    # === 2. Definisi kolom numerik ===
    columns_to_convert = ['RH_avg', 'Tavg', 'RR', 'ss']

    for col in columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # === 3. Ganti nilai tidak valid dengan NaN ===
    invalid_values = ['-', 9999, 8888]
    df.replace(invalid_values, np.nan, inplace=True)

    # RH_avg = 0 dianggap tidak valid
    if 'RH_avg' in df.columns:
        df['RH_avg'] = df['RH_avg'].replace(0, np.nan)

    # === 4. Interpolasi untuk isi kosong ===
    df = df.sort_values(by='Tanggal')  # urutkan data berdasarkan waktu
    df[columns_to_convert] = df[columns_to_convert].interpolate(method='linear')

    # Reset index jika dibutuhkan (optional)
    df = df.reset_index(drop=True)

    return df