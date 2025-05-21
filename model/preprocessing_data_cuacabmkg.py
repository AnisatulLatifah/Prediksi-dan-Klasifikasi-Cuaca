import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def bersihkan_data(df):
    # 1. Validasi & parsing tanggal
    if 'Tanggal' not in df.columns:
        raise ValueError("Kolom 'Tanggal' tidak ditemukan!")

    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['Tanggal'])

    # 2. Ubah ke numerik
    columns_to_convert = ['RH_avg', 'Tavg', 'RR', 'ss']
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Ganti nilai tidak valid
    df.replace(['-', 9999, 8888], np.nan, inplace=True)
    df['RH_avg'] = df['RH_avg'].replace(0, np.nan)

    # 4. Interpolasi dan sort
    df = df.sort_values('Tanggal')
    df[columns_to_convert] = df[columns_to_convert].interpolate(method='linear')

    # 5. Susun ulang kolom (optional tapi direkomendasikan)
    df = df[['Tanggal', 'RH_avg', 'Tavg', 'RR', 'ss']]

    return df.reset_index(drop=True)
