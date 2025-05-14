from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from datetime import datetime
from datetime import timedelta
import mysql.connector
import os
from werkzeug.utils import secure_filename  # ⬅️ penting

from model.get_prediksi_klasifikasi import (
    get_prediksi_klasifikasi_hari_ini,
    get_prediksi_7_hari
)
from model.preprocessing_data_cuacabmkg import bersihkan_data

app = Flask(__name__)
app.secret_key = 'rahasia'

# ====== Konfigurasi Upload CSV ======
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ====== Koneksi ke MySQL ======
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="cuaca_db"
)
cursor = db.cursor()

# ====== HALAMAN HOME ======
@app.route('/')
def home():
    # Ambil data dari database cuaca_lama
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='cuaca_db'
    )
    query = "SELECT * FROM cuaca_lama ORDER BY tanggal ASC"
    df_bersih = pd.read_sql(query, conn)
    conn.close()

    # Pastikan format tanggal benar
    df_bersih['Tanggal'] = pd.to_datetime(df_bersih['Tanggal'])

    # Proses prediksi hari ini
    hasil = get_prediksi_klasifikasi_hari_ini(df_bersih)

    # Format tanggal Indonesia
    now = datetime.now()
    hari_indonesia = {
        'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
        'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'
    }
    bulan_indonesia = {
        'January': 'Januari', 'February': 'Februari', 'March': 'Maret', 'April': 'April',
        'May': 'Mei', 'June': 'Juni', 'July': 'Juli', 'August': 'Agustus',
        'September': 'September', 'October': 'Oktober', 'November': 'November', 'December': 'Desember'
    }

    hari = hari_indonesia[now.strftime('%A')]
    bulan = bulan_indonesia[now.strftime('%B')]
    tanggal_indo = f"{now.day:02d} {bulan} {now.year}"

    return render_template('home.html', hasil=hasil, hari=hari, tanggal=tanggal_indo)


# ====== HALAMAN PREDIKSI 7 HARI ======
@app.route('/prediksi')
def prediksi():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='cuaca_db'
    )
    query = "SELECT * FROM cuaca_lama ORDER BY tanggal ASC"
    df_bersih = pd.read_sql(query, conn)
    conn.close()

    df_bersih['Tanggal'] = pd.to_datetime(df_bersih['Tanggal'])

    hasil_7hari = get_prediksi_7_hari(df_bersih)
    return render_template('prediksi.html', prediksi=hasil_7hari)


# ====== HALAMAN INPUT DATA ======
@app.route('/input')
def input():
    return render_template('input.html')

# ====== SUBMIT INPUT MANUAL ======
from datetime import timedelta

@app.route('/submit_input', methods=['POST'])
def submit_input():
    cursor.execute("SELECT MAX(tanggal) FROM cuaca_input_user")
    max_tanggal = cursor.fetchone()[0]

    if max_tanggal:
        max_tanggal = pd.to_datetime(max_tanggal)
        next_allowed_date = max_tanggal + timedelta(days=1)
    else:
        next_allowed_date = None  # Jika belum ada data, izinkan tanggal bebas

    # Ambil data dari form
    tanggal_list = request.form.getlist('tanggal[]')
    rh_list = request.form.getlist('rh_avg[]')
    tavg_list = request.form.getlist('tavg[]')
    rr_list = request.form.getlist('rr[]')
    ss_list = request.form.getlist('ss[]')

    df_input = pd.DataFrame({
        'Tanggal': tanggal_list,
        'RH_avg': rh_list,
        'Tavg': tavg_list,
        'RR': rr_list,
        'ss': ss_list
    })

    df_input['Tanggal'] = pd.to_datetime(df_input['Tanggal'], errors='coerce')
    df_bersih = bersihkan_data(df_input.copy())

    if df_bersih.empty:
        flash("Data tidak valid setelah preprocessing!", "danger")
        return redirect(url_for('input'))

    for _, row in df_bersih.iterrows():
        tanggal = row['Tanggal']

        if pd.isna(tanggal):
            flash("❌ Tanggal kosong atau tidak valid.", "danger")
            continue

        # Validasi: hanya boleh 1 hari setelah max_tanggal
        if next_allowed_date and tanggal != next_allowed_date:
            flash(f"⚠️ Hanya boleh input data untuk tanggal {next_allowed_date.date()}. Dilewati: {tanggal.date()}", "warning")
            continue

        # Validasi data numerik
        if any(pd.isna([row['RH_avg'], row['Tavg'], row['RR'], row['ss']])):
            flash(f"⚠️ Baris {tanggal.date()} dilewati karena nilai kosong.", "warning")
            continue

        try:
            cursor.execute("""
                INSERT INTO cuaca_input_user (tanggal, RH_avg, Tavg, RR, ss)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                tanggal.strftime('%Y-%m-%d'),
                float(row['RH_avg']),
                float(row['Tavg']),
                float(row['RR']),
                float(row['ss'])
            ))
            flash(f"✔️ Tersimpan: {tanggal.date()}", "success")
        except Exception as e:
            flash(f"❌ Gagal simpan {tanggal.date()}: {e}", "danger")

    db.commit()
    return redirect(url_for('input'))


# ====== UPLOAD CSV ======
from datetime import timedelta

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        flash("Tidak ada file dipilih", "danger")
        return redirect(url_for('input'))

    file = request.files['file']
    if file.filename == '':
        flash("Nama file kosong", "danger")
        return redirect(url_for('input'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        try:
            df = pd.read_csv(path)
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y', errors='coerce')

            df_bersih = bersihkan_data(df)
            if df_bersih.empty:
                flash("File tidak berisi data valid setelah preprocessing!", "danger")
                return redirect(url_for('input'))

            # Ambil tanggal terakhir dari DB
            cursor.execute("SELECT MAX(tanggal) FROM cuaca_input_user")
            max_tanggal_db = cursor.fetchone()[0]

            # Ambil semua tanggal dari CSV (pastikan sudah ascending)
            tanggal_csv = df_bersih['Tanggal'].dropna().sort_values().tolist()

            # Validasi awal: harus sehari setelah tanggal terakhir di DB
            if max_tanggal_db:
                expected_start = max_tanggal_db + timedelta(days=1)
                if tanggal_csv[0].date() != expected_start:
                    flash(f"❌ Tanggal pertama ({tanggal_csv[0].date()}) harus tepat sehari setelah data terakhir ({max_tanggal_db}).", "danger")
                    return redirect(url_for('input'))

            # Validasi urutan tanggal dalam CSV harus 1 hari berurutan
            for i in range(1, len(tanggal_csv)):
                if tanggal_csv[i].date() != tanggal_csv[i-1].date() + timedelta(days=1):
                    flash(f"❌ Data CSV tidak urut. Tanggal {tanggal_csv[i-1].date()} langsung loncat ke {tanggal_csv[i].date()}. Periksa dataset Anda.", "danger")
                    return redirect(url_for('input'))

            # Proses penyimpanan jika lolos validasi
            for _, row in df_bersih.iterrows():
                tanggal = row['Tanggal']
                if pd.isna(tanggal): continue

                cursor.execute("""
                    INSERT INTO cuaca_input_user (tanggal, RH_avg, Tavg, RR, ss)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    tanggal.strftime('%Y-%m-%d'),
                    float(row['RH_avg']),
                    float(row['Tavg']),
                    float(row['RR']),
                    float(row['ss'])
                ))

            db.commit()
            flash("CSV berhasil diunggah dan disimpan.", "success")

        except Exception as e:
            flash(f"Gagal memproses file CSV: {e}", "danger")
    else:
        flash("Format file tidak didukung. Harus .csv", "danger")

    return redirect(url_for('input'))

# ====== HALAMAN DOWNLOAD ======
@app.route('/download')
def download():
    return render_template('download.html')

# ====== HALAMAN LATIH MODEL ======
@app.route('/latihmodel')
def latihmodel():
    return render_template('latihmodel.html')

# ====== TERAPKAN MODEL ======
@app.route('/terapkan_model', methods=['POST'])
def terapkan_model():
    model_terpilih = request.form.get('model_terpilih')
    if model_terpilih:
        with open('model_aktif.txt', 'w') as f:
            f.write(model_terpilih)
        flash(f"Model '{model_terpilih}' berhasil diterapkan.", 'success')
    return redirect(url_for('latihmodel'))

# ====== JALANKAN FLASK ======
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)