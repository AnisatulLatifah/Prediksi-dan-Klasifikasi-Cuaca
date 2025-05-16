from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from datetime import datetime
from datetime import timedelta
import mysql.connector
import os
from werkzeug.utils import secure_filename  # ⬅️ penting
from model.train_model import train_and_save_all

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
            flash(f"✅ Tersimpan: {tanggal.date()}, Model terbaru berhasil diretrain!", "success")
        except Exception as e:
            flash(f"❌ Gagal simpan {tanggal.date()}: {e}", "danger")

    db.commit()

    train_and_save_all()
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
            flash("✅ CSV berhasil diunggah dan disimpan. Model terbaru berhasil diretrain!", "success")
            train_and_save_all()

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
import os
import mysql.connector
from flask import render_template, request, redirect, url_for, flash

def get_last_date_from(table_name):
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='cuaca_db'
        )
        cursor = conn.cursor()
        cursor.execute(f"SELECT MAX(Tanggal) FROM `{table_name}`")
        result = cursor.fetchone()[0]
        conn.close()
        return result.strftime("%Y-%m-%d") if result else "-"
    except Exception as e:
        print(f"DB error on {table_name}:", e)
        return "-"

@app.route('/latihmodel', methods=['GET'])
def latihmodel():
    model_list = []

    # Model utama (cuaca_lama)
    model_list.append({
        "versi": 1,
        "nama": "model utama.pkl",
        "path": "model/model_utama",
        "tgl_data": get_last_date_from("cuaca_lama"),
        "tgl_retrain": "2025-05-01"
    })

    # Model retrain (cuaca_input_user)
    retrain_dir = os.path.join("model", "model_retrain")
    if os.path.exists(retrain_dir):
        subdirs = sorted([d for d in os.listdir(retrain_dir) if os.path.isdir(os.path.join(retrain_dir, d))])
        for i, folder in enumerate(subdirs):
            folder_path = os.path.join(retrain_dir, folder).replace("\\", "/")

            tanggal_retrain = "-"
            tanggal_file = os.path.join(folder_path, "tanggal_retrain.txt")
            if os.path.exists(tanggal_file):
                with open(tanggal_file) as f:
                    tanggal_retrain = f.read().strip()

            model_list.append({
                "versi": i + 2,
                "nama": f"{folder}.pkl",
                "path": folder_path,
                "tgl_data": get_last_date_from("cuaca_input_user"),
                "tgl_retrain": tanggal_retrain
            })

    # Model aktif
    aktif_path_file = os.path.join("model", "model_aktif.txt")
    default_path = "model/model_utama"
    if not os.path.exists(aktif_path_file) or os.path.getsize(aktif_path_file) == 0:
        with open(aktif_path_file, 'w') as f:
            f.write(default_path)

    try:
        with open(aktif_path_file, "r") as f:
            aktif_path = f.read().strip()
    except:
        aktif_path = default_path

    return render_template("latihmodel.html", models=model_list, aktif_path=aktif_path)


# ====== TERAPKAN MODEL ======
@app.route('/terapkan_model', methods=['POST'])
def terapkan_model():
    model_terpilih = request.form.get('model_terpilih')
    if model_terpilih:
        path_file = os.path.join("model", "model_aktif.txt")
        with open(path_file, 'w') as f:
            f.write(model_terpilih.strip())

        model_name = model_terpilih.strip().split("/")[-1]
        flash(f"Model '{model_name}' berhasil diterapkan. Silahkan lihat prediksi dan klasifikasinya di halaman home dan prediksi!", 'success')

    return redirect(url_for('latihmodel'))


# ====== JALANKAN FLASK ======
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)