from flask import Flask, render_template, request, redirect, url_for, flash, send_file, make_response
import pandas as pd
from datetime import date, timedelta
import mysql.connector
import os
import sys
import pdfkit
from werkzeug.utils import secure_filename
from model.train_model import train_and_save_all
from model.get_prediksi_klasifikasi import get_prediksi_7_hari
from flask import Flask, render_template, send_file
import os
from io import BytesIO
from data_dummy import generate_dummy_data
from jinja2 import Environment, FileSystemLoader

# utils
sys.path.append('utils')
from utils.export_excel_template import isi_template_excel


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

def get_last_data_date_by_model(path_model):
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

    return pd.to_datetime(result)

# ====== HALAMAN HOME ======
@app.route('/')
def home():
    from datetime import date, timedelta
    import locale

    # Set locale ke Bahasa Indonesia (pastikan sistem mendukung)
    try:
        locale.setlocale(locale.LC_TIME, 'id_ID.UTF-8')  # Linux/macOS
    except:
        try:
            locale.setlocale(locale.LC_TIME, 'ind')  # Windows
        except:
            pass  # Lewatkan jika locale tidak tersedia

    # Ambil model yang sedang aktif
    with open("model/model_aktif.txt", "r") as f:
        path_model = f.read().strip()

    # Ambil tanggal terakhir sesuai sumber data model
    def get_last_data_date_by_model(path_model):
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

        if result is None:
            return None
        return pd.to_datetime(result)

    # Tangani jika last_date None
    last_date = get_last_data_date_by_model(path_model)
    if last_date is None:
        last_date = date.today()

    prediksi_date = last_date + timedelta(days=1)

    # Ambil data untuk model
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

    # Prediksi hari ini
    hasil = get_prediksi_klasifikasi_hari_ini(df_bersih, last_date)

    # Gunakan hari dan bulan langsung dari locale Indonesia
    hari = prediksi_date.strftime('%A')  # e.g., "Rabu"
    bulan = prediksi_date.strftime('%B')  # e.g., "Mei"
    tanggal_indo = f"{prediksi_date.day:02d} {bulan} {prediksi_date.year}"

    last_data_str = f"{last_date.day:02d} {last_date.strftime('%B')} {last_date.year}"

    return render_template('home.html',
                           hasil=hasil,
                           hari=hari,
                           tanggal=tanggal_indo,
                           last_data_str=last_data_str)

# ====== HALAMAN PREDIKSI ======
@app.route('/prediksi')
def prediksi():
    from datetime import date
    import pandas as pd
    from model.get_prediksi_klasifikasi import (
        get_prediksi_7_hari,
        get_df_bersih_by_model,
        get_last_date_by_model
    )

    # Ambil data & last_date sesuai model aktif
    df_bersih = get_df_bersih_by_model()
    last_data_date = get_last_date_by_model()

    hasil_7hari = get_prediksi_7_hari(df_bersih, last_data_date)

    tanggal_awal = hasil_7hari[0]['tanggal']
    tanggal_akhir = hasil_7hari[-1]['tanggal']
    range_prediksi = f"{tanggal_awal} hingga {tanggal_akhir}"

    def format_tanggal(tgl):
        bulan_indonesia = {
            'January': 'Januari', 'February': 'Februari', 'March': 'Maret', 'April': 'April',
            'May': 'Mei', 'June': 'Juni', 'July': 'Juli', 'August': 'Agustus',
            'September': 'September', 'October': 'Oktober', 'November': 'November', 'December': 'Desember'
        }
        nama_bulan_inggris = tgl.strftime('%B')
        bulan = bulan_indonesia.get(nama_bulan_inggris, nama_bulan_inggris)
        return f"{tgl.day:02d} {bulan} {tgl.year}"

    last_data_str = format_tanggal(last_data_date)

    return render_template('prediksi.html',
        prediksi=hasil_7hari,
        range_prediksi=range_prediksi,
        last_data_str=last_data_str
    )


# ====== HALAMAN INPUT DATA ======
@app.route('/input')
def input():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='cuaca_db'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(tanggal) FROM cuaca_input_user")
    last_date = cursor.fetchone()[0]
    conn.close()

    import locale
    locale.setlocale(locale.LC_TIME, 'id_ID.UTF-8')
    formatted_date = last_date.strftime('%d %B %Y') if last_date else 'Belum ada data'
    
    return render_template('input.html', last_date=formatted_date)


# ====== SUBMIT INPUT MANUAL ======
@app.route('/submit_input', methods=['POST'])
def submit_input():
    cursor.execute("SELECT MAX(tanggal) FROM cuaca_input_user")
    max_tanggal = cursor.fetchone()[0]

    if max_tanggal:
        max_tanggal = pd.to_datetime(max_tanggal)
        next_allowed_date = max_tanggal + timedelta(days=1)
    else:
        next_allowed_date = None

    tanggal_list = request.form.getlist('tanggal[]')
    rh_list = request.form.getlist('rh_avg[]')
    tavg_list = request.form.getlist('tavg[]')
    rr_list = request.form.getlist('rr[]')
    ss_list = request.form.getlist('ss[]')

    df_input = pd.DataFrame({
        'tanggal': tanggal_list,
        'RH_avg': rh_list,
        'Tavg': tavg_list,
        'RR': rr_list,
        'ss': ss_list
    })

    df_input['tanggal'] = pd.to_datetime(df_input['tanggal'], errors='coerce')
    df_lama = pd.read_sql("SELECT * FROM cuaca_lama", db)
    df_user = pd.read_sql("SELECT * FROM cuaca_input_user", db)
    df_full = pd.concat([df_lama, df_user, df_input], ignore_index=True)
    df_full['Tanggal'] = pd.to_datetime(df_full['Tanggal'])

    df_bersih_full = bersihkan_data(df_full)
    df_bersih = df_bersih_full[df_bersih_full['Tanggal'].isin(df_input['tanggal'])]


    if df_bersih.empty:
        flash("Data tidak valid setelah preprocessing!", "danger")
        return redirect(url_for('input'))

    for _, row in df_bersih.iterrows():
        tanggal = row['tanggal']
        if pd.isna(tanggal):
            flash("❌ Tanggal kosong atau tidak valid.", "danger")
            continue

        if next_allowed_date and tanggal != next_allowed_date:
            flash(f"⚠️ Hanya boleh input data untuk tanggal {next_allowed_date.date()}. Dilewati: {tanggal.date()}", "warning")
            continue

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
            flash(f"✅ Tersimpan: {tanggal.date()}", "success")
        except Exception as e:
            flash(f"❌ Gagal simpan {tanggal.date()}: {e}", "danger")

    db.commit()
    train_and_save_all()
    return render_template('input.html')  


# ====== UPLOAD CSV ======
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
            # Baca file
            df_csv = pd.read_csv(path)
            df_csv['Tanggal'] = pd.to_datetime(df_csv['Tanggal'], format='%d-%m-%Y', errors='coerce')

            # Gabungkan dengan data lama + input user
            df_lama = pd.read_sql("SELECT * FROM cuaca_lama", db)
            df_user = pd.read_sql("SELECT * FROM cuaca_input_user", db)
            df_full = pd.concat([df_lama, df_user, df_csv], ignore_index=True)
            df_full['Tanggal'] = pd.to_datetime(df_full['Tanggal'])

            # Bersihkan seluruh data
            df_bersih_full = bersihkan_data(df_full)

            # Ambil hanya baris hasil dari input CSV
            tanggal_input = pd.to_datetime(df_csv['Tanggal'])
            df_bersih = df_bersih_full[df_bersih_full['Tanggal'].isin(tanggal_input)]

            if df_bersih.empty:
                flash("File tidak berisi data valid setelah preprocessing!", "danger")
                return redirect(url_for('input'))

            # Validasi tanggal
            cursor.execute("SELECT MAX(tanggal) FROM cuaca_input_user")
            max_tanggal_db = cursor.fetchone()[0]

            tanggal_csv = df_bersih['Tanggal'].dropna().sort_values().tolist()
            if max_tanggal_db:
                expected_start = max_tanggal_db + timedelta(days=1)
                if tanggal_csv[0].date() != expected_start:
                    flash(f"❌ Tanggal pertama ({tanggal_csv[0].date()}) harus tepat sehari setelah data terakhir ({max_tanggal_db}).", "danger")
                    return redirect(url_for('input'))

            for i in range(1, len(tanggal_csv)):
                if tanggal_csv[i].date() != tanggal_csv[i-1].date() + timedelta(days=1):
                    flash(f"❌ Data CSV tidak urut. Tanggal {tanggal_csv[i-1].date()} langsung loncat ke {tanggal_csv[i].date()}.", "danger")
                    return redirect(url_for('input'))

            # Simpan ke database
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
            train_and_save_all()
            flash("✅ CSV berhasil disimpan dan model berhasil dilatih!", "success")
            return redirect(url_for('input'))

        except Exception as e:
            flash(f"Gagal memproses file CSV: {e}", "danger")
            return redirect(url_for('input'))
    else:
        flash("Format file tidak didukung. Harus .csv", "danger")
        return redirect(url_for('input'))


# ====== HALAMAN DOWNLOAD ======
@app.route('/download')
def download():
    return render_template('download.html')


# ====== HALAMAN DOWNLOAD EXCEL ======
@app.route('/download_excel')
def download_excel():
    # Ambil data dari DB
    conn = mysql.connector.connect(host='localhost', user='root', password='', database='cuaca_db')
    df_lama = pd.read_sql("SELECT * FROM cuaca_lama", conn)
    df_input = pd.read_sql("SELECT * FROM cuaca_input_user", conn)
    conn.close()

    df_bersih = pd.concat([df_lama, df_input], ignore_index=True)
    df_bersih['Tanggal'] = pd.to_datetime(df_bersih['Tanggal'])
    df_bersih = df_bersih.sort_values('Tanggal')

    # Ambil prediksi 7 hari ke depan
    hasil_7hari = get_prediksi_7_hari(df_bersih, df_bersih['Tanggal'].max().date())
    df_pred = pd.DataFrame(hasil_7hari)

    # Isi template
    isi_template_excel(df_pred)

    # Kirim file ke user
    return send_file("static/hasil/data_prediksi.xlsx", as_attachment=True)


# ====== HALAMAN DOWNLOAD PDF ======
def buat_keterangan(row):
    deskripsi = [f"<strong>📌 {row['tanggal']}</strong>", "<ul>"]

    tavg = row["Tavg"]
    rh = row["RH_avg"]
    rr = row["RR"]
    ss = row["ss"]  # pakai huruf kecil

    if 25 <= tavg <= 30:
        deskripsi.append(f"<li>Suhu optimal di {tavg}°C, mendukung pertumbuhan.</li>")
    else:
        deskripsi.append(f"<li>Suhu {tavg}°C kurang ideal.</li>")

    if 50 <= rh <= 85:
        deskripsi.append(f"<li>Kelembapan ideal ({rh}%).</li>")
    else:
        deskripsi.append(f"<li>Kelembapan {rh}%, perlu perhatian.</li>")

    if rr > 8:
        deskripsi.append(f"<li>Curah hujan tinggi ({rr} mm), potensi genangan.</li>")
    else:
        deskripsi.append(f"<li>Curah hujan rendah ({rr} mm).</li>")

    if ss < 4:
        deskripsi.append(f"<li>Penyinaran rendah ({ss} jam).</li>")
    else:
        deskripsi.append(f"<li>Penyinaran cukup ({ss} jam).</li>")

    deskripsi.append("</ul>")
    return '\n'.join(deskripsi)


@app.route('/cetak-pdf')
def cetak_pdf():
    from jinja2 import Environment, FileSystemLoader
    import os

    # Ambil model aktif
    with open("model/model_aktif.txt", "r") as f:
        path_model = f.read().strip()

    table = "cuaca_lama" if "model_utama" in path_model else "cuaca_input_user"

    # Ambil tanggal terakhir dari tabel
    conn = mysql.connector.connect(host='localhost', user='root', password='', database='cuaca_db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT MAX(tanggal) FROM {table}")
    last_date = cursor.fetchone()[0]
    last_date = pd.to_datetime(last_date).date()

    # Ambil dan gabungkan data dari dua tabel
    df_lama = pd.read_sql("SELECT * FROM cuaca_lama", conn)
    df_input = pd.read_sql("SELECT * FROM cuaca_input_user", conn)
    conn.close()

    if 'model_utama' in path_model:
        df_bersih = df_lama.copy()
    else:
        df_bersih = pd.concat([df_lama, df_input], ignore_index=True)

    df_bersih['Tanggal'] = pd.to_datetime(df_bersih['Tanggal'])
    df_bersih = df_bersih.sort_values('Tanggal')


    # Ambil hasil prediksi dari model
    hasil = get_prediksi_7_hari(df_bersih, last_date)

    # Tambahkan kolom Klasifikasi agar bisa dirender di HTML
    for row in hasil:
        row["Klasifikasi"] = row.get("klasifikasi", "-")

    # Deskripsi + periode
    periode = f"{hasil[0]['tanggal']} - {hasil[-1]['tanggal']}"
    deskripsi = [buat_keterangan(row) for row in hasil]

    # ==== Ambil path absolut logo ====
    logo_path = os.path.abspath("static/images/logo_website.png").replace("\\", "/")
    logo_url = "file:///" + logo_path

    # Render template
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('report_template.html')
    html_out = template.render(data=hasil, periode=periode, deskripsi=deskripsi, logo_path=logo_url)

    # Konfigurasi dan generate PDF
    config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
    options = {
    'enable-local-file-access': None 
}
    pdf_bytes = pdfkit.from_string(html_out, False, configuration=config, options=options)

    return send_file(BytesIO(pdf_bytes), as_attachment=True, download_name="laporan_prediksi.pdf", mimetype='application/pdf')


# ====== HALAMAN LATIH MODEL ======
def get_last_date_from(table):
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='cuaca_db',
        port=3306
    )
    cursor = conn.cursor()
    cursor.execute(f"SELECT MAX(tanggal) FROM {table}")
    result = cursor.fetchone()[0]
    conn.close()

    if result is None:
        return "-"
    return pd.to_datetime(result).strftime('%Y-%m-%d')


# ====== HALAMAN LATIH MODEL ======
@app.route('/latihmodel', methods=['GET'])
def latihmodel():
    model_list = []

    # === 1. Model utama
    model_list.append({
        "versi": 1,
        "nama": "model utama",
        "path": "model/model_utama",
        "tgl_data": get_last_date_from("cuaca_lama"),
        "tgl_retrain": "2025-05-01"
    })

    # === 2. Model retrain dari folder
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
                "nama": f"{folder}",
                "path": folder_path,
                "tgl_data": get_last_date_from("cuaca_input_user"),
                "tgl_retrain": tanggal_retrain
            })

    # === 3. Tambahkan kolom datetime untuk pengurutan
    for model in model_list:
        try:
            model["tgl_retrain_dt"] = pd.to_datetime(model["tgl_retrain"])
        except:
            model["tgl_retrain_dt"] = pd.Timestamp("1900-01-01")

    model_list = sorted(model_list, key=lambda x: x["tgl_retrain_dt"], reverse=True)

    # === 4. Ambil model aktif dari file model_aktif.txt
    aktif_path = ""
    model_aktif_file = os.path.join("model", "model_aktif.txt")
    if os.path.exists(model_aktif_file):
        with open(model_aktif_file) as f:
            aktif_path = f.read().strip()

    # === 5. Tandai model aktif
    for model in model_list:
        model["disabled"] = model["path"] != aktif_path

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