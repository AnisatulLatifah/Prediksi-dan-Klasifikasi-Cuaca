from flask import Flask, render_template, request, redirect, url_for, flash, send_file, make_response
import pandas as pd
from datetime import date, timedelta
import mysql.connector
import os
import sys
import pdfkit
from werkzeug.utils import secure_filename
from model.get_prediksi_klasifikasi import get_prediksi_7_hari
import os
from io import BytesIO
from jinja2 import Environment, FileSystemLoader
from model.get_prediksi_klasifikasi import get_prediksi_klasifikasi_hari_ini

# utils
sys.path.append('utils')
from utils.export_excel_template import isi_template_excel


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

def get_last_data_date():
    conn = mysql.connector.connect(host='localhost', user='root', password='', database='cuaca_db')
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(tanggal) FROM cuaca_input_user")
    result_input = cursor.fetchone()[0]
    if result_input:
        conn.close()
        return pd.to_datetime(result_input)
    cursor.execute("SELECT MAX(tanggal) FROM cuaca_lama")
    result_lama = cursor.fetchone()[0]
    conn.close()
    return pd.to_datetime(result_lama)

# ====== HALAMAN HOME ======
@app.route('/')
def home():
    import locale
    try:
        locale.setlocale(locale.LC_TIME, 'id_ID.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_TIME, 'ind')
        except:
            pass

    # Ambil data dari database langsung
    conn = mysql.connector.connect(host='localhost', user='root', password='', database='cuaca_db')
    df_lama = pd.read_sql("SELECT * FROM cuaca_lama", conn)
    df_user = pd.read_sql("SELECT * FROM cuaca_input_user", conn)
    conn.close()

    # Gabungkan dan urutkan
    df_bersih = pd.concat([df_lama, df_user], ignore_index=True)
    df_bersih['Tanggal'] = pd.to_datetime(df_bersih['Tanggal'])
    df_bersih = df_bersih.sort_values('Tanggal')

    # Ambil tanggal terakhir
    last_date = df_bersih['Tanggal'].max()
    prediksi_date = last_date + timedelta(days=1)

    # Jalankan prediksi hari ini
    hasil = get_prediksi_klasifikasi_hari_ini(df_bersih, last_date)

    hari = prediksi_date.strftime('%A')
    bulan = prediksi_date.strftime('%B')
    tanggal_indo = f"{prediksi_date.day:02d} {bulan} {prediksi_date.year}"
    last_data_str = f"{last_date.day:02d} {bulan} {last_date.year}"

    return render_template(
        'home.html',
        hasil=hasil,
        hari=hari,
        tanggal=tanggal_indo,
        last_data_str=last_data_str
    )

# ====== HALAMAN PREDIKSI ======
@app.route('/prediksi')
def prediksi():

    # Ambil data & last_date sesuai model aktif
    conn = mysql.connector.connect(host='localhost', user='root', password='', database='cuaca_db')
    df_lama = pd.read_sql("SELECT * FROM cuaca_lama", conn)
    df_input = pd.read_sql("SELECT * FROM cuaca_input_user", conn)
    conn.close()

    df_bersih = pd.concat([df_lama, df_input], ignore_index=True)
    df_bersih['Tanggal'] = pd.to_datetime(df_bersih['Tanggal'])
    df_bersih = df_bersih.sort_values('Tanggal')

    last_data_date = df_bersih['Tanggal'].max()

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
            flash("✅ Data CSV berhasil diunggah dan disimpan.", "success")
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
    conn = mysql.connector.connect(host='localhost', user='root', password='', database='cuaca_db')
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(tanggal) FROM cuaca_input_user")
    result_input = cursor.fetchone()[0]

    if result_input:
        last_date = pd.to_datetime(result_input).date()
        table_used = "cuaca_input_user"
    else:
        cursor.execute("SELECT MAX(tanggal) FROM cuaca_lama")
        last_date = pd.to_datetime(cursor.fetchone()[0]).date()
        table_used = "cuaca_lama"

    # Ambil data dari dua tabel
    df_lama = pd.read_sql("SELECT * FROM cuaca_lama", conn)
    df_input = pd.read_sql("SELECT * FROM cuaca_input_user", conn)
    conn.close()

    df_bersih = pd.concat([df_lama, df_input], ignore_index=True)
    df_bersih['Tanggal'] = pd.to_datetime(df_bersih['Tanggal'])
    df_bersih = df_bersih.sort_values('Tanggal')

    hasil = get_prediksi_7_hari(df_bersih, last_date)

    for row in hasil:
        row["Klasifikasi"] = row.get("klasifikasi", "-")

    periode = f"{hasil[0]['tanggal']} - {hasil[-1]['tanggal']}"
    deskripsi = [buat_keterangan(row) for row in hasil]

    logo_path = os.path.abspath("static/images/logo_website.png").replace("\\", "/")
    logo_url = "file:///" + logo_path

    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('report_template.html')
    html_out = template.render(data=hasil, periode=periode, deskripsi=deskripsi, logo_path=logo_url)

    config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
    options = {'enable-local-file-access': None}
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


# ====== JALANKAN FLASK ======
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)