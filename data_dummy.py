from datetime import date, timedelta

def generate_dummy_data():
    base = date.today()
    data = []

    for i in range(7):
        d = base + timedelta(days=i)
        data.append({
            "Tanggal": d.strftime("%d-%m-%Y"),
            "RH_avg": 80 - i,
            "Tavg": 28.5 - i*0.2,
            "RR": round(1.5 + i*0.5, 1),
            "SS": round(7.5 - i*0.6, 1),
            "Klasifikasi": "Baik" if i < 5 else "Buruk"
        })
    
    return data
