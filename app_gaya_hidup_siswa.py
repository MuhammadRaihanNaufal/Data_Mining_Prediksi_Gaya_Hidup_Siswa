from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# --- Muat Aset yang Diperlukan ---
try:
    # Memuat model, scaler, dan daftar fitur yang benar
    model = load_model('model_student_stress.h5')
    scaler = joblib.load('scaler_student_stress.joblib')
    label_encoder = joblib.load('label_encoder_student_stress.joblib')
    feature_names = joblib.load('feature_columns.joblib')
except Exception as e:
    print(f"Error loading assets. Pastikan Anda sudah menjalankan model_student_stress.py: {e}")
    # Jika file tidak ditemukan, aplikasi tidak bisa berjalan
    exit()

# --- Definisi Pertanyaan dan Opsi ---
# Hanya 6 input numerik
kuisioner = [
    {'name': 'Study_Hours_Per_Day', 'question': 'Rata-rata jam belajar per hari?', 'min': 0, 'max': 12, 'unit': 'jam'},
    {'name': 'Extracurricular_Hours_Per_Day', 'question': 'Rata-rata jam kegiatan ekstrakurikuler per hari?', 'min': 0, 'max': 5, 'unit': 'jam'},
    {'name': 'Sleep_Hours_Per_Day', 'question': 'Rata-rata jam tidur per hari?', 'min': 4, 'max': 12, 'unit': 'jam'},
    {'name': 'Social_Hours_Per_Day', 'question': 'Rata-rata jam bersosialisasi per hari?', 'min': 0, 'max': 8, 'unit': 'jam'},
    {'name': 'Physical_Activity_Hours_Per_Day', 'question': 'Rata-rata jam aktivitas fisik/olahraga per hari?', 'min': 0, 'max': 4, 'unit': 'jam'},
    {'name': 'GPA', 'question': 'Rata-rata nilai/IPK Anda saat ini (skala 0-4)?', 'min': 0, 'max': 4, 'step': 0.01, 'unit': 'poin'},
]

@app.route("/", methods=["GET"])
def index():
    # Menggunakan template baru
    return render_template("index_student_stress.html", kuisioner=kuisioner)


@app.route('/predict', methods=['POST'])
def predict():
    input_data = []

    # Mengumpulkan 6 input numerik
    for item in kuisioner:
        name = item['name']
        val = request.form.get(name)
        try:
            # Mengubah input form (string) menjadi float
            input_data.append(float(val))
        except (ValueError, TypeError):
            # Asumsi default 0 jika ada error input
            input_data.append(0.0) 

    # Siapkan input untuk model (ubah ke array 2D)
    X_input = np.array([input_data])

    # Scaling input menggunakan scaler yang sudah dilatih
    X_scaled = scaler.transform(X_input)

    # Prediksi
    pred_probs = model.predict(X_scaled, verbose=0)
    pred_index = np.argmax(pred_probs)
    
    # Konversi index ke label asli (High, Low, Moderate)
    predicted_stress_level_raw = label_encoder.inverse_transform([pred_index])[0] 
    
    # Mapping output agar lebih mudah dibaca
    if predicted_stress_level_raw == 'Low':
        result_label = 'Tingkat Stres Rendah (Low)'
    elif predicted_stress_level_raw == 'Moderate':
        result_label = 'Tingkat Stres Sedang (Moderate)'
    else: # 'High'
        result_label = 'Tingkat Stres Tinggi (High)'

    # Menggunakan template baru
    return render_template('result_student_stress.html', 
                           result=predicted_stress_level_raw, 
                           result_label=result_label)

if __name__ == '__main__':
    # Pastikan Anda sudah menjalankan model_student_stress.py yang baru
    app.run(debug=True, port=5001)