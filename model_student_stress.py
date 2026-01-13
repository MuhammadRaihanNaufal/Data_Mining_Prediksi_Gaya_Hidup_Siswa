import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

# --- 1. Load Data ---
df = pd.read_csv("student_lifestyle_dataset.csv")

# >>> PERUBAHAN UTAMA UNTUK SKALA IPK 0-4 <<<
# Jika nilai GPA lama berada pada skala 0-5, transformasikan ke skala 0-4.
# (Diasumsikan nilai maksimal GPA lama adalah 5.0)
df['GPA'] = df['GPA'] * (4 / 5) 

# --- 2. Preprocessing Data ---

# A. Identifikasi Fitur (X) dan Target (y)
features = [
    'Study_Hours_Per_Day', 
    'Extracurricular_Hours_Per_Day', 
    'Sleep_Hours_Per_Day', 
    'Social_Hours_Per_Day', 
    'Physical_Activity_Hours_Per_Day', 
    'GPA' # Fitur GPA kini dalam skala 0-4
]
X = df[features].values
y = df['Stress_Level'].values

# B. Label Encoding untuk Target (Stress_Level)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Kelas yang terdeteksi: 0=High, 1=Low, 2=Moderate

# C. Normalisasi Fitur (Penting untuk NN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# D. One-hot encode label
y_categorical = to_categorical(y_encoded)

# Simpan scaler, encoder, dan daftar fitur untuk digunakan di aplikasi
joblib.dump(scaler, 'scaler_student_stress.joblib')
joblib.dump(label_encoder, 'label_encoder_student_stress.joblib')
joblib.dump(features, 'feature_columns.joblib') 

# Split data 80:20
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# --- 3. Bangun Model Neural Network ---
input_dim = X_train.shape[1] 
output_dim = y_categorical.shape[1] 

model = Sequential([
    Dense(64, input_shape=(input_dim,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(output_dim, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Latih model
print("Memulai pelatihan model...")
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0) 

# Evaluasi model
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = y_pred_probs.argmax(axis=1)
y_true = y_test.argmax(axis=1)

# Laporan klasifikasi
print(f"\nAkurasi Model: {accuracy_score(y_true, y_pred) * 100:.2f}%")
print("\nLaporan Klasifikasi:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Simpan model
model.save("model_student_stress.h5")
print("✅ Model Neural Network berhasil disimpan sebagai 'model_student_stress.h5'.")