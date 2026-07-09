import streamlit as st
import numpy as np
import joblib
import os

# 1. DETEKSI OTOMATIS SECARA MENDALAM (Recursive Smart Scan)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def cari_file_otomatis(daftar_nama_kemungkinan):
    """Fungsi untuk menyisir semua folder demi menemukan file model"""
    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if f.strip() in daftar_nama_kemungkinan:
                return os.path.join(root, f)
    return None

# Mencari file secara fleksibel (mengantisipasi jika file belum atau sudah di-rename)
scaler_path = cari_file_otomatis(['scaler.pkl', 'scaler (1).pkl', 'scaler(1).pkl'])
model_path = cari_file_otomatis(['model_ann.pkl', 'model_ann (1).pkl', 'model_ann(1).pkl'])

# Validasi jika file benar-benar tidak ada sama sekali di seluruh repositori
if not scaler_path or not model_path:
    st.error("❌ Komponen model ML (.pkl) tidak ditemukan di repositori GitHub kamu!")
    st.write("Berikut adalah daftar seluruh file yang terdeteksi di repositori kamu saat ini:")
    
    # Menampilkan daftar file asli untuk membantu troubleshooting
    list_file = []
    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            list_file.append(os.path.relpath(os.path.join(root, f), BASE_DIR))
    st.code("\n".join(list_file))
    st.stop()

# 2. LOAD COMPONEN MODEL YANG SUDAH BERHASIL DITEMUKAN
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# 3. ANTARMUKA APLIKASI WEB STREAMLIT
st.title("🔬 Breast Cancer Diagnostic System (ANN)")
st.write("Aplikasi cerdas penunjang keputusan klinis diagnosis kanker payudara.")

st.sidebar.header("Input Parameter Sel Hasil Biopsi")

# Antarmuka input data klinis sederhana
radius = st.sidebar.slider("Radius Mean", 6.0, 30.0, 14.0)
texture = st.sidebar.slider("Texture Mean", 9.0, 40.0, 20.0)
perimeter = st.sidebar.slider("Perimeter Mean", 43.0, 190.0, 92.0)
area = st.sidebar.slider("Area Mean", 143.0, 2500.0, 650.0)
smoothness = st.sidebar.slider("Smoothness Mean", 0.05, 0.16, 0.09)

# Menyamakan dimensi array 30 fitur sesuai standar input training
features = [radius, texture, perimeter, area, smoothness] + [0.0] * 25
features_array = np.array([features])

if st.button("Prediksi Hasil Diagnosis"):
    # Penskalaan data masukan menggunakan StandardScaler
    scaled_features = scaler.transform(features_array)
    # Prediksi kelas target menggunakan arsitektur jaringan saraf tiruan
    prediction = model.predict(scaled_features)
    
    st.subheader("Hasil Analisis Sistem:")
    if prediction[0] == 1:
        st.error("🚨 HASIL PREDIKSI: MALIGNANT (GANAS) - Risiko Tinggi / Segera Rujuk!")
    else:
        st.success("🟢 HASIL PREDIKSI: BENIGN (JINAK) - Risiko Rendah")
