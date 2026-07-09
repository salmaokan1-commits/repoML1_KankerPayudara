import streamlit as st
import numpy as np
import joblib
import os

# 1. OTOMATISASI PATH (Anti-Error Direktori)
# Mendapatkan lokasi folder absolut tempat app.py ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Menyusun beberapa kemungkinan jalur folder (mengantisipasi huruf besar/kecil & struktur folder)
path_subfolder_caps = os.path.join(BASE_DIR, 'repoML1_KankerPayudara')
path_subfolder_low = os.path.join(BASE_DIR, 'repoml1_kankerpayudara')

# Mencari di mana file scaler.pkl berada secara dinamis
if os.path.exists(os.path.join(path_subfolder_caps, 'scaler.pkl')):
    scaler_path = os.path.join(path_subfolder_caps, 'scaler.pkl')
    model_path = os.path.join(path_subfolder_caps, 'model_ann.pkl')
elif os.path.exists(os.path.join(path_subfolder_low, 'scaler.pkl')):
    scaler_path = os.path.join(path_subfolder_low, 'scaler.pkl')
    model_path = os.path.join(path_subfolder_low, 'model_ann.pkl')
elif os.path.exists(os.path.join(BASE_DIR, 'scaler.pkl')):
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    model_path = os.path.join(BASE_DIR, 'model_ann.pkl')
else:
    st.error(f"❌ File model (.pkl) tidak ditemukan di repositori! Lokasi saat ini: {BASE_DIR}")
    st.stop()

# 2. LOAD COMPONEN MODEL YANG SUDAH AMAN PATH-NYA
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
