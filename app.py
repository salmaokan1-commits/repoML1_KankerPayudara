import streamlit as st
import numpy as np
import joblib

# Memuat komponen model dari dalam sub-folder repositori kamu
scaler = joblib.load('repoML1_KankerPayudara/scaler.pkl')
model = joblib.load('repoML1_KankerPayudara/model_ann.pkl')

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
