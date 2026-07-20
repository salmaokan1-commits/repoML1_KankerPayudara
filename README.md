# 🎗️ Aplikasi Klasifikasi Kanker Payudara (Multi-Page Streamlit App)

Repositori ini berisi proyek akhir Machine Learning untuk mendeteksi/mengklasifikasikan kanker payudara (Malignant/Benign) menggunakan beberapa algoritma klasifikasi. Aplikasi ini dideploy menggunakan Streamlit Cloud.

## 📊 Dataset
Proyek ini menggunakan **Breast Cancer Wisconsin (Diagnostic) Dataset** (`wdbc.data` & `wdbc.names`). Fitur data diekstraksi dari gambar digital *Fine Needle Aspirate* (FNA) dari massa payudara.

## 🧠 Algoritma & Performa Model
Eksperimen dilakukan menggunakan tiga algoritma utama dengan hasil sebagai berikut:
* **XGBoost (Model Utama):** Akurasi ~XX% (Performa Terbaik)
* **Artificial Neural Network (ANN):** Akurasi ~XX%
* **DWKNN (Dual Weighted KNN):** Akurasi ~XX%

---

## 📂 Struktur Repositori
```text
├── app.py                     # Script utama aplikasi Streamlit
├── requirements.txt           # Dependensi library python
├── README.md                  # Dokumentasi proyek
├── .gitignore                 # Mengabaikan file sampah lokal
├── Projek_Kanker_Payudara_ML.ipynb  # Notebook dokumentasi EDA & Modeling
├── wdbc.data                  # Dataset mentah
├── wdbc.names                 # Informasi atribut dataset
├── scaler.pkl                 # Hasil penyeragaman skala (StandardScaler)
├── model_xgb.pkl              # Binary model XGBoost
├── model_ann.pkl              # Binary model ANN
└── model_dwknn.pkl            # Binary model DWKNN
