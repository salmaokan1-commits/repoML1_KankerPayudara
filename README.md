# Klasifikasi Diagnosis Kanker Payudara - Machine Learning

Projek ini merupakan pemenuhan tugas UTS Mata Kuliah Pembelajaran Mesin di Universitas Dian Nuswantoro (UDINUS). Projek ini berfokus pada klasifikasi tumor payudara (Malignant/Ganas vs Benign/Jinak) menggunakan karakteristik klinis sel hasil ekstraksi laboratorium.

## 📂 Struktur Repositori
* `repoML1_KankerPayudara/Projek_Kanker_Payudara_ML.ipynb` : Skrip eksperimen utama (Google Colab).
* `repoML1_KankerPayudara/model_ann.pkl` : File biner model terbaik (Artificial Neural Network).
* `repoML1_KankerPayudara/model_dwknn.pkl` : File biner model pembanding (DWKNN).
* `repoML1_KankerPayudara/scaler.pkl` : Objek standardisasi fitur (StandardScaler).
* `repoML1_KankerPayudara/wdbc.data` & `.names` : Dataset Medis UCI.
* `app.py` : Skrip antarmuka inferensi aplikasi web berbasis Streamlit.
* `requirements.txt` : Daftar pustaka dependensi sistem.

## 👥 Anggota Tim
* **Nama:** Okan Salma
* **NIM:** A11202416083

## 📊 Dataset
Dataset yang digunakan adalah **Breast Cancer Wisconsin (Diagnostic)** dari UCI Machine Learning Repository yang terdiri dari 569 sampel pasien dan 30 fitur klinis numerik.

## 🤖 Algoritma & Hasil Eksperimen
Eksperimen dilakukan di Google Colab menggunakan teknik 5-Fold Cross Validation dengan hasil sebagai berikut:
1. **Distance-Weighted KNN (DWKNN):** Rata-rata Akurasi 96.26%
2. **Artificial Neural Network (ANN):** Rata-rata Akurasi 97.58% (Model Terbaik dengan Recall Data Uji tertinggi).

## 🚀 Cara Menjalankan Aplikasi Web Secara Lokal
1. Clone repositori ini:
   ```bash
   git clone [https://github.com/salmaokan1-commits/repoML1_KankerPayudara.git](https://github.com/salmaokan1-commits/repoML1_KankerPayudara.git)
