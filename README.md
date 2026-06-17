# Klasifikasi Diagnosis Kanker Payudara - Machine Learning

Projek ini merupakan pemenuhan tugas UTS Mata Kuliah Pembelajaran Mesin di Universitas Dian Nuswantoro (UDINUS). Projek ini berfokus pada klasifikasi tumor payudara (Malignant/Ganas vs Benign/Jinak) menggunakan karakteristik klinis sel hasil ekstraksi laboratorium.

## 👥 Anggota Tim
* **Nama:** Okan Salma
* **NIM:** A11202416083

## 📊 Dataset
Dataset yang digunakan adalah **Breast Cancer Wisconsin (Diagnostic)** dari UCI Machine Learning Repository yang terdiri dari 569 sampel pasien dan 30 fitur klinis numerik.

## 🤖 Algoritma & Hasil Eksperimen
Eksperimen dilakukan di Google Colab menggunakan teknik 5-Fold Cross Validation dengan hasil sebagai berikut:
1. **Distance-Weighted KNN (DWKNN):** Rata-rata Akurasi 96.26%
2. **Artificial Neural Network (ANN):** Rata-rata Akurasi 97.58% (Model Terbaik dengan Recall Data Uji tertinggi).

## 📂 Struktur File
* `wdbc.data` & `wdbc.names`: Dataset Medis UCI.
* `UTS_ML_Kanker_Payudara.ipynb`: File notebook koding lengkap di Google Colab.
* `model_ann.pkl` & `model_dwknn.pkl`: File biner model yang sudah terlatih.
* `scaler.pkl`: Objek normalisasi data.
