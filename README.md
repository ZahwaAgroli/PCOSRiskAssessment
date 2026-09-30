# PCOS Risk Assesment System Using Naive Bayes Classification

Sistem skrining risiko *Polycystic Ovary Syndrome* (PCOS) dengan **Naive Bayes hibrida** (Gaussian + Bernoulli) yang diimplementasikan dengan NumPy, dalam dua mode: **mandiri** (tanpa USG/lab) dan **klinis** (dengan USG & AMH).

> Alat bantu belajar, **bukan diagnosis medis**.

**Link Aplikasi:** https://pcosriskassessment-projectfinalai.streamlit.app/

**Demo Live:** https://youtu.be/wBs6dOXc9Ig

# Menjalankan
```bash
pip install -r requirements.txt
python clean_data.py       # ringkasan pembersihan data
python eda.py              # grafik EDA (mode klinis & mandiri)
python train_model.py      # validasi silang, ablasi fitur, melatih model final
python calibration.py      # cek kalibrasi probabilitas
python -m pytest -q tests  # verifikasi model.py terhadap scikit-learn
streamlit run app.py       # jalankan aplikasi web
```
**Catatan:** jika `model.py` diubah, jalankan ulang `python train_model.py`
sebelum `streamlit run app.py`, supaya file `model/*.pkl` ikut diperbarui dan
cocok dengan kode terbaru.

# Struktur
| File | Isi |
|---|---|
| `model.py` | **Inti algoritma**: `NaiveBayesPCOS` (fit, predict_proba, explain) |
| `clean_data.py` | Pembersihan data dan definisi set fitur |
| `train_model.py` | *cross validation*, ablasi fitur, melatih dan menyimpan model final |
| `calibration.py` | Cek apakah probabilitas yang dihasilkan model bisa dipercaya |
| `eda.py` | Grafik eksplorasi data |
| `app.py` | Antarmuka aplikasi web (Streamlit) |
| `.streamlit/config.toml` | Tema aplikasi |
| `tests/` | Tes: probabilitas identik dengan `GaussianNB`/`BernoulliNB` scikit-learn |
| `model/*.pkl` | Model yang sudah dilatih, dipakai langsung oleh `app.py` |
| `outputs/` | Grafik dan angka hasil |

## Data
Kaggle *PCOS (without infertility)*: 541 pasien, 10 rumah sakit di Kerala,
India (data asli: Prasoon Kottarathil, `prasoonkottarathil/polycystic-ovary-syndrome-pcos`;
salinan yang dipakai: `shreyasvedpathak/pcos-dataset`).

Isu yang dibersihkan: AMH bernilai `'a'` (1), AMH = 2333 ng/mL (1), `Cycle(R/I)` = 5 (1), `Fast food` kosong (1). Hasil akhir 537 baris (mode klinis) dan 539 baris (mode mandiri).

## Fitur

- **Mode klinis** (13 fitur): butuh hasil USG (jumlah folikel kiri/kanan) dan AMH.
- **Mode mandiri** (10 fitur): tanpa USG/lab, cukup usia, BMI, siklus, dan gejala.

Fitur dipilih berdasarkan kriteria Rotterdam (oligo/anovulasi, hiperandrogenisme
klinis, morfologi ovarium polikistik) dan kekuatan korelasinya terhadap label
PCOS pada dataset.

## Hasil
Validasi silang berulang (stratified 5-fold × 5, seed 42), rerata ± simpangan baku, dalam persen.

| Model | Akurasi | Recall | F1 | AUC |
|---|---|---|---|---|
| Baseline mayoritas | 67.2 ± 0.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 50.0 ± 0.0 |
| GaussianNB scikit-learn (klinis) | 89.3 ± 3.5 | 86.5 ± 7.4 | 84.1 ± 5.2 | 95.6 ± 1.6 |
| **Hybrid NB sendiri (klinis)** | 89.5 ± 2.9 | 85.5 ± 6.3 | 84.2 ± 4.4 | 95.7 ± 1.6 |
| GaussianNB scikit-learn (mandiri) | 82.6 ± 3.4 | 77.4 ± 8.9 | 74.2 ± 5.2 | 87.6 ± 4.4 |
| **Hybrid NB sendiri (mandiri)** | 84.0 ± 3.9 | 74.9 ± 8.8 | 75.2 ± 6.4 | 87.2 ± 4.7 |

## Keterbatasan
- Asumsi independensi Naive Bayes dilanggar (folikel kiri-kanan berkorelasi
0.80)
- Jumlah folikel adalah bagian dari kriteria diagnosis PCOS itu sendiri,
sehingga mode klinis berisiko sebagian "mempelajari ulang" aturan diagnosis.
- Probabilitas yang dihasilkan cenderung terlalu ekstrem.
- Evaluasi hanya berdasarkan satu dataset dari
satu wilayah.
- Sistem ini bukan alat diagnosis dan hasil harus dikonfirmasi tenaga medis.

## Referensi
Kottarathil, P. (2020). *Polycystic ovary syndrome (PCOS)*. Kaggle.












