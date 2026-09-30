"""
Hybrid Naive Bayes untuk PCOScreen diimplementasikan sendiri dari nol dengan NumPy
tanpa scikit-learn, dengan for-loop eksplisit supaya mudah ditelusuri baris per baris.

Teorema Bayes :
    P(kelas | gejala pasien) ~ P(kelas) x P(gejala pasien | kelas)

"Naive" = setiap fitur (gejala/hasil lab) dianggap saling bebas satu sama lain,
sehingga peluang gabungan tinggal dikalikan satu-satu:
    P(x1, x2, ..., xd | kelas) = P(x1|kelas) x P(x2|kelas) x ... x P(xd|kelas)

Fitur kontinu (BMI, AMH, folikel, dst) dimodelkan dengan kurva normal (Gaussian).
Fitur biner (gejala Y/N) dimodelkan dengan Bernoulli + Laplace smoothing supaya
tidak pernah menghasilkan peluang tepat 0.

Semua perkalian dihitung dalam bentuk LOG (log-likelihood), lalu dijumlahkan,
karena log(a x b) = log(a) + log(b). Ini mencegah angka jadi terlalu kecil untuk
disimpan komputer (underflow) saat banyak fitur dikalikan sekaligus.
"""
import math
import numpy as np
import pandas as pd


class NaiveBayesPCOS:
    def __init__(self, cont, binf, target="PCOS (Y/N)", laplace=1.0, var_smoothing=1e-9):
        self.cont = list(cont)          # fitur kontinu
        self.bin = list(binf)           # fitur biner
        self.target = target            # kolom target
        self.laplace = laplace          # smoothing Laplace untuk fitur biner
        self.var_smoothing = var_smoothing  # mencegah varians bernilai nol


    """Menghitung parameter model dari train data."""
    def fit(self, df):
        y = df[self.target].to_numpy()

        # Tambahan pada varians agar tidak terjadi pembagian dengan nol.
        if self.cont:
            eps = self.var_smoothing * df[self.cont].to_numpy(float).var(axis=0).max()
        else:
            eps = 0.0
        
        # Prior = proporsi masing-masing kelas dalam train data.
        self.prior = {}
        self.mean = {}
        self.var = {}
        self.p1 = {}
        for c in (0, 1):
            subset = df[y == c]                     
            n_c = len(subset)
            self.prior[c] = n_c / len(df)

            means, variances = [], []
            for f in self.cont:
                nilai = subset[f].to_numpy(float)
                means.append(nilai.mean())
                variances.append(nilai.var() + eps)   
            self.mean[c] = np.array(means)
            self.var[c] = np.array(variances)

            # Probabilitas fitur biner bernilai 1 pada masing-masing kelas.
            proporsi = []
            for f in self.bin:
                jumlah_ya = subset[f].to_numpy(float).sum()
                # Laplace smoothing agar probabilitas tidak menjadi 0.
                proporsi.append((jumlah_ya + self.laplace) / (n_c + 2 * self.laplace))
            self.p1[c] = np.array(proporsi)

        return self

    @staticmethod
    def _log_gaussian(x, mean, var):
        """Log-likelihood untuk satu fitur kontinu."""
        return -0.5 * math.log(2 * math.pi * var) - ((x - mean) ** 2) / (2 * var)

    @staticmethod
    def _log_bernoulli(x, p):
        """Log-likelihood untuk satu fitur biner."""
        return math.log(p) if x == 1 else math.log(1 - p)


    def _log_joint_satu_pasien(self, pasien, c):
        """Menghitung log peluang gabungan untuk satu pasien."""
        total = math.log(self.prior[c])
        for i, f in enumerate(self.cont):
            total += self._log_gaussian(float(pasien[f]), self.mean[c][i], self.var[c][i])
        for i, f in enumerate(self.bin):
            total += self._log_bernoulli(float(pasien[f]), self.p1[c][i])
        return total


    def predict_proba(self, df):
        """Menghasilkan probabilitas PCOS untuk setiap pasien."""
        hasil = []
        for _, pasien in df.iterrows():
            log0 = self._log_joint_satu_pasien(pasien, 0)   # skor "bukan PCOS"
            log1 = self._log_joint_satu_pasien(pasien, 1)   # skor "PCOS"

            # Menggeser nilai log agar exp() tetap stabil.
            geser = max(log0, log1)
            skor0 = math.exp(log0 - geser)
            skor1 = math.exp(log1 - geser)

            hasil.append(skor1 / (skor0 + skor1))
        return np.array(hasil)

    def predict(self, df, threshold=0.5):
        return (self.predict_proba(df) >= threshold).astype(int)

    def explain(self, row):
        """
        Menghitung kontribusi masing-masing fitur terhadap prediksi.
        Nilai positif menunjukkan kontribusi ke arah PCOS,
        sedangkan nilai negatif menunjukkan kontribusi ke arah non-PCOS.
        """
        baris = []
        for i, f in enumerate(self.cont):
            x = float(row[f])
            kontribusi = (self._log_gaussian(x, self.mean[1][i], self.var[1][i])
                          - self._log_gaussian(x, self.mean[0][i], self.var[0][i]))
            baris.append((f, x, kontribusi))
        for i, f in enumerate(self.bin):
            x = float(row[f])
            kontribusi = (self._log_bernoulli(x, self.p1[1][i])
                          - self._log_bernoulli(x, self.p1[0][i]))
            baris.append((f, x, kontribusi))
        # Kontribusi dari prior kelas.
        prior_kontribusi = math.log(self.prior[1]) - math.log(self.prior[0])
        baris.append(("(prior kelas)", np.nan, prior_kontribusi))

        out = pd.DataFrame(baris, columns=["fitur", "nilai", "kontribusi"])
        return out.sort_values("kontribusi", key=abs, ascending=False).reset_index(drop=True)
