import pickle
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
import streamlit as st
from clean_data import load_mode
from model import NaiveBayesPCOS  # noqa: F401  (dibutuhkan untuk unpickle)

THRESHOLD = 0.40
PINK_SCALE = LinearSegmentedColormap.from_list("pinkrisk", ["#F0A6C0", "#D6457F", "#8E1F4D"])
LABEL = {"Follicle No. (L)": "Jumlah folikel ovarium kiri", "Follicle No. (R)": "Jumlah folikel ovarium kanan",
         "AMH(ng/mL)": "Kadar AMH", "BMI": "BMI", "Cycle length(days)": "Lama haid (hari)",
         "Age (yrs)": "Usia", "Cycle(R/I)": "Siklus tidak teratur", "hair growth(Y/N)": "Rambut tumbuh berlebih",
         "Skin darkening (Y/N)": "Penggelapan kulit", "Pimples(Y/N)": "Jerawat", "Weight gain(Y/N)": "Kenaikan berat badan",
         "Fast food (Y/N)": "Sering makan fast food", "Hair loss(Y/N)": "Rambut rontok", "(prior kelas)": "Peluang awal (prior)"}

st.set_page_config(page_title="PCOScreen", page_icon="🌸", layout="centered")

st.markdown("""
<style>
h1 {letter-spacing: -0.02em; padding-bottom: 0;}
[data-testid="stForm"] {border-radius: 12px; padding: 1.5rem;}
.hasil {border-left: 6px solid; border-radius: 8px; background: #FFFFFF; padding: 1rem 1.25rem; margin: 1rem 0;}
.hasil .angka {font-size: 2.6rem; font-weight: 700; line-height: 1.1;}
.hasil .judul {font-weight: 600; margin-top: .25rem;}
.hasil .saran {color: #5c5c66; font-size: .92rem; margin-top: .25rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load(mode):
    with open(f"model/pcos_{mode}.pkl", "rb") as f:
        model = pickle.load(f)
    df, _ = load_mode(mode)
    return model, df


st.title("PCOS Risk Assessment")
st.caption("Hasil skrining ini merupakan alat bantu pembelajaran, bukan diagnosis medis."
           "Model Naive Bayes dilatih menggunakan dataset pasien dari 10 rumah sakit di Kerala, India."
           "Hasil skrining perlu dikonfirmasi oleh dokter.")

mode = st.radio("Mode input", ["mandiri", "klinis"], horizontal=True,
                format_func=lambda m: {"mandiri": "Mandiri (tanpa USG/lab)", "klinis": "Klinis (dengan USG & AMH)"}[m])
model, ref = load(mode)

with st.form("input"):
    c1, c2 = st.columns(2)
    x = {}
    x["Age (yrs)"] = c1.number_input("Usia (tahun)", 15, 60, 30)
    w = c1.number_input("Berat badan (kg)", 30.0, 150.0, 60.0, step=1.0, format="%.1f")
    h = c2.number_input("Tinggi badan (cm)", 120.0, 200.0, 158.0, step=1.0, format="%.0f")
    x["BMI"] = w / (h / 100) ** 2
    c2.metric("BMI dihitung", f"{x['BMI']:.1f}")
    x["Cycle length(days)"] = c1.number_input(
        "Lama haid (hari)", 0, 15, 5,
        help="Berapa hari haid berlangsung setiap bulan. Umumnya sekitar 5 hari.")
    x["Cycle(R/I)"] = int(c2.radio("Keteraturan siklus haid", [0, 1], horizontal=True,
                                    format_func=lambda v: "Teratur" if v == 0 else "Tidak teratur"))

    sym = ["hair growth(Y/N)", "Skin darkening (Y/N)", "Pimples(Y/N)", "Weight gain(Y/N)", "Fast food (Y/N)", "Hair loss(Y/N)"]
    dipilih = st.pills("Gejala yang dialami (pilih semua yang sesuai)", [LABEL[s] for s in sym],
                       selection_mode="multi") or []
    for s in sym:
        x[s] = int(LABEL[s] in dipilih)

    if mode == "klinis":
        st.markdown("**Hasil pemeriksaan**")
        k = st.columns(3)
        x["Follicle No. (L)"] = k[0].number_input("Folikel kiri (USG)", 0, 30, 6)
        x["Follicle No. (R)"] = k[1].number_input("Folikel kanan (USG)", 0, 30, 6)
        x["AMH(ng/mL)"] = k[2].number_input("AMH (ng/mL)", 0.0, 100.0, 4.0)
    go = st.form_submit_button("Hitung risiko", type="primary")

if go:
    row = pd.Series({f: float(x[f]) for f in model.cont + model.bin})
    p = float(model.predict_proba(pd.DataFrame([row]))[0])

    tinggi = p >= THRESHOLD
    warna = to_hex(PINK_SCALE(p))   # makin tinggi risiko, makin pink tua
    judul = "Risiko lebih tinggi" if tinggi else "Risiko lebih rendah"
    saran = ("Disarankan konsultasi ke dokter." if tinggi
             else "Tetap konsultasi ke dokter bila ada keluhan.")
    st.markdown(f"""
<div class="hasil" style="border-left-color:{warna}">
  <div class="angka" style="color:{warna}">{p*100:.0f}%</div>
  <div class="judul">{judul}</div>
  <div class="saran">{saran}</div>
</div>""", unsafe_allow_html=True)

    out_of_range = [LABEL[f] for f in model.cont if not ref[f].min() <= row[f] <= ref[f].max()]
    if out_of_range:
        st.info("Nilai di luar rentang data latih (hasil kurang andal): " + ", ".join(out_of_range))

    ex = model.explain(row)
    ex["Faktor"] = ex["fitur"].map(LABEL)
    top = ex.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    mx = top["kontribusi"].abs().max() or 1
    warna_bar = [PINK_SCALE((v + mx) / (2 * mx)) for v in top["kontribusi"]]
    ax.barh(top["Faktor"], top["kontribusi"], color=warna_bar)
    ax.axvline(0, color="#2B2B33", lw=.8)
    ax.set_xlabel("Dorongan ke arah PCOS (+) atau bukan PCOS (−)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("Bagaimana membaca grafik ini?"):
        st.write( "Model Naive Bayes membandingkan setiap nilai input dengan pola pada "
        "pasien PCOS dan non-PCOS. Semakin jauh nilai kontribusi dari nol, "
        "semakin besar pengaruh faktor tersebut terhadap hasil prediksi. "
        "Nilai positif menunjukkan kontribusi ke arah PCOS, sedangkan nilai "
        "negatif menunjukkan kontribusi ke arah non-PCOS. Seluruh kontribusi "
        "faktor, bersama dengan peluang awal (prior probability), digunakan "
        "untuk menghitung log-odds akhir.")
    with st.expander("Seberapa bisa dipercaya hasil ini?"):
        st.write(f"Model telah diuji menggunakan *cross-validation* dan mencapai akurasi "
        "**89,5% pada mode klinis** dan **84,0% pada mode mandiri**. "
        "Hasil ini dapat digunakan sebagai **indikasi awal**, tetapi bukan sebagai "
        "**diagnosis medis**."
)