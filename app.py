import pandas as pd
import streamlit as st

FITUR = ["Usia", "BMI", "Glukosa", "Tekanan Darah", "Riwayat Keluarga"]

DATA = pd.DataFrame(
    [
        ["P01", 25, 21.5, 90, 110, 0, "Rendah"],
        ["P02", 30, 23.1, 95, 115, 0, "Rendah"],
        ["P03", 35, 25.0, 105, 120, 0, "Rendah"],
        ["P04", 40, 27.5, 120, 125, 1, "Tinggi"],
        ["P05", 45, 30.2, 140, 135, 1, "Tinggi"],
        ["P06", 50, 31.5, 155, 140, 1, "Tinggi"],
        ["P07", 55, 33.0, 165, 145, 1, "Tinggi"],
        ["P08", 28, 22.5, 92, 112, 0, "Rendah"],
        ["P09", 38, 26.8, 115, 122, 0, "Rendah"],
        ["P10", 48, 29.5, 145, 130, 1, "Tinggi"],
    ],
    columns=["Pasien"] + FITUR + ["Kelas"],
)

WARNA_UTAMA = "#0f766e"
WARNA_TINGGI = "#e5484d"
WARNA_RENDAH = "#30a46c"

# Tombol Prediksi: ubah angka di bawah untuk mengatur tampilan tombol.
# TRANSPARANSI_TOMBOL: 0.05 (sangat transparan) sampai 1 (pekat)
# LEBAR_TOMBOL: 100% = selebar kartu, 70% = lebih pendek
TRANSPARANSI_TOMBOL = 0.15
LEBAR_TOMBOL = "100%"

CSS_MENTAH = """
<style>
.block-container { max-width: 720px; padding-top: 5rem; padding-bottom: 3rem; }
#MainMenu, footer { visibility: hidden; }
.judul { font-size: 1.8rem; font-weight: 700; line-height: 1.25; margin: 0; }
.subjudul { opacity: 0.7; margin: 0.25rem 0 1.25rem 0; }
div[data-testid="stForm"] { border-radius: 16px; padding: 1.25rem; }
[data-testid="stFormSubmitButton"] { width: __LEBAR__ !important; }
.stElementContainer:has([data-testid="stFormSubmitButton"]) { width: 100% !important; }
[data-testid="stFormSubmitButton"] button { width: 100% !important; border-radius: 14px; padding: 0.85rem 1.5rem; background-color: rgba(15, 118, 110, __ALFA__) !important; border: 1.5px solid rgba(15, 118, 110, 0.5) !important; }
[data-testid="stFormSubmitButton"] button, [data-testid="stFormSubmitButton"] button * { color: __UTAMA__ !important; font-weight: 600; }
[data-testid="stFormSubmitButton"] button:hover { background-color: rgba(15, 118, 110, 0.28) !important; border-color: __UTAMA__ !important; }
[data-testid="stFormSubmitButton"] button:focus { box-shadow: 0 0 0 0.2rem rgba(15, 118, 110, 0.3) !important; }
.hasil { border-radius: 16px; padding: 1.1rem 1.25rem; margin: 0.5rem 0 1rem 0; }
.hasil .label { font-size: 0.85rem; opacity: 0.75; }
.hasil .nilai { font-size: 1.9rem; font-weight: 700; line-height: 1.2; }
.hasil .ket { margin-top: 0.4rem; opacity: 0.9; }
.bar { display: flex; height: 34px; border-radius: 10px; overflow: hidden; margin-bottom: 0.4rem; }
.bar div { display: flex; align-items: center; justify-content: center; color: #fff; font-size: 0.85rem; font-weight: 600; white-space: nowrap; }
</style>
"""
CSS = (
    CSS_MENTAH.replace("__UTAMA__", WARNA_UTAMA)
    .replace("__ALFA__", str(TRANSPARANSI_TOMBOL))
    .replace("__LEBAR__", LEBAR_TOMBOL)
)


def hitung_knn(data_baru, k, normalisasi=False):
    """Hitung KNN dengan jarak Euclidean kuadrat.

    Mengembalikan (prediksi, suara_tinggi, suara_rendah, tabel_peringkat).
    """
    X = DATA[FITUR].astype(float)
    x = pd.Series(data_baru, index=FITUR, dtype=float)

    if normalisasi:
        minimum, maksimum = X.min(), X.max()
        rentang = (maksimum - minimum).replace(0, 1)
        X = (X - minimum) / rentang
        x = (x - minimum) / rentang

    hasil = DATA[["Pasien", "Kelas"]].copy()
    hasil["Jarak²"] = ((X - x) ** 2).sum(axis=1).round(4)
    hasil = hasil.sort_values("Jarak²", kind="stable").reset_index(drop=True)
    hasil.insert(0, "Peringkat", range(1, len(hasil) + 1))

    tetangga = hasil.head(k)
    tinggi = int((tetangga["Kelas"] == "Tinggi").sum())
    rendah = k - tinggi
    prediksi = "Tinggi" if tinggi > rendah else "Rendah"
    return prediksi, tinggi, rendah, hasil


def warnai_kelas(nilai):
    if nilai == "Tinggi":
        return "background-color: rgba(229, 72, 77, 0.25)"
    if nilai == "Rendah":
        return "background-color: rgba(48, 164, 108, 0.25)"
    return ""


def tabel_berwarna(df):
    gaya = df.style.format({"Jarak²": "{:g}"})
    terapkan = gaya.map if hasattr(gaya, "map") else gaya.applymap
    return terapkan(warnai_kelas, subset=["Kelas"])


def kartu_hasil(prediksi, tinggi, rendah, k):
    warna = WARNA_TINGGI if prediksi == "Tinggi" else WARNA_RENDAH
    html = (
        f'<div class="hasil" style="background:{warna}22; border-left:6px solid {warna};">'
        f'<div class="label">Hasil prediksi</div>'
        f'<div class="nilai" style="color:{warna};">Risiko {prediksi}</div>'
        f'<div class="ket">{tinggi} dari {k} tetangga terdekat berisiko Tinggi '
        f"dan {rendah} berisiko Rendah.</div>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def bar_suara(tinggi, rendah):
    total = tinggi + rendah
    bagian = ""
    if tinggi:
        bagian += (
            f'<div style="width:{tinggi / total * 100}%; background:{WARNA_TINGGI};">'
            f"Tinggi {tinggi}</div>"
        )
    if rendah:
        bagian += (
            f'<div style="width:{rendah / total * 100}%; background:{WARNA_RENDAH};">'
            f"Rendah {rendah}</div>"
        )
    st.markdown(f'<div class="bar">{bagian}</div>', unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="SPK Risiko Diabetes",
        page_icon="📊",
        layout="centered",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<p class="judul">📊 Prediksi Risiko Diabetes</p>'
        '<p class="subjudul">Sistem pendukung keputusan berbasis '
        "K-Nearest Neighbors (KNN)</p>"
        '<p class="subjudul">Masukkan data baru, lalu sistem akan '
        "menilai risikonya dari kasus yang paling mirip.</p>",
        unsafe_allow_html=True,
    )

    with st.form("form_pasien"):
        st.markdown("**Data pasien baru**")
        kolom_1, kolom_2 = st.columns(2)
        usia = kolom_1.number_input("Usia (tahun)", min_value=1, max_value=120, value=42, step=1)
        bmi = kolom_2.number_input("BMI", min_value=10.0, max_value=60.0, value=28.5, step=0.1, format="%.1f")

        kolom_3, kolom_4 = st.columns(2)
        glukosa = kolom_3.number_input("Glukosa", min_value=50, max_value=400, value=130, step=1)
        tekanan = kolom_4.number_input("Tekanan darah", min_value=60, max_value=250, value=128, step=1)

        riwayat = st.radio(
            "Riwayat diabetes dalam keluarga", ["Ada", "Tidak ada"], horizontal=True
        )
        k = st.selectbox("Nilai K (jumlah tetangga)", [1, 3, 5, 7, 9], index=1)
        normalisasi = st.checkbox(
            "Gunakan normalisasi min-max",
            help="Menyamakan skala semua atribut ke rentang 0 sampai 1 sebelum jarak dihitung.",
        )
        kirim = st.form_submit_button("Prediksi", type="primary")

    if kirim:
        data_baru = [usia, bmi, glukosa, tekanan, 1 if riwayat == "Ada" else 0]
        prediksi, tinggi, rendah, hasil = hitung_knn(data_baru, k, normalisasi)

        st.subheader("Kesimpulan")
        kartu_hasil(prediksi, tinggi, rendah, k)
        bar_suara(tinggi, rendah)
        st.caption("Kelas dengan suara terbanyak menjadi hasil prediksi.")

        st.markdown(f"**{k} tetangga terdekat**")
        st.dataframe(tabel_berwarna(hasil.head(k)), hide_index=True)

        with st.expander("Jarak ke seluruh data latih"):
            lengkap = hasil.copy()
            lengkap["Dipakai"] = ["Ya" if i < k else "-" for i in range(len(lengkap))]
            st.dataframe(tabel_berwarna(lengkap), hide_index=True)

    with st.expander("Tentang model"):
        st.write(
            "Data latih terdiri dari 10 pasien (5 Rendah dan 5 Tinggi). "
            "Jarak dihitung dengan rumus Euclidean kuadrat pada lima atribut: "
            "Usia, BMI, Glukosa, Tekanan darah, dan Riwayat keluarga. "
            "Kelas pasien baru ditentukan lewat voting mayoritas dari K "
            "tetangga terdekat."
        )

    st.caption(
        "Simulasi bertujuan untuk mempelajari sistem pendukung keputusan risiko diabetes dengan metode KNN."
    )


if __name__ == "__main__":
    main()
