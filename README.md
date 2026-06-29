# Clustering Provinsi Indonesia Berdasarkan Faktor Risiko Stunting menggunakan Fuzzy C-Means

> Analisis klasterisasi 36 provinsi Indonesia menggunakan metode **Fuzzy C-Means (FCM)** terhadap faktor keluarga, akses air minum, dan sanitasi berdasarkan data **Survei Status Gizi Indonesia (SSGI) 2024**.

---

## Daftar Isi

1. [Deskripsi Proyek](#deskripsi-proyek)
2. [Fitur Input Model](#fitur-input-model)
3. [Struktur Direktori](#struktur-direktori)
4. [Alur Menjalankan Proyek](#alur-menjalankan-proyek)
5. [Hasil dan Temuan](#hasil-dan-temuan)
6. [Persyaratan](#persyaratan)
7. [Catatan Metodologi](#catatan-metodologi)

---

## Deskripsi Proyek

Penelitian ini mengklasterisasi 36 provinsi Indonesia berdasarkan enam indikator faktor risiko stunting (bukan berdasarkan prevalensi stuntingnya). Clustering dilakukan menggunakan algoritma **Fuzzy C-Means (FCM)** untuk menghasilkan pengelompokan *soft* yang mampu menangkap provinsi dengan profil risiko yang ambigu atau berada di perbatasan antar kelompok.

Prevalensi stunting **tidak** digunakan sebagai input model — variabel tersebut hanya digunakan untuk **validasi eksternal** *post-hoc* guna mengevaluasi apakah kelompok yang terbentuk berkorelasi dengan beban stunting aktual.

---

## Fitur Input Model

Model FCM dibangun dari enam fitur berikut (dalam skala z-score):

| Fitur (Z-Score) | Deskripsi |
|---|---|
| `maternal_age_risk_z` | % ibu dengan usia risiko (&lt;21 atau &gt;40 tahun) |
| `low_knowledge_z` | % ibu dengan pengetahuan gizi rendah |
| `water_no_or_unimproved_z` | % RT tanpa akses / akses air minum tidak layak |
| `water_limited_z` | % RT dengan akses air minum terbatas |
| `sanitation_babs_z` | % RT buang air besar sembarangan (BABS) |
| `sanitation_unimproved_z` | % RT dengan sanitasi tidak layak |

---

## Struktur Direktori

```
pkk-stunting-fcm-clustering/
├── data/
│   ├── raw/                        # Data mentah SSGI 2024 (CSV)
│   ├── interim/                    # Data sementara hasil integrasi awal
│   └── processed/                  # Data bersih & terstandarisasi siap model
├── notebooks/                      # Notebook visualisasi dan eksplorasi
│   ├── 01_harmonize_province_names.ipynb
│   ├── 02_plot_cluster_map.ipynb
│   ├── 03_plot_membership_certainty_map.ipynb
│   ├── 04_plot_centroid_heatmap_and_validity.ipynb
│   ├── 05_export_tables.ipynb
│   └── preprocessing_pipeline.ipynb
├── src/                            # Pipeline utama Python
│   ├── preprocessing.py
│   ├── 04_fcm_model.py
│   └── 05_validation_robustness_analysis.py
├── outputs/
│   ├── model/                      # Artefak model (centroid, membership, params)
│   ├── analysis/                   # Hasil analisis validasi
│   └── figures/                    # Visualisasi (peta, heatmap, grafik)
├── requirements.txt
└── README.md
```

---

## Alur Menjalankan Proyek

Jalankan tahap-tahap berikut **secara berurutan**. Setiap tahap bergantung pada output tahap sebelumnya.

### Prasyarat

```bash
pip install -r requirements.txt
```

---

### Tahap 1 — Preprocessing Data (`src/preprocessing.py`)

Mengintegrasikan data mentah SSGI 2024 dari berbagai tabel (usia ibu, pengetahuan gizi, akses air, sanitasi, prevalensi stunting). Melakukan harmonisasi nama provinsi, pengecekan kualitas data (missing value, outlier IQR & Z-score), dan standardisasi Z-score pada fitur FCM.

```bash
python src/preprocessing.py
```

**Input:** File CSV di `data/raw/`

| File Output | Deskripsi |
|---|---|
| `data/interim/family_stunting_indicators.csv` | Indikator keluarga gabungan pra-merge |
| `data/processed/fcm_risk_profile_2024.csv` | Profil risiko lengkap 36 provinsi (skala asli %) |
| `data/processed/fcm_model_matrix_zscore.csv` | Matriks fitur FCM (skala Z-score, input model) |
| `data/processed/correlation_matrix.png` | Heatmap korelasi Pearson antar fitur |

---

### Tahap 2 — Eksperimen Fuzzy C-Means (`src/04_fcm_model.py`)

Menjalankan grid search FCM pada kombinasi jumlah klaster `c ∈ {2, 3, 4, 5}` dan *fuzziness exponent* `m ∈ {1.5, 1.75, 2.0, 2.25, 2.5}`, masing-masing dengan **20 random seed** untuk mengevaluasi stabilitas. Konfigurasi terbaik dipilih menggunakan **composite average rank** dari sembilan indikator validitas dan stabilitas.

```bash
python src/04_fcm_model.py
```

**Input:** `data/processed/fcm_model_matrix_zscore.csv`

| File Output | Deskripsi |
|---|---|
| `outputs/model/fcm_experiment_results.csv` | Hasil lengkap seluruh kombinasi `c`, `m`, dan seed |
| `outputs/model/best_fcm_parameters.json` | Konfigurasi terpilih + ringkasan validitas & stabilitas |
| `outputs/model/cluster_centroids_standardized.csv` | Koordinat centroid final (Z-score) |
| `outputs/model/cluster_membership.csv` | Derajat keanggotaan fuzzy tiap provinsi |

---

### Tahap 3 — Analisis Validasi (`src/05_validation_robustness_analysis.py`)

Menginterpretasi centroid klaster, menghitung skor per dimensi (keluarga, air, sanitasi), serta melakukan validasi eksternal menggunakan prevalensi stunting yang sebelumnya dieksklusi dari input model.

```bash
python src/05_validation_robustness_analysis.py
```

**Input:** Output model dari Tahap 2 + `data/processed/fcm_risk_profile_2024.csv`

| File Output | Deskripsi |
|---|---|
| `outputs/analysis/cluster_profiles.csv` | Profil klaster: skor dimensi, label, jumlah provinsi |
| `outputs/analysis/dominant_factors.csv` | Centroid + faktor/dimensi dominan tiap klaster |
| `outputs/analysis/province_membership_analysis.csv` | Keanggotaan fuzzy + label klaster + status kepastian tiap provinsi |
| `outputs/analysis/ambiguous_provinces.csv` | Provinsi dengan status keanggotaan ambigu/transisi |
| `outputs/analysis/external_validation.csv` | Statistik prevalensi stunting per klaster (validasi eksternal) |

---

### Tahap 4 — Visualisasi (Jupyter Notebooks)

Setelah ketiga skrip Python dijalankan, buka Notebook berikut di folder `notebooks/` untuk menghasilkan visualisasi:

| Notebook | Visualisasi yang Dihasilkan | Output |
|---|---|---|
| `02_plot_cluster_map.ipynb` | Peta choropleth klaster 36 provinsi | `outputs/figures/peta_klaster_provinsi.png` |
| `03_plot_membership_certainty_map.ipynb` | Peta tingkat kepastian keanggotaan *fuzzy* | `outputs/figures/peta_kepastian_membership.png` |
| `04_plot_centroid_heatmap_and_validity.ipynb` | Heatmap profil centroid + grafik validitas model | `outputs/figures/heatmap_centroid.png`, `outputs/figures/grafik_indeks_validitas.png` |
| `05_export_tables.ipynb` | Tabel ringkasan untuk laporan penelitian | — |

---

## Hasil dan Temuan

### Konfigurasi Model Terpilih

| Parameter | Nilai |
|---|---|
| Jumlah klaster (`c`) | **2** |
| *Fuzziness exponent* (`m`) | **1.5** |
| *Representative seed* | 13 |
| Jumlah inisialisasi | 20 |
| *Convergence rate* | 100% (20/20 seed konvergen) |
| *Composite rank score* | **2.11** (terbaik dari semua kombinasi) |

### Indeks Validitas Klaster Terpilih

| Indeks | Nilai | Interpretasi |
|---|---|---|
| Partition Coefficient (PC) | **0.8064** | Mendekati 1 → keanggotaan tegas |
| Modified PC (MPC) | **0.6129** | Menunjukkan partisi yang baik |
| Partition Entropy (PE) | **0.3109** | Mendekati 0 → ketidakpastian rendah |
| Xie-Beni Index (XB) | **0.5658** | Klaster kompak dan terpisah |
| Mean Pairwise ARI | **1.000** | Sempurna — identik di semua 20 seed |
| Min. Centroid Distance | **6.526** | Dua klaster terkemuka |

### Profil Centroid Klaster (Z-Score)

| Klaster | Usia Ibu Risiko | Penget. Rendah | Air Tidak Layak | Air Terbatas | BABS | Sanitasi Tidak Layak |
|---|---|---|---|---|---|---|
| Klaster 1 (Risiko Lebih Rendah) | −0.64 | −0.04 | −0.41 | −0.53 | −0.03 | −0.47 |
| Klaster 2 (Risiko Lebih Tinggi) | +0.91 | +0.16 | +0.60 | +0.79 | −0.02 | +0.69 |

### Ringkasan Profil Klaster

| Klaster | Label | Jumlah Provinsi | Dimensi Dominan | Mean Membership |
|---|---|---|---|---|
| 1 | Profil faktor risiko relatif lebih rendah | **23** | Sanitasi | 0.861 |
| 2 | Profil faktor risiko relatif lebih tinggi | **13** | Air Minum | 0.867 |

### Validasi Eksternal — Prevalensi Stunting per Klaster

> Prevalensi stunting **tidak digunakan sebagai input model**, namun perbedaan antar klaster di bawah ini mengkonfirmasi validitas hasil klasterisasi secara substantif.

| Klaster | Label | N Provinsi | Rata-rata | Median | Min | Max | Std Dev |
|---|---|---|---|---|---|---|---|
| 1 | Profil risiko lebih rendah | 23 | **19.60%** | 18.8% | 8.6% | 29.8% | 4.84% |
| 2 | Profil risiko lebih tinggi | 13 | **26.97%** | 26.1% | 20.8% | 37.0% | 4.83% |

**Selisih rata-rata prevalensi stunting antar klaster: ~7.4 poin persentase.**

### Keanggotaan Provinsi per Klaster

**Klaster 1 — Profil Faktor Risiko Relatif Lebih Rendah (23 provinsi)**

| Provinsi | Max Membership | Status |
|---|---|---|
| SULAWESI SELATAN | 0.502 | Ambigu tinggi |
| PAPUA | 0.508 | Ambigu tinggi |
| KALIMANTAN UTARA | 0.659 | Transisi moderat |
| JAWA TENGAH | 0.666 | Transisi moderat |
| SUMATERA BARAT | 0.673 | Transisi moderat |
| KALIMANTAN SELATAN | 0.711 | Transisi moderat |
| BANGKA BELITUNG | 0.767 | Keanggotaan kuat |
| NUSA TENGGARA BARAT | 0.774 | Keanggotaan kuat |
| JAWA BARAT | 0.818 | Keanggotaan kuat |
| BALI | 0.943 | Keanggotaan kuat |
| KEPULAUAN RIAU | 0.967 | Keanggotaan kuat |
| DI YOGYAKARTA | 0.967 | Keanggotaan kuat |
| DKI JAKARTA | 0.971 | Keanggotaan kuat |
| BENGKULU | 0.975 | Keanggotaan kuat |
| LAMPUNG | 0.976 | Keanggotaan kuat |
| ACEH | 0.986 | Keanggotaan kuat |
| SUMATERA SELATAN | 0.990 | Keanggotaan kuat |
| JAWA TIMUR | 0.990 | Keanggotaan kuat |
| SUMATERA UTARA | 0.991 | Keanggotaan kuat |
| KALIMANTAN TIMUR | 0.991 | Keanggotaan kuat |
| BANTEN | 0.994 | Keanggotaan kuat |
| RIAU | 0.994 | Keanggotaan kuat |
| JAMBI | 0.999 | Keanggotaan kuat |

**Klaster 2 — Profil Faktor Risiko Relatif Lebih Tinggi (13 provinsi)**

| Provinsi | Max Membership | Status |
|---|---|---|
| SULAWESI UTARA | 0.559 | Ambigu tinggi |
| KALIMANTAN TENGAH | 0.789 | Keanggotaan kuat |
| NUSA TENGGARA TIMUR | 0.800 | Keanggotaan kuat |
| MALUKU | 0.833 | Keanggotaan kuat |
| PAPUA SELATAN | 0.840 | Keanggotaan kuat |
| SULAWESI TENGGARA | 0.874 | Keanggotaan kuat |
| SULAWESI TENGAH | 0.894 | Keanggotaan kuat |
| KALIMANTAN BARAT | 0.905 | Keanggotaan kuat |
| PAPUA BARAT | 0.918 | Keanggotaan kuat |
| PAPUA BARAT DAYA | 0.938 | Keanggotaan kuat |
| MALUKU UTARA | 0.958 | Keanggotaan kuat |
| GORONTALO | 0.967 | Keanggotaan kuat |
| SULAWESI BARAT | 0.993 | Keanggotaan kuat |

---

## Persyaratan

```bash
pip install -r requirements.txt
```

Dependency utama: `pandas`, `numpy`, `scikit-fuzzy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `geopandas` (untuk peta).

---

## Catatan Metodologi

- **Nomor klaster bersifat arbitrer.** Label "risiko lebih tinggi" dan "risiko lebih rendah" diberikan berdasarkan analisis centroid pasca-clustering, bukan urutan numerik FCM.
- **Prevalensi stunting tidak masuk sebagai fitur model.** Variabel ini sepenuhnya diisolasi dan hanya digunakan untuk validasi eksternal *post-hoc*.
- **Interpretasi keanggotaan fuzzy:** Provinsi dengan `maximum_membership < 0.60` atau `membership_margin < 0.20` dikategorikan sebagai **Ambigu tinggi**; nilai `0.60–0.75` sebagai **Transisi moderat**; di atas `0.75` sebagai **Keanggotaan kuat**.
- **Stabilitas sempurna:** Nilai *pairwise ARI = 1.0* menunjukkan bahwa ke-20 inisialisasi random seed menghasilkan pembagian klaster yang identik, menjamin reprodusibilitas hasil.
