# Clustering Provinsi Indonesia Berdasarkan Faktor Risiko Stunting dengan Fuzzy C-Means

Analisis ini mengelompokkan provinsi di Indonesia berdasarkan kemiripan profil faktor risiko stunting menggunakan **Fuzzy C-Means (FCM)**. Input clustering adalah enam indikator risiko yang telah distandardisasi. Prevalensi stunting tidak digunakan untuk membentuk klaster; variabel tersebut hanya dipakai setelah clustering sebagai validasi eksternal.

## Objective Penelitian

Tujuan utama penelitian adalah membentuk klaster provinsi berdasarkan faktor keluarga, akses air minum, dan sanitasi, kemudian menganalisis karakteristik centroid, faktor yang paling menonjol, tingkat keanggotaan fuzzy, persebaran spasial, dan hubungan post-hoc dengan prevalensi stunting.

Secara rinci, pipeline ini:

1. membentuk klaster dari enam indikator z-score;
2. menguji `c = 2, 3, 4, 5` dan `m = 1.5, 1.75, 2.0, 2.25, 2.5`;
3. memilih konfigurasi paling valid dan stabil tanpa memaksakan jumlah klaster tertentu;
4. menginterpretasi centroid dan dimensi risiko;
5. mengidentifikasi membership kuat, transisi, dan ambigu;
6. membuat visualisasi dan peta;
7. membandingkan profil klaster dengan prevalensi stunting sebagai validasi eksternal nonkausal.

Hasil data saat ini memilih **`c=2, m=1.5`**. Nilai ini adalah hasil ranking dari data dan eksperimen saat ini, bukan nilai hardcoded. Pipeline tetap dirancang berjalan apabila konfigurasi terbaik berubah menjadi `c=3`, `c=4`, atau `c=5`.

## Struktur Folder

```text
pkk-stunting-fcm-clustering/
├── data/
│   ├── raw/                 # CSV mentah
│   ├── interim/             # hasil harmonisasi antara
│   ├── processed/           # matriks model dan profil risiko
│   └── external/            # GeoJSON batas provinsi
├── notebooks/               # eksplorasi; bukan dependency wajib pipeline
├── outputs/
│   ├── model/               # hasil eksperimen FCM dan model terpilih
│   ├── analysis/            # interpretasi klaster dan validasi eksternal
│   ├── figures/             # grafik dan alias output lama
│   ├── maps/                # peta PNG/PDF publikasi
│   └── tables/              # tabel laporan dari workflow lama
├── src/
│   ├── preprocessing.py
│   ├── 04_fcm_model.py
│   ├── 05_validation_robustness_analysis.py
│   ├── 06_spatial_mapping.py
│   └── 07_visualization.py
├── tests/
├── run_pipeline.py
├── requirements.txt
└── README.md
```

## Input dan Fitur FCM

File input utama FCM:

```text
data/processed/fcm_model_matrix_zscore.csv
```

Enam fitur FCM:

| Fitur | Makna |
|---|---|
| `maternal_age_risk_z` | risiko usia ibu hamil |
| `low_knowledge_z` | rendahnya pengetahuan stunting |
| `water_no_or_unimproved_z` | tanpa akses / air minum tidak layak |
| `water_limited_z` | akses air minum terbatas |
| `sanitation_babs_z` | BABS |
| `sanitation_unimproved_z` | sanitasi tidak layak |

Prevalensi stunting berada di:

```text
data/processed/fcm_risk_profile_2024.csv
```

Kolom `stunting_prevalence_pct` hanya dipakai setelah klaster terbentuk untuk validasi eksternal. Ia bukan input model FCM.

## Pemilihan Model

`src/04_fcm_model.py` menjalankan FCM untuk kombinasi:

```text
c = 2, 3, 4, 5
m = 1.5, 1.75, 2.0, 2.25, 2.5
seed = 0..19
```

Ranking tidak lagi memakai semua metrik sebagai suara independen. Metrik dipisahkan menjadi kelompok:

| Kelompok | Metrik |
|---|---|
| Validitas struktur | Xie-Beni, minimum centroid distance |
| Stabilitas | mean pairwise ARI, membership change, centroid variation |
| Kualitas fuzzy | Modified Partition Coefficient, Partition Entropy |
| Diagnostik | convergence rate, centroid collision, empty crisp cluster |

Partition Coefficient tetap disimpan sebagai diagnostik, tetapi tidak dihitung sebagai bukti independen penuh bersama MPC dan PE. Ini mengurangi bias ke solusi yang terlalu crisp.

Pipeline juga menyimpan sensitivity ranking:

```text
outputs/model/fcm_configuration_summary.csv
outputs/model/fcm_ranking_sensitivity.csv
```

Skema ranking:

| Skema | Tujuan |
|---|---|
| `balanced` | keseimbangan struktur, stabilitas, fuzzy quality, diagnostik |
| `validity_focused` | menekankan validitas struktur |
| `stability_focused` | menekankan stabilitas antar seed |
| `fuzzy_quality_focused` | menekankan kualitas partisi fuzzy |

Konfigurasi terbaik dipilih dari ranking balanced dengan mempertimbangkan convergence, centroid yang tidak berhimpitan, tidak ada klaster kosong, dan konsistensi pada sensitivity ranking.

## Interpretasi Centroid

Nomor klaster bersifat arbitrer. Urutan risiko dibuat setelah centroid dianalisis melalui `overall_risk_score`, lalu disimpan sebagai `risk_rank`. Karena itu `cluster_1` tidak otomatis berarti risiko rendah atau tinggi.

Pipeline membedakan tiga konsep indikator:

| Kolom | Makna |
|---|---|
| `highest_centroid_indicator` | indikator dengan nilai centroid numerik tertinggi |
| `most_elevated_risk_indicator` | indikator positif tertinggi; kosong secara substantif jika semua centroid tidak positif |
| `most_distinguishing_indicator` | indikator dengan nilai absolut centroid terbesar, yaitu paling membedakan profil klaster |

Untuk klaster dengan semua skor negatif, output tidak menyebut indikator tersebut sebagai risiko tinggi. Kolom `dominance_interpretation` memberi teks kontekstual, misalnya bahwa semua dimensi berada di bawah rata-rata dan satu dimensi hanya paling mendekati rata-rata.

## Membership Dinamis

`src/05_validation_robustness_analysis.py` mendeteksi kolom:

```text
membership_cluster_<nomor>
```

Deteksi dilakukan berdasarkan nomor klaster, bukan urutan alfabet. Untuk setiap provinsi, pipeline memvalidasi nilai membership `[0, 1]`, jumlah membership mendekati 1, lalu menghitung ulang:

```text
maximum_membership
second_highest_membership
membership_margin
crisp_cluster
membership_status
```

Analisis ini bekerja untuk `c=2` sampai `c=5`. Kolom lama tetap dipertahankan selama masih valid.

## Label Klaster

Label dibuat setelah skor risiko dihitung:

| c | Label |
|---|---|
| 2 | lebih rendah, lebih tinggi |
| 3 | rendah, sedang, tinggi |
| 4 | rendah, menengah bawah, menengah atas, tinggi |
| 5 | sangat rendah, rendah, sedang, tinggi, sangat tinggi |

Jika skor dua klaster sangat dekat, label diberi konteks dimensi dominan agar tidak menyesatkan.

## Visualisasi dan Pemetaan

Notebook tetap ada untuk eksplorasi, tetapi pipeline final tidak bergantung pada notebook.

Script produksi:

```text
src/06_spatial_mapping.py
src/07_visualization.py
```

Output baru:

```text
outputs/maps/fcm_cluster_map.png
outputs/maps/fcm_cluster_map.pdf
outputs/maps/membership_certainty_map.png
outputs/maps/membership_certainty_map.pdf
outputs/figures/fcm_validity_plot.png
outputs/figures/centroid_heatmap.png
```

Alias output lama juga tetap dibuat:

```text
outputs/figures/peta_klaster_provinsi.png
outputs/figures/peta_kepastian_membership.png
outputs/figures/grafik_indeks_validitas.png
outputs/figures/heatmap_centroid.png
```

File spasial yang dibutuhkan:

```text
data/external/indonesia_38_provinces.geojson
```

Jika file spasial atau dependency pemetaan tidak tersedia, gunakan `--skip-mapping`. Pipeline tidak membuat data spasial palsu.

## Cara Menjalankan

Instal dependency:

```bash
python3 -m pip install -r requirements.txt
```

Jalankan seluruh pipeline:

```bash
python3 run_pipeline.py
```

Lewati pemetaan:

```bash
python3 run_pipeline.py --skip-mapping
```

Gunakan input FCM tertentu:

```bash
python3 run_pipeline.py --input data/processed/fcm_model_matrix_zscore.csv
```

Jalankan tahap satu per satu:

```bash
python3 src/preprocessing.py
python3 src/04_fcm_model.py
python3 src/05_validation_robustness_analysis.py
python3 src/07_visualization.py
python3 src/06_spatial_mapping.py
```

## Output Utama

Model:

```text
outputs/model/fcm_experiment_results.csv
outputs/model/fcm_configuration_summary.csv
outputs/model/fcm_ranking_sensitivity.csv
outputs/model/best_fcm_parameters.json
outputs/model/cluster_centroids_standardized.csv
outputs/model/cluster_membership.csv
```

Analisis:

```text
outputs/analysis/province_membership_analysis.csv
outputs/analysis/ambiguous_provinces.csv
outputs/analysis/cluster_profiles.csv
outputs/analysis/dominant_factors.csv
outputs/analysis/external_validation.csv
```

## Test

Jalankan seluruh test:

```bash
python3 -m unittest discover -s tests -v
```

Test mencakup:

1. metrik FCM lama;
2. deteksi membership dinamis untuk `c=2..5`;
3. indikator dominan pada centroid positif, negatif, dan tie;
4. ranking dan sensitivity analysis;
5. validasi eksternal dinamis;
6. kontrak pipeline dan schema output.

## Catatan Metodologi

Analisis ini bersifat ekologis pada level provinsi. Klaster menunjukkan kemiripan profil faktor risiko, bukan hubungan sebab-akibat individual. Prevalensi stunting digunakan sebagai validasi eksternal post-hoc, bukan ground truth dan bukan target optimasi. Pola prevalensi tidak dipaksa monoton terhadap `overall_risk_score`; apabila tidak searah, pipeline melaporkannya apa adanya.

Interpretasi label klaster harus merujuk pada centroid, `risk_rank`, dan `cluster_label`. Nomor klaster tidak memiliki makna substantif sebelum diberi interpretasi.
