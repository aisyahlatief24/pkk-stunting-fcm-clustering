# pkk-stunting-fcm-clustering

Clustering provinsi Indonesia berdasarkan faktor keluarga, air minum, dan sanitasi menggunakan Fuzzy C-Means (SSGI 2024).

## Fuzzy C-Means Modeling

Modul `src/04_fcm_model.py` menjalankan eksperimen Fuzzy C-Means pada matriks fitur yang sudah distandardisasi. Tujuannya memilih konfigurasi FCM terbaik berdasarkan validitas klaster dan kestabilan antar-inisialisasi, tanpa membandingkan dengan K-Means atau algoritma clustering lain.

### Input

File input:

```text
data/processed/fcm_model_matrix_zscore.csv
```

Kolom identifier:

```text
province_name
```

Kolom fitur FCM:

```text
maternal_age_risk_z
low_knowledge_z
water_no_or_unimproved_z
water_limited_z
sanitation_babs_z
sanitation_unimproved_z
```

Data input sudah berupa z-score, sehingga modul FCM tidak melakukan standardisasi ulang.

### Instalasi Dependency

```bash
pip install -r requirements.txt
```

Dependency utama untuk modul FCM adalah `pandas`, `numpy`, `scikit-fuzzy`, `scipy`, dan `scikit-learn`.

### Cara Menjalankan

```bash
python src/04_fcm_model.py
```

### Parameter Eksperimen

Jumlah klaster yang diuji:

```text
c = [2, 3, 4, 5]
```

Fuzziness exponent yang diuji:

```text
m = [1.5, 1.75, 2.0, 2.25, 2.5]
```

Setiap kombinasi dijalankan dengan 20 seed reproducible:

```text
RANDOM_SEEDS = list(range(20))
```

Parameter numerik:

```text
ERROR_TOLERANCE = 1e-5
MAX_ITERATIONS = 1000
```

### Indeks Validitas

`Partition Coefficient (PC)` mengukur ketegasan membership. Nilai lebih besar menunjukkan membership lebih tegas.

`Modified Partition Coefficient (MPC)` adalah versi ternormalisasi dari PC. Nilai lebih besar menunjukkan hasil yang lebih baik.

`Partition Entropy (PE)` mengukur ketidakpastian membership. Nilai lebih kecil menunjukkan membership lebih jelas.

`Xie-Beni Index (XB)` mengukur kekompakan dan keterpisahan klaster. Nilai lebih kecil menunjukkan klaster lebih kompak dan lebih terpisah.

Stabilitas antar-seed dihitung setelah label klaster diselaraskan menggunakan Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). ARI (`adjusted_rand_score`) dipakai hanya sebagai ukuran kestabilan internal crisp assignment antar-inisialisasi FCM.

### Output

Output model disimpan di:

```text
outputs/model/fcm_experiment_results.csv
outputs/model/best_fcm_parameters.json
outputs/model/cluster_centroids_standardized.csv
outputs/model/cluster_membership.csv
```

`fcm_experiment_results.csv` berisi satu baris untuk setiap kombinasi `c`, `m`, dan `seed`.

`best_fcm_parameters.json` berisi konfigurasi terbaik, seed representatif, metode seleksi, ringkasan validitas, dan ringkasan stabilitas.

`cluster_centroids_standardized.csv` berisi centroid final dalam skala z-score.

`cluster_membership.csv` berisi membership tiap provinsi di semua klaster, maximum membership, second highest membership, membership margin, dan crisp cluster sementara.

### Catatan Interpretasi

Nomor klaster FCM bersifat arbitrer. `cluster_1`, `cluster_2`, dan seterusnya bukan label risiko rendah, sedang, atau tinggi. Pelabelan substantif dilakukan setelah centroid dianalisis pada tahap berikutnya.

Prevalensi stunting tidak digunakan sebagai input FCM. Variabel tersebut hanya digunakan untuk validasi eksternal pasca-clustering.
