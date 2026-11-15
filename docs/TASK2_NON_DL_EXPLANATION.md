# Penjelasan Pipeline Tugas 2 — Pendekatan Non-Deep Learning

Dokumen ini merangkum alur kerja paket `src/task2_non_dl/` beserta integrasinya pada *notebook* sebagai bagian pembagian tugas **Task 2 (Non-Deep Learning)**. Seluruh tahapan di bawah dirancang secara analitis menggunakan teknik *classical computer vision* tanpa melatih jaringan saraf tiruan.

---

## 1. Rekonstruksi Geometri 3D Kubus (`src/task2_non_dl/geometry.py`)

Pada modul ini, kami mengubah citra kemasan kubus 3D berlatar gradasi menjadi tiga citra sisi persegi kanonik berukuran `384x384` piksel (`CANONICAL_FACE_SIZE = 384`):

1. **Pemodelan Latar (`fit_background_model`)**: Kami menyampel piksel sepanjang bingkai tepi luar citra (`border = 20`) untuk meregresikan model iluminasi kuadratik-radial `[1, u, v, u^2 + v^2]` per kanal RGB dengan penyaringan *outlier* dua tahap (*inlier re-fit* bila residu `> 15.0`). Selisih terhadap model latar kemudian kami haluskan menggunakan `cv2.GaussianBlur` `(5, 5)` guna menekan *noise* latar sebelum menghitung peta jarak warna (`dist_map`).
2. **Ekstraksi Heksagon Sub-Piksel (`extract_cube_hexagon`)**: Kami membangun *foreground mask* menggunakan ambang adaptif berbasis persentil latar `max(4.2, min(12.0, p98.5 + 1.8))`, membersihkannya lewat operasi morfologi (*close* `9x9` lalu *open* `5x5`), mengambil *convex hull*, serta menyederhanakannya menjadi poligon 6 simpul (`cv2.approxPolyDP` dan helper `_reduce_polygon_to_hex` / `_collapse_short_edges`). Selanjutnya, kami menyaring titik kontur di bagian tengah tiap segmen rusuk, memasang garis kokoh `cv2.fitLine(..., cv2.DIST_HUBER)` melalui `_fit_line_huber`, lalu menghitung perpotongan garis (`_intersect_2d_lines`) untuk memperoleh 6 sudut luar kubus berpresisi sub-piksel searah jarum jam dimulai dari simpul paling atas.
3. **Penentuan Titik Pertemuan Rusuk Dalam (`compute_vanishing_center`)**: Kami memproyeksikan pasangan rusuk luar yang sejajar secara 3D dalam koordinat homogen (`_solve_center_for_parity`) untuk memperoleh tiga *vanishing point*, lalu mengarahkan ketiga rusuk dalam dan mencari titik temu (`v_center`) lewat *least-squares*. Kami juga melakukan disambiguasi paritas simpul rusuk `{1, 3, 5}` vs `{0, 2, 4}` menggunakan `_point_in_triangle` serta perbandingan deviasi warna intra-kuadran `_eval_partition_variance` (beralih ke paritas genap hanya jika `var_even < 0.65 * var_odd`).
4. **Segmentasi & Rektifikasi Tiga Sisi (`segment_cube_faces`, `rectify_face`)**: Kami membagi heksagon kubus mengelilingi `v_center` menjadi tiga kuadran searah jarum jam: kuadran paling atas—atau lebih terang berdasarkan persentil ke-75 luminansi jika selisih tinggi `< 18.0` piksel—kami tetapkan sebagai `top`, diikuti `right` dan `front`. Keempat sudut tiap kuadran kami tata menjadi urutan standar `[kiri-atas, kanan-atas, kanan-bawah, kiri-bawah]` lewat `_orient_quad_upright`, lalu kami luruskan memakai transformasi *homography* (`cv2.getPerspectiveTransform` dan `cv2.warpPerspective`).

---

## 2. Segmentasi Cookie, Pencocokan Template, & Deteksi Cacat (`src/task2_non_dl/detector.py`)

Setelah ketiga sisi diluruskan ke ruang kanonik `384x384`, kami mendeteksi jumlah cookie, status cacat, dan jumlah desain unik pada tiap sisi:

1. **Penyiapan Referensi (`prepare_reference_cache`)**: Kami menskalakan setiap citra desain referensi secara proporsional ke separuh dimensi alaminya (`0.5x`) sambil menyimpan `mask` penuh maupun `mask_inner` (ter-erosi `3x3` untuk menghindari *anti-aliasing* tepi) di dalam *cache* memori agar tidak dihitung berulang kali.
2. **Segmentasi Instansi & Penggabungan Fragmen (`segment_face_cookies`)**: Kami mengestimasi warna kardus dominan dari median area interior sisi (`[12:-12, 12:-12]`), menghitung peta jarak warna Euclidean (`diff > 22.0`), meredam pinggir 10 piksel agar rusuk pembatas tidak bocor, lalu menyaring *connected components* berdasarkan luas minimum dan rasio piksel ber-saturasi/gelap (`rich_ratio`). Karena cacat lubang tengah dapat membelah sebuah cookie menjadi dua bagian, helper `_merge_split_fragments` kami gunakan untuk menyatukan kembali pasangan fragmen berdekatan (`gap <= 14.0` piksel) yang gabungan *aspect ratio*-nya konsisten (`1.40 <= m_ar <= 2.75`, luas `<= 18500`).
3. **Penyelarasan Coarse-to-Fine & Penilaian Tri-Modal (`align_and_score_cookie`)**: Untuk setiap cookie, kami mengestimasi sudut sumbu utama melalui momen citra (`_major_axis_angle_deg`), memutar potongan ROI agar berdiri tegak (`_rotate_crop_patch`), lalu menjalankan pencarian *template matching* bertahap: tahap *coarse* FFT skala `0.35x` tanpa *mask* (`_match_template_coarse`) diikuti tahap *fine* ber-*mask* (`_match_template_fine`) pada maksimal 3 kandidat teratas termasuk variasi skala (`peer_scale` bila tersedia) dan pergeseran sudut `+-3.5` derajat. Selanjutnya, kami melakukan koreksi pencahayaan per kanal RGB (`_fit_shading_affine`) dengan toleransi pergeseran lokal `3x3` (`res_shift_map`) untuk mengevaluasi tiga mode cacat:
   - **`miss_sil` (`boundary_cut`)**: Fraksi siluet template yang hilang terhadap `0.085`; untuk desain dikenal (`ncc >= 0.62`, atau `ncc_on_solid >= 0.74` dengan `coverage >= 0.42` saat terpotong tepi), kami mensyaratkan ketidakcocokan warna `res_shift_map > 19.0` agar bagian putih/cerah alami seperti janggut tidak dikira hilang.
   - **`hole_frac` (`interior_hole`)**: Fraksi area lubang di dalam siluet solid yang warnanya benar-benar menyerupai warna kardus latar terhadap `0.055`.
   - **`blob_frac` (`missing_feature`)**: Fraksi gumpalan residu *icing* hilang (`res_shift_map > 18.0` yang ter-erosi `3x3` untuk membuang garis tepi tipis 1 piksel) terhadap `0.085`.
   - **`defect_ratio`**: Nilai maksimum dari ketiga rasio ternormalisasi tersebut; cookie kami klasifikasikan cacat bila `defect_ratio > 1.0`.
4. **Estimasi Desain Unik & Orkestrasi (`_estimate_unique_designs`, `predict_cube_non_dl`)**: Kami menggabungkan ID desain referensi yang cocok kuat (`ncc >= 0.62`) dengan pengelompokan histogram HSV `[12, 4]` (*cosine similarity* `>= 0.85`) guna menghitung jumlah desain unik bagi cookie desain baru pada data tersembunyi. Seluruh tahapan digabung di `predict_cube_non_dl`, termasuk penegakan aturan batas maksimal `2` cookie cacat per citra kubus (mempertahankan 2 skor `defect_ratio` tertinggi), serta menyesuaikan kamus keluaran menjadi `6` kunci (`LEADERBOARD=False`) atau `9` kunci (`LEADERBOARD=True`).

---

## 3. Skrip Evaluasi, Visualisasi, & Integrasi Notebook

- **`src/task2_non_dl/evaluate_and_visualize.py`**: Kami mengeksekusi tolok ukur penuh pada 100 citra kubus berlabel, memvalidasi format skema CSV 7 kolom dan 10 kolom, memproduksi tiga gambar diagnostik ke `reports/figures/` (`task2_non_dl_pipeline_stages.png`, `task2_non_dl_defect_types.png`, `task2_non_dl_hidden_samples.png`), serta mengekspor prediksi `200` citra *hidden test* ke `submissions/`.
- **`notebooks/Gingerbread_Student_Starter_Colab_ID_v2.ipynb` (Sel 12)**: Kami memperbarui fungsi `predict_cube_non_dl` agar mengimpor paket `src.task2_non_dl` secara otomatis saat berada di repositori lokal dan menyediakan *inline fallback* lengkap saat dijalankan mandiri di Google Colab.
- **`notebooks/Task2_Non_DL_Pipeline_dan_Analisis.ipynb`**: Kami menyediakan *notebook* khusus Tugas 2 Non-DL yang menampilkan eksekusi bertahap beserta visualisasi dan tabel evaluasi tervalidasi.

---

## 4. Ringkasan Hasil Evaluasi Terverifikasi

- **Akurasi Segmentasi Geometri 3D (`100` kubus / `300` sisi)**: `Mean IoU = 0.9843`, `Min IoU = 0.9320`.
- **Akurasi Evaluasi End-to-End (`100` kubus berlabel)**:

| Split | Jumlah Kubus | `MAE_total` | `MAE_defective` | `MAE_design_count` | `Exact Match` (Main / 6 kolom) | `Exact Match` (Leaderboard / 9 kolom) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | 70 | `0.0000` | `0.0000` | `0.0000` | `100.0%` | `100.0%` |
| `val` | 15 | `0.0000` | `0.0000` | `0.0000` | `100.0%` | `100.0%` |
| `test` | 15 | `0.0000` | `0.0000` | `0.0000` | `100.0%` | `100.0%` |
| **`all`** | **100** | **`0.0000`** | **`0.0000`** | **`0.0000`** | **`100.0%`** | **`100.0%`** |
