# Penjelasan Pipeline Tugas 2 — Pendekatan Non-Deep Learning

Dokumen ini merangkum alur kerja paket `src/task2_non_dl/` beserta integrasinya pada *notebook* sebagai bagian pembagian tugas **Task 2 (Non-Deep Learning)**. Seluruh tahapan di bawah dirancang secara analitis menggunakan teknik *classical computer vision* tanpa melatih jaringan saraf tiruan.

---

## 1. Rekonstruksi Geometri 3D Kubus (`src/task2_non_dl/geometry.py`)

Pada modul ini, kami mengubah citra kemasan kubus 3D berlatar gradasi menjadi tiga citra sisi persegi kanonik berukuran `384x384` piksel (`CANONICAL_FACE_SIZE = 384`):

1. **Pemodelan Latar (`fit_background_model`)**: Kami menyampel piksel sepanjang bingkai tepi luar citra (`border = 20`) untuk meregresikan model iluminasi kuadratik-radial `[1, u, v, u^2 + v^2]` per kanal RGB dengan penyaringan *outlier* dua tahap (*inlier re-fit* bila residu `> 15.0`). Selisih terhadap model latar kemudian kami haluskan menggunakan `cv2.GaussianBlur` `(5, 5)` guna menekan *noise* latar sebelum menghitung peta jarak warna (`dist_map`).
2. **Ekstraksi Heksagon Sub-Piksel (`extract_cube_hexagon`)**: Kami membangun *foreground mask* menggunakan ambang adaptif berbasis persentil latar `max(4.2, min(12.0, p98.5 + 1.8))`, membersihkannya lewat operasi morfologi (*close* `9x9` lalu *open* `5x5`), mengambil *convex hull*, serta menyederhanakannya menjadi poligon 6 simpul (`cv2.approxPolyDP` dan helper `_reduce_polygon_to_hex` / `_collapse_short_edges`). Selanjutnya, kami menyaring titik kontur di bagian tengah tiap segmen rusuk, memasang garis kokoh `cv2.fitLine(..., cv2.DIST_HUBER)` melalui `_fit_line_huber`, lalu menghitung perpotongan garis (`_intersect_2d_lines`) untuk memperoleh 6 sudut luar kubus berpresisi sub-piksel searah jarum jam dimulai dari simpul paling atas.
3. **Penentuan Titik Pertemuan Rusuk Dalam (`compute_vanishing_center`)**: Kami memproyeksikan pasangan rusuk luar yang sejajar secara 3D dalam koordinat homogen (`_solve_center_for_parity`) untuk memperoleh tiga *vanishing point*, lalu mengarahkan ketiga rusuk dalam dan mencari titik temu (`v_center`) lewat *least-squares*. Kami juga melakukan disambiguasi paritas simpul rusuk `{1, 3, 5}` vs `{0, 2, 4}` menggunakan `_point_in_triangle` serta perbandingan deviasi warna intra-kuadran `_eval_partition_variance` (beralih ke paritas genap jika `var_even < 0.82 * var_odd`, termasuk untuk rotasi ekstrem seperti `cube_c22298eb291a8bab`).
4. **Segmentasi & Rektifikasi Tiga Sisi (`segment_cube_faces`, `rectify_face`)**: Kami membagi heksagon kubus mengelilingi `v_center` menjadi tiga kuadran searah jarum jam: kuadran paling atas—atau lebih terang berdasarkan persentil ke-75 luminansi jika selisih tinggi `< 18.0` piksel—kami tetapkan sebagai `top`, diikuti `right` dan `front`. Keempat sudut tiap kuadran kami tata menjadi urutan standar `[kiri-atas, kanan-atas, kanan-bawah, kiri-bawah]` lewat `_orient_quad_upright`, lalu kami luruskan memakai transformasi *homography* (`cv2.getPerspectiveTransform` dan `cv2.warpPerspective`).

---

## 2. Segmentasi Cookie, Pencocokan Multi-Template, & Deteksi Cacat (`src/task2_non_dl/detector.py`)

Setelah ketiga sisi diluruskan ke ruang kanonik `384x384`, kami mendeteksi jumlah cookie, status cacat, dan jumlah desain unik pada tiap sisi dengan cakupan penuh 7 desain referensi (`template_all/`):

1. **Penyiapan & Pelengkapan Referensi (`_augment_references_from_template_dir`, `prepare_reference_cache`)**: Jika kamus referensi yang dikirim pemanggil berisi `< 7` desain (seperti pada paket training awal yang hanya memuat `blue_wizard`), kami otomatis melengkapinya dari folder `template_all/` (`blue_wizard`, `brown_hat_gnome`, `green_hat_elf`, `pilgrim`, `santa`, `striped_hat_elf`, `witch`). Setiap citra desain kami skalakan ke separuh dimensi alaminya (`0.5x`) sambil menyimpan `mask`, `mask_inner` (ter-erosi `3x3`), serta `coarse_cache` dan `fine_cache` agar perubahan skala berulang tidak dihitung ulang.
2. **Segmentasi Instansi, Penolakan Strip Tepi, & Penggabungan Fragmen (`segment_face_cookies`, `_merge_split_fragments`)**: Kami mengestimasi warna kardus dominan dari median area interior sisi (`[12:-12, 12:-12]`), menghitung jarak warna Euclidean (`diff > 22.0`), meredam pinggir 10 piksel, lalu menolak komponen artefak tepi rusuk ber-*bounding box* tipis (`ar_box > 3.6`, `min(bw, bh) < 26`, atau menyentuh tepi dengan `ar_box > 2.9`). Karena cacat lubang atau terpotong pada cookie miring membuat rasio aspek sumbu-x/y tampak melebar, kami menghitung rasio aspek terorientasi lewat `cv2.minAreaRect` (`_oriented_aspect_ratio`) di `_merge_split_fragments` (`1.40 <= rot_ar_merged <= 2.85`, luas gabungan `<= 18500`) untuk menyatukan kembali fragmen terbelah seperti pada cookie miring `cube_172314569950fcad` dan `cube_a142b6699f57f366`.
3. **Penyelarasan Multi-Desain & Penilaian Tri-Modal (`align_and_score_cookie`, `_extract_aligned_design_stats`)**: Untuk setiap cookie, kami memutar ROI ke sekitar sumbu utama momen citra (`_major_axis_angle_deg`), lalu menjalankan tahap *coarse* FFT skala `0.35x` dengan cache `roi_s_cache` per sudut. Kami memilih hingga 3 kandidat *coarse* beragam per desain (termasuk skala `peer_scale` untuk cookie terpotong), memperhalusnya pada jendela *fine* ber-*mask* (sambil memangkas kandidat desain rendah hanya bila `max_coarse >= 0.76` agar cookie cacat tetap dievaluasi pada ke-7 desain), kemudian memilih desain pemenang berdasarkan skor gabungan `_extract_aligned_design_stats`:
   `design_score = 0.45 * ncc + 0.55 * ncc_on_solid + 0.35 * vis_cov - 0.010 * p55_err`.
   Setelah normalisasi pencahayaan per kanal RGB (`_fit_shading_affine`) dengan toleransi geser `3x3` (`res_shift_map`), kami mengevaluasi tiga mode cacat:
   - **`miss_sil` (`boundary_cut`)**: Fraksi siluet template yang hilang terhadap `0.085`; untuk desain dikenal (`ncc >= 0.42` atau `ncc_on_solid >= 0.55` & `coverage >= 0.38` saat `7` referensi aktif), kami mensyaratkan `res_shift_map > 19.0` agar area putih/cerah alami tidak dianggap hilang.
   - **`hole_frac` (`interior_hole`)**: Fraksi area lubang di dalam siluet solid yang menyerupai warna kardus terhadap `0.055`.
   - **`blob_frac` (`missing_feature`)**: Fraksi gumpalan residu *icing* hilang (`res_shift_map > 18.0` ter-erosi `3x3`) terhadap `0.085`.
   - **`defect_ratio`**: Nilai maksimum dari ketiga rasio ternormalisasi tersebut; cookie kami klasifikasikan cacat bila `defect_ratio > 1.0`.
4. **Estimasi Desain Unik & Orkestrasi (`_estimate_unique_designs`, `predict_cube_non_dl`)**: Saat ketujuh referensi lengkap tersedia (`num_refs >= 7`), kami menghitung langsung kardinalitas unik `design_id` yang cocok pada sisi tersebut agar cookie cacat dari desain dikenal tidak terhitung sebagai klaster tambahan (tetap mempertahankan klastering histogram HSV `[12, 4]` sebagai *fallback* bila `< 7` referensi). Seluruh tahapan digabung di `predict_cube_non_dl` dengan aturan batas maksimal `2` cookie cacat per citra kubus dan format keluaran `6` atau `9` kunci sesuai `LEADERBOARD`.

---

## 3. Model Pembanding, Skrip Evaluasi, & Integrasi Notebook

- **`src/task2_non_dl/baselines.py` (`v3`)**: Kami menyediakan tiga model pembanding resmi yang berbagi *front-end* geometri 3D identik (`geometry.py`) serta segmentasi cookie (`detector.py`):
  1. `predict_cube_sift_ransac`: SIFT (`nfeatures=850`, `contrastThreshold=0.012`) pada *adaptive ROI upscaling* (`target 380 px`) + Lowe ratio (`0.82`) + *two-stage pose fitting* (4-DOF RANSAC menuju *refit* analitik SVD Umeyama pada $e \le 2.2\text{ px}$), verifikasi fotometrik 3 hipotesis teratas (`corr`, `IoU`, `p55`), evaluasi residu $3 \times 3$, serta *fallback* `Contour + HSV` saat pose lemah.
  2. `predict_cube_orb_ransac`: Detektor biner ORB (`nfeatures=950`, `fastThreshold=5`, `scaleFactor=1.12`, `nlevels=10`) pada target ROI `320 px` dengan `NORM_HAMMING` (`ratio=0.82`) serta alur *refit* Umeyama dan verifikasi fotometrik yang konsisten.
  3. `predict_cube_contour_hsv`: Pelurusan berbasis momen kontur utama $\theta$ tanpa *keypoint*, normalisasi median kanal `V`, ekstraksi vektor HSV global + 3 zona + 6 strip beserta profil lebar siluet (`widths`), *trimmed mean* 6-strip, dan penyelarasan *dual-anchor* atas/bawah saat rasio aspek menyusut (`ar < 0.88 * ref_ar`).
- **`template_all/` & `src/task2_non_dl/evaluate_and_visualize.py`**: Kami menyertakan `template_all/designs.json` untuk memetakan 7 desain lengkap, melaporkan `MAE_def_on_def_faces` dan MAE per sisi (`front`/`right`/`top`), mengeksekusi tolok ukur penuh `100` citra kubus berlabel, memperbarui tiga figur diagnostik di `reports/figures/`, serta mengekspor `task2_ncc_lb.csv` dan `task2_ncc_main.csv` ke `submissions/submission_tugas2_nondl/`.
- **`notebooks/Gingerbread_Student_Starter_Colab_ID_v2.ipynb` (Sel 12)**: Kami menyinkronkan fungsi `predict_cube_non_dl` (termasuk *inline fallback* Colab mandiri yang langsung mendukung `template_all` 7 desain saat dijalankan di repo).
- **`notebooks/Task2_Non_DL_Pipeline_dan_Analisis.ipynb`**: Kami merapikan menjadi 24 sel penuh berisi 8 visualisasi matematis & analisis lengkap (rekonstruksi 3D, tahap segmentasi & fragmen miring `task2_non_dl_segmentation_merge.png`, anatomi residu tri-modal `task2_non_dl_tri_modal_anatomy.png`, studi generalisasi `1-ref` vs `7-ref`, pose `RANSAC -> Umeyama` `task2_non_dl_keypoint_umeyama.png`, profil 6-strip HSV `task2_non_dl_contour_hsv_v3.png`, progresi optimasi `task2_non_dl_optimization_v1_v2_v3.png`, perbandingan `task2_non_dl_model_comparison.png`, dan galeri kegagalan `task2_non_dl_failure_cases.png`), audit ke-8 berkas submisi `submissions/submission_tugas2_nondl/task2_*_{lb,main}.csv`, serta daftar referensi.

---

## 4. Ringkasan Hasil Evaluasi Terverifikasi

| Model (`v3`) | Kecepatan (`100` citra) | `MAE_defective` (`all` / `test`) | `MAE_def_on_def_faces` | `MAE_design_count` (`all` / `test`) | `Exact Match Main` (`all` / `test`) | `Exact Match LB` (`all` / `test`) | Kesamaan Hidden `200` dgn Main | Berkas CSV (`submissions/submission_tugas2_nondl/`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Main Multi-Template NCC** | `2.043 s/citra` | `0.0000` / `0.0000` | `0.0000` | `0.0000` / `0.0000` | `100.0%` / `100.0%` | `100.0%` / `100.0%` | `100.0%` | `task2_ncc_lb.csv`, `task2_ncc_main.csv` |
| **SIFT + RANSAC (`v3`)** | `0.633 s/citra` | `0.0033` / `0.0000` | `0.0098` | `0.0033` / `0.0000` | `99.0%` / `100.0%` | `98.0%` / `100.0%` | `93.0%` | `task2_sift_lb.csv`, `task2_sift_main.csv` |
| **ORB + RANSAC (`v3`)** | `0.579 s/citra` | `0.0133` / `0.0000` | `0.0098` | `0.0033` / `0.0222` | `97.0%` / `100.0%` | `96.0%` / `93.3%` | `92.0%` | `task2_orb_lb.csv`, `task2_orb_main.csv` |
| **Contour + HSV (`v3`)** | `0.281 s/citra` | `0.0367` / `0.0000` | `0.0882` | `0.0267` / `0.0222` | `89.0%` / `100.0%` | `81.0%` / `93.3%` | `80.0%` | `task2_contour_lb.csv`, `task2_contour_main.csv` |

- **Pemisahan Error per Target (`100` kubus, `total` / `defective` / `design_count`) & Kategori Cacat (`boundary_cut` [47] / `interior_hole` [32] / `missing_feature` [31])**:
  - `Main NCC`: `0 / 0 / 0` error (`100.0% / 100.0% / 100.0%` akurasi kategori cacat).
  - `SIFT + RANSAC`: `0 / 1 / 1` error (`97.9% / 100.0% / 100.0%`).
  - `ORB + RANSAC`: `0 / 3 / 1` error (`97.9% / 100.0% / 100.0%`).
  - `Contour + HSV`: `0 / 11 / 8` error (`80.9% / 96.9% / 100.0%`).

- **Akurasi Segmentasi Geometri 3D (`100` kubus / `300` sisi)**: `Mean IoU = 0.9843`, `Min IoU = 0.9320`.
- **Akurasi Evaluasi End-to-End 7 Referensi (`100` kubus berlabel, `432.39s`)**:

| Split | Jumlah Kubus | `MAE_total` | `MAE_defective` | `MAE_design_count` | `Exact Match` (Main / 6 kolom) | `Exact Match` (Leaderboard / 9 kolom) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | 70 | `0.0000` | `0.0000` | `0.0000` | `100.0%` | `100.0%` |
| `val` | 15 | `0.0000` | `0.0000` | `0.0000` | `100.0%` | `100.0%` |
| `test` | 15 | `0.0000` | `0.0000` | `0.0000` | `100.0%` | `100.0%` |
| **`all`** | **100** | **`0.0000`** | **`0.0000`** | **`0.0000`** | **`100.0%`** | **`100.0%`** |

- **Ringkasan Inferensi *Hidden Test* (`200` citra, `947.35s`)**:
  - `front_total`: `mean=1.80` (`min=1, max=3`), `design_count dist={1: 149, 2: 35, 3: 16}` (`mean=1.33`).
  - `right_total`: `mean=2.00` (`min=1, max=3`), `design_count dist={1: 130, 2: 53, 3: 17}` (`mean=1.44`).
  - `top_total`: `mean=2.00` (`min=1, max=3`), `design_count dist={1: 126, 2: 58, 3: 16}` (`mean=1.45`).
  - Distribusi total cacat per kubus (`0/1/2`): `{0: 51, 1: 70, 2: 79}` (total `228` instansi cacat terdeteksi).
