# Gingerbread CV Inspection

Computer vision pipeline for gingerbread defect inspection and 3D cube packaging counting using non-deep learning and deep learning approaches.

> **Dokumentasi Lengkap Proyek:** Lihat [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md) untuk rincian masalah, format CSV/Kaggle, metrik evaluasi, jadwal, dan ketentuan pengumpulan.

---

## Ringkasan Tugas

| Tugas | Deskripsi | Pendekatan Wajib | Target Output |
| :--- | :--- | :--- | :--- |
| **Task 1** | Deteksi cacat gingerbread individual (`boundary_cut`, `interior_hole`, `missing_feature`) | Non-Deep Learning & Deep Learning | `image_id,is_defective,defect_score` *(Kaggle: `image_id,is_defective`)* |
| **Task 2** | Menghitung total gingerbread, jumlah cacat, & jumlah desain unik per sisi kemasan kubus (`front`, `right`, `top`) | Non-Deep Learning & Deep Learning | CSV 7 kolom (tugas utama) / 10 kolom (leaderboard) |

---

## Struktur Repository

```text
gingerbread-cv-inspection/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── PROJECT_SPEC.md                              # Spesifikasi lengkap proyek (Bagian 1–14)
│   └── TASK2_NON_DL_EXPLANATION.md                  # Penjelasan alur kode Tugas 2 Non-Deep Learning
├── data/                                            # Dataset lokal (di-gitignore, tidak ikut push)
│   ├── gingerbread_seed2026_students_training.zip   # Dataset training (taruh di sini)
│   └── hidden_students.zip                          # Hidden test dataset (taruh di sini)
├── template_all/                                    # 7 desain referensi lengkap & designs.json
├── notebooks/
│   ├── Gingerbread_Student_Starter_Colab_ID_v2.ipynb
│   └── Task2_Non_DL_Pipeline_dan_Analisis.ipynb     # Notebook analisis & evaluasi Tugas 2 Non-DL
├── src/
│   ├── task1_non_dl/                                # Metode klasik Task 1 (SIFT/ORB, RANSAC, morfologi)
│   ├── task1_dl/                                    # Metode deep learning Task 1
│   ├── task2_non_dl/                                # Pipeline klasik Task 2 (geometry, detector, baselines, eval)
│   ├── task2_dl/                                    # Metode deep learning Task 2
│   └── utils/                                       # Loader dataset, evaluasi metrik, visualisasi
├── models/                                          # Bobot model terlatih (di-gitignore untuk file besar)
├── submissions/                                     # Output CSV prediksi Kaggle & evaluasi
└── reports/
    └── figures/                                     # Visualisasi diagnostik & PDF presentasi kelompok
```

---

## Setup Lokal / Google Colab

### 1. Clone & Instalasi Dependensi
```bash
git clone https://github.com/K10-Computer-Vision/gingerbread-cv-inspection.git
cd gingerbread-cv-inspection
pip install -r requirements.txt
```

### 2. Menyiapkan Dataset (Wajib Manual / Di-gitignore)
Dataset berukuran >100 MB sehingga tidak disimpan di GitHub. Unduh arsip berikut lalu simpan ke folder `data/` (atau ekstrak ke subfolder `data/extracted/`):
- **Training Dataset (`gingerbread_seed2026_students_training.zip`)**: [Google Drive](https://drive.google.com/file/d/1M1IRpFds1VyIkc9IJBehrprcjAEaPXSw/view?usp=sharing)
- **Hidden Test Dataset (`hidden_students.zip`)**: [Google Drive](https://drive.google.com/file/d/1Jkl9WL75xv_Gxb24kaDKE48Q1q6iYFYJ/view?usp=sharing)
- **Template Desain (`template_all`)**: Gunakan tautan resmi dari dokumen kelas untuk referensi/augmentasi 7 desain.

> **Catatan Hidden Test:** `hidden_students.zip` tidak menyertakan folder `references/`. Pastikan pipeline mengarah ke `references/` dari paket training saat melakukan inferensi pada data `hidden`.

### 3. Starter Notebook
Gunakan `notebooks/Gingerbread_Student_Starter_Colab_ID_v2.ipynb` untuk eksplorasi awal data, validasi format prediksi, dan evaluasi metrik standar (Confusion Matrix, Precision, Recall, F1, Balanced Accuracy, MAE, & Exact Match).

---

## Tautan Kaggle Competition

- **Task 1 — Non-Deep Learning:** https://www.kaggle.com/t/bf2e55de99c04fca9eb6b89f12ebbc99
- **Task 1 — Deep Learning:** https://www.kaggle.com/t/eb7d78511623430da9bedadda6abfc6e
- **Task 2 — Non-Deep Learning:** https://www.kaggle.com/t/fdc879988e044c91a67e0ec091b489b6
- **Task 2 — Deep Learning:** https://www.kaggle.com/t/041fcbab6bb24778a9e777417ec302c6

---

## Alur Kerja Tim (Git Workflow)

1. Selalu buat branch baru untuk setiap eksperimen/fitur, misalnya `feat/task1-orb-ransac` atau `feat/task2-yolo`.
2. Gunakan format commit `[Topic] Aksi yang dilakukan`, misalnya:
   - `[Feature] Menambahkan pipeline segmentasi warna untuk Task 1 Non-DL`
   - `[Docs] Memperbarui hasil eksperimen Task 2 pada README`
3. Pastikan file `.zip`, citra dataset, dan bobot model besar (`.pt`/`.pth`) tidak ikut ter-commit.

---

## Implementasi Task 2 — Non-Deep Learning (`feat/task2-non-dl`)

Kami merancang pipeline *classical computer vision* modular pada `src/task2_non_dl/` tanpa komponen pembelajaran mendalam dan hanya memanfaatkan kanal RGB serta 7 referensi resmi `template_all/`. Seluruh model berbagi *front-end* geometri 3D yang sama agar evaluasi murni menguji akurasi pengenalan desain dan deteksi cacat.

### Arsitektur, Model, & Langkah Matematis

1. **Rekonstruksi Geometri 3D & Rektifikasi (`src/task2_non_dl/geometry.py`)**:
   - **Model Latar Kuadratik-Radial**: Kami memodelkan iluminasi latar tiap kanal RGB memakai $B(u,v) = c_0 + c_1 u + c_2 v + c_3 (u^2 + v^2)$ dengan *inlier re-fit* dua tahap ($r < 12.0$) dan selisih Euclidean $D(u,v) = \|I(u,v) - B(u,v)\|_2$ untuk mengekstrak siluet kubus secara adaptif.
   - **Ekstraksi Heksagon & Rektifikasi Kanonik**: Keenam sudut terluar kubus kami lokalisasi pada presisi sub-piksel lewat regresi garis Huber (`cv2.fitLine`, `DIST_HUBER`), kemudian titik temu rusuk interior kami hitung melalui proyeksi *vanishing point* homogen beserta disambiguasi paritas (`var_even < 0.82 * var_odd`). Ketiga sisi (`front`, `right`, `top`) lalu kami rektifikasi menjadi bidang kanonik `384x384` lewat homography proyeksi (`cv2.getPerspectiveTransform`).
2. **Segmentasi Cookie & Pipeline Utama (`src/task2_non_dl/detector.py`)**:
   - **Segmentasi & Pemulihan Fragmen**: Pada tiap sisi kanonik, kami mengestimasi warna kardus melalui median area interior, menyegmentasi cookie memakai jarak warna, menolak artefak strip tepi rusuk, serta menyatukan kembali fragmen cookie miring yang terbelah oleh lubang memakai `cv2.minAreaRect`.
   - **Main Multi-Template NCC (`predict_cube_non_dl`)**: Tiap instansi kami sejajarkan melalui korelasi *coarse-to-fine* ber-mask (`TM_CCOEFF_NORMED` `0.35x` diikuti penyempurnaan skala), dinormalisasi pencahayaannya per kanal memakai regresi affine kokoh ($\hat{T} = g \cdot T + b$), lalu dinilai memakai residu toleran-geser $3 \times 3$ ($R_{\text{shift}}(x,y)$) ke dalam tiga indikator cacat ortogonal: `miss_sil` (`boundary_cut`), `hole_frac` (`interior_hole`), dan `blob_frac` (`missing_feature`).
3. **Tiga Model Pembanding Non-DL (`src/task2_non_dl/baselines.py`)**:
   - **SIFT + RANSAC (`predict_cube_sift_ransac`)**: Mengup-scale ROI ke target `380 px` (`2.0x–3.6x`), mengekstrak fitur float SIFT 128-D (`nfeatures=850`, `contrastThreshold=0.012`), menyaring pasangan `NORM_L2` via Lowe ratio (`0.82`), lalu melakukan *two-stage pose fitting*: RANSAC 4-DOF diikuti *refit* analitik SVD Umeyama pada *tight inliers* ($e \le 2.2\text{ px}$), verifikasi fotometrik 3 kandidat teratas, serta fallback ke `Contour + HSV` bila pose lemah.
   - **ORB + RANSAC (`predict_cube_orb_ransac`)**: Menggunakan detektor FAST+BRIEF (`nfeatures=950`, `fastThreshold=5`, `scaleFactor=1.12`, `nlevels=10`) pada target ROI `320 px` dengan `NORM_HAMMING` (`ratio=0.82`), lalu menerapkan *refit* Umeyama, verifikasi fotometrik, serta evaluasi cacat $3 \times 3$ yang konsisten.
   - **Contour + HSV (`predict_cube_contour_hsv`)**: Pendekatan bebas-*keypoint* dan bebas-NCC yang meluruskan cookie menurut orientasi momen kontur $\theta = \frac{1}{2}\arctan2(2\mu_{11}, \mu_{20}-\mu_{02})$, menormalisasi kanal `V`, lalu mengukur *cosine similarity* HSV global (`[H,S]` `16x6` + `[V]` `8`), 3 zona vertikal, serta profil 6-strip beserta lebar konturnya dengan mekanisme *trimmed mean* dan *dual-anchor* atas/bawah saat `ar_obs < 0.88 * ar_ref`.

### Analisis Analitik (Mengapa Metode A Dipilih Dibanding B)

- **Regresi Kuadratik-Radial + Line-Fitting Huber vs Thresholding Global / OLS**: Permukaan latar pada citra kubus memiliki gradasi vignetting halus sehingga *thresholding* statis memotong tepi terang atau memasukkan bayangan; di sisi lain, regresi Huber menekan pencilan sudut kubus yang patah ketimbang OLS kuadratik biasa.
- **Multi-Template Masked NCC vs Keypoint Matching (SIFT/ORB) untuk Pipeline Utama**: Permukaan cookie bertipe kecil (`~50–100 px`) dengan tekstur frosting yang repetitif. NCC ber-mask mengukur kecocokan struktur intensitas padat secara langsung di seluruh piksel tanpa bergantung pada gradien sudut lokal yang sering menipis akibat kompresi atau pencahayaan datar.
- **Deskriptor Float SIFT (`NORM_L2`) vs Biner ORB (`NORM_HAMMING`)**: Secara teoretis dan empiris, SIFT 128-D mempertahankan histogram orientasi gradien kontinu sub-piksel sehingga jauh lebih stabil menghadapi perubahan skala homography antarsisi (`front`/`right`/`top`), sedangkan bit-string BRIEF pada ORB lebih cepat (`0.579 s` vs `0.633 s/citra`) namun lebih sensitif terhadap blur resampling.
- **Two-Stage 4-DOF Similarity RANSAC + Umeyama SVD Refit vs Single-Stage RANSAC / 8-DOF Homography**: Bidang sisi sudah kami rektifikasi oleh homography tahap 1, sehingga sisa transformasi cookie di dalam sisi bersifat isometrik+skala (4-DOF). Memaksakan homography 8-DOF memicu overfitting numerik bila jumlah *inlier* terbatas, sementara *refit* analitik Umeyama pada himpunan *tight inliers* menghilangkan jitter kuantisasi RANSAC.
- **Normalisasi Shading Affine + Residu Toleran-Geser 3x3 vs Selisih RGB Langsung / SSIM Global**: Ketiga sisi kubus mengalami iluminasi berbeda; koreksi affine per kanal menyelaraskan kontras dan offset sebelum komparasi. Toleransi geser $3 \times 3$ meredam eror sub-piksel batas dekorasi, sedangkan pemisahan ortogonal (`miss_sil`, `hole_frac`, `blob_frac`) menjaga kesalahan kecil pada tepi agar tidak menutupi lubang interior atau dekorasi yang hilang.
- **Profil 6-Strip Trimmed + Dual-Anchor HSV vs Histogram Global / Hu-Moments Murni**: Cookie cacat ringan dan cookie sehat dari desain serupa kerap berbagi distribusi warna global dan momen invarian global hampir identik; pemecahan menjadi 6 strip terkalibrasi siluet bersama *trimmed mean* mengisolasi potongan lokal (`boundary_cut`) dan hilangnya ornamen tengah (`interior_hole`) tanpa mengacaukan identifikasi desain.

### Tantangan & Langkah Optimasi (`v1 -> v2 -> v3`)

Tantangan tersulit meliputi paritas rotasi antarkuadran saat rektifikasi, serpihan cookie miring terbelah lubang, perbedaan iluminasi antarsisi, sedikitnya keypoint pada skala kecil, serta bias skor pada area cacat saat identifikasi desain.

- **Pipeline Utama (`v1 -> v2`)**: Kami menambahkan pemulihan 7 referensi penuh `template_all/`, disambiguasi paritas vanishing center, penolakan strip tepi, penggabungan fragmen miring (`cv2.minAreaRect`), serta kalibrasi tri-modal (`miss_sil`, `hole_frac`, `blob_frac`).
- **Model Pembanding (`v1 -> v2 -> v3`)**: Pada `v2`, kami memperkenalkan *adaptive ROI upscaling*, perluasan toleransi rasio Lowe, serta segmentasi zona 3-tingkat dengan normalisasi kanal `V`. Pada optimasi final `v3`, kami mengganti estimasi pose sekali jalan menjadi **Two-Stage RANSAC + Umeyama SVD Refit** dengan verifikasi fotometrik 3 hipotesis teratas (`corr`, `IoU`, `p55`) serta *fallback* aman pada pose lemah, dan meningkatkan `Contour + HSV` memakai **profil lebar 6-strip, *trimmed mean*, serta penyelarasan *dual-anchor* atas/bawah**.

### Analisis Empiris & Hasil Terukur

Evaluasi pada `100` citra berlabel (`70` train, `15` val, `15` test) dan `200` citra *hidden* ([Task2_Non_DL_Pipeline_dan_Analisis.ipynb](notebooks/Task2_Non_DL_Pipeline_dan_Analisis.ipynb)) — termasuk metrik khusus sisi cacat (`MAE_def_on_def_faces`) dan studi generalisasi `1-ref` vs `7-ref` — menghasilkan kinerja berikut:

| Model (`v3`) | Kecepatan (`100` citra) | `MAE_total` / `MAE_def` (`all`) | `MAE_def_on_def_faces` | `MAE_design_count` (`all`) | `Exact Match Main` (`all` / `test`) | `Exact Match LB` (`all` / `test`) | Kesamaan Hidden (`200` citra) | Kaggle Public Score Tercatat | Berkas Submisi (`submissions/`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Main Multi-Template NCC** | `2.043 s/citra` | `0.0000` / `0.0000` | `0.0000` | `0.0000` | `100.0%` / `100.0%` | `100.0%` / `100.0%` | `100.0%` | **`0.00000`** | `task2_ncc_lb.csv`, `task2_ncc_main.csv` |
| **SIFT + RANSAC (`v3`)** | `0.633 s/citra` | `0.0000` / `0.0033` | `0.0098` | `0.0033` | `99.0%` / `100.0%` | `98.0%` / `100.0%` | `93.0%` | `0.03333` (`v1`) → `0.01296` (`v2`) | `task2_sift_lb.csv`, `task2_sift_main.csv` |
| **ORB + RANSAC (`v3`)** | `0.579 s/citra` | `0.0000` / `0.0133` | `0.0098` | `0.0033` | `97.0%` / `100.0%` | `96.0%` / `93.3%` | `92.0%` | `0.03333` (`v1`) → `0.01666` (`v2`) | `task2_orb_lb.csv`, `task2_orb_main.csv` |
| **Contour + HSV (`v3`)** | `0.281 s/citra` | `0.0000` / `0.0367` | `0.0882` | `0.0267` | `89.0%` / `100.0%` | `81.0%` / `93.3%` | `80.0%` | `0.10555` (`v1`) | `task2_contour_lb.csv`, `task2_contour_main.csv` |

Pada uji generalisasi `cube_0817258b17ffb7d2` di `hidden_students.zip`, pemakaian `1` referensi (`blue_wizard` tunggal, skor Kaggle `0.08333`) hanya menghasilkan `NCC = 0.352–0.590` terhadap karakter baru (`santa`, `witch`, `striped_hat_elf`, `pilgrim`, `brown_hat_gnome`, `green_hat_elf`), sedangkan `template_all` 7 referensi mengangkat `NCC` menjadi `0.650–0.978` hingga `0.00000` di Kaggle. Pada pemisahan kesalahan per kategori cacat (`boundary_cut` [n=47], `interior_hole` [n=32], `missing_feature` [n=31]), `Main NCC` mencetak `100.0%` pada ketiga jenis, `SIFT` & `ORB` meraih `97.9% / 100.0% / 100.0%`, sedangkan `Contour + HSV` tercepat (`0.281 s/citra`) namun turun menjadi `80.9%` pada `boundary_cut` (`MAE_def_on_def_faces = 0.0882`).

- **Dokumentasi kode & hasil visual (`reports/figures/`)**: Lihat [docs/TASK2_NON_DL_EXPLANATION.md](docs/TASK2_NON_DL_EXPLANATION.md), [notebooks/Task2_Non_DL_Pipeline_dan_Analisis.ipynb](notebooks/Task2_Non_DL_Pipeline_dan_Analisis.ipynb), serta figur siap pakai di `reports/figures/` (`task2_non_dl_pipeline_stages.png`, `task2_non_dl_segmentation_merge.png`, `task2_non_dl_tri_modal_anatomy.png`, `task2_non_dl_keypoint_umeyama.png`, `task2_non_dl_contour_hsv_v3.png`, `task2_non_dl_optimization_v1_v2_v3.png`, `task2_non_dl_model_comparison.png`, `task2_non_dl_failure_cases.png`).
- **Menjalankan evaluasi & visualisasi**:
  ```bash
  python -m src.task2_non_dl.evaluate_and_visualize
  ```

### Referensi

Bradski, G. (2000). The OpenCV library. *Dr. Dobb's Journal*, *25*(11), 120–125.

Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. *Communications of the ACM*, *24*(6), 381–395. https://doi.org/10.1145/358669.358692

Gonzalez, R. C., & Woods, R. E. (2018). *Digital image processing* (4th ed.). Pearson.

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). Array programming with NumPy. *Nature*, *585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2

Hartley, R., & Zisserman, A. (2004). *Multiple view geometry in computer vision* (2nd ed.). Cambridge University Press.

Hu, M.-K. (1962). Visual pattern recognition by moment invariants. *IRE Transactions on Information Theory*, *8*(2), 179–187. https://doi.org/10.1109/TIT.1962.1057692

Huber, P. J. (1964). Robust estimation of a location parameter. *The Annals of Mathematical Statistics*, *35*(1), 73–101. https://doi.org/10.1214/aoms/1177703732

Lewis, J. P. (1995). Fast template matching. In *Proceedings of Vision Interface '95* (pp. 120–123). Canadian Image Processing and Pattern Recognition Society.

Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision*, *60*(2), 91–110. https://doi.org/10.1023/B:VISI.0000029664.99615.94

OpenCV Team. (2026). *OpenCV 4.x documentation: Camera calibration and 3D reconstruction (`calib3d`), 2D features framework (`features2d`), and image processing (`imgproc`)*. https://docs.opencv.org/4.x/

Rahadianti, L., & Rachmadi, M. F. (2026). *Basic image processing 1 – Spatial domain* [Slide perkuliahan]. CSCE604133 Computer Vision, Fakultas Ilmu Komputer, Universitas Indonesia.

Rahadianti, L., & Rachmadi, M. F. (2026). *Basic image processing 2 – Frequency domain* [Slide perkuliahan]. CSCE604133 Computer Vision, Fakultas Ilmu Komputer, Universitas Indonesia.

Rahadianti, L., & Rachmadi, M. F. (2026). *Feature matching, recognition (object detection) and registration* [Slide perkuliahan]. CSCE604133 Computer Vision, Fakultas Ilmu Komputer, Universitas Indonesia.

Rahadianti, L., & Rachmadi, M. F. (2026). *Image features and recognition (classification)* [Slide perkuliahan]. CSCE604133 Computer Vision, Fakultas Ilmu Komputer, Universitas Indonesia.

Rahadianti, L., & Rachmadi, M. F. (2026). *Image segmentation and morphology* [Slide perkuliahan]. CSCE604133 Computer Vision, Fakultas Ilmu Komputer, Universitas Indonesia.

Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. In *2011 International Conference on Computer Vision (ICCV)* (pp. 2564–2571). IEEE. https://doi.org/10.1109/ICCV.2011.6126544

Szeliski, R. (2022). *Computer vision: Algorithms and applications* (2nd ed.). Springer. https://doi.org/10.1007/978-3-030-34372-9

Umeyama, S. (1991). Least-squares estimation of transformation parameters between two point patterns. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *13*(4), 376–380. https://doi.org/10.1109/34.88573

