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
│   ├── task2_non_dl/                                # Pipeline klasik Task 2 (geometry, detector, eval)
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

Pipeline *classical computer vision* untuk inspeksi kubus 3D diimplementasikan secara modular pada paket `src/task2_non_dl/`:

- **`src/task2_non_dl/geometry.py`**: Rekonstruksi latar kuadratik-radial, ekstraksi heksagon sub-piksel (`cv2.fitLine` Huber), penentuan *vanishing center* homogen (`var_even < 0.82 * var_odd`), dan rektifikasi homography sisi `front`, `right`, `top` ke `384x384` (`Mean IoU = 0.9843`, `Min IoU = 0.9320` pada `100` kubus / `300` sisi).
- **`src/task2_non_dl/detector.py` + `template_all/`**: Auto-pemuatan lengkap ke-7 desain referensi (`blue_wizard`, `brown_hat_gnome`, `green_hat_elf`, `pilgrim`, `santa`, `striped_hat_elf`, `witch`), penolakan *border strip*, penggabungan fragmen miring berbasis `cv2.minAreaRect`, pencocokan *multi-template* `design_score`, dan penilaian tri-modal (`miss_sil`, `hole_frac`, `blob_frac`).
- **`src/task2_non_dl/evaluate_and_visualize.py`**: Skrip evaluasi lengkap (`MAE = 0.0000`, `Exact Match = 100.0%` pada `train`, `val`, `test`), pembuatan gambar diagnostik ke `reports/figures/`, serta inferensi CSV 7-referensi tervalidasi ke `submissions/`.
- **`notebooks/Gingerbread_Student_Starter_Colab_ID_v2.ipynb`** (Sel 12 `predict_cube_non_dl`) & **`notebooks/Task2_Non_DL_Pipeline_dan_Analisis.ipynb`**: Integrasi notebook beserta *inline fallback* mandiri untuk Google Colab.
- **Dokumentasi alur kode**: Lihat [docs/TASK2_NON_DL_EXPLANATION.md](docs/TASK2_NON_DL_EXPLANATION.md).
- **Menjalankan evaluasi & visualisasi**:
  ```bash
  python -m src.task2_non_dl.evaluate_and_visualize
  ```

