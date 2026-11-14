# Proyek Computer Vision: Inspeksi Kue Gingerbread dan Analisis Desain pada Kemasan

**Version Log:**
- `v22-10-2026`: Publikasi awal dokumen proyek CV Gasal 2026/2027
- `v23-10-2026`: Pembaruan *Initial Google Colab* (lihat Bagian 10): `Gingerbread_Student_Starter_Colab_ID_v2.ipynb`
- `v04-11-2026`: Template gambar untuk semua desain gingerbread (`template_all`)
- `v05-11-2026`: *Hidden dataset and the Kaggle competition pages are open!*

---

## 1. Latar Belakang

Sebuah pabrik memproduksi kue *gingerbread* dengan berbagai desain karakter. Setiap desain memiliki bentuk, warna, dan hiasan *icing* yang menjadi ciri produknya. Sebelum dipasarkan, *gingerbread* harus melalui pemeriksaan kualitas untuk memastikan bahwa bentuk dan hiasannya sesuai dengan desain yang telah disetujui.

Dalam proses produksi, sebagian produk dapat mengalami kerusakan. Sebagai contoh, ujung topi atau kaki dapat terpotong, bagian tengah *gingerbread* dapat berlubang, atau sebagian hiasan wajah dan pakaian dapat hilang. Lokasi, jenis, dan luas kerusakan tersebut dapat bervariasi.

Pabrik juga menggunakan kemasan berbentuk kubus dengan gambar *gingerbread* pada permukaannya. Setiap sisi kemasan dapat menampilkan jumlah dan kombinasi desain yang berbeda. Pemeriksaan kemasan diperlukan untuk mengetahui berapa banyak gambar yang muncul, desain apa saja yang ditampilkan, dan apakah terdapat gambar yang cacat.

Dalam proyek ini, mahasiswa diminta membangun sistem *computer vision* untuk menyelesaikan dua tugas tersebut. Masing-masing tugas harus diselesaikan menggunakan **pendekatan *non-deep learning* dan pendekatan *deep learning***, kemudian membandingkan hasil keduanya.

---

## 2. Tujuan Pembelajaran

Setelah menyelesaikan proyek ini, mahasiswa diharapkan mampu:
1. Mengembangkan alur pemrosesan citra untuk pemeriksaan kualitas dan pengenalan desain menggunakan berbagai teknik *computer vision*, baik *non-deep learning* maupun *deep learning*.
2. Menggunakan fitur visual, pencocokan fitur, dan estimasi transformasi geometris untuk menganalisis citra.
3. Menggunakan model *deep learning* untuk menyelesaikan masalah yang sama.
4. Menangani variasi posisi, rotasi, skala, perspektif, pencahayaan, dan tingkat kerusakan.
5. Mengevaluasi kinerja sistem serta menjelaskan kelebihan, keterbatasan, dan pola kesalahan masing-masing pendekatan.

---

## 3. Dataset

Dataset dibuat secara sintetis menggunakan gambar *gingerbread* yang berasal dari foto produk nyata. Sumber data menyediakan tujuh desain karakter. Dataset yang dibagikan dapat menggunakan satu atau beberapa desain tersebut. Gunakan daftar referensi dalam paket data untuk mengetahui desain yang tersedia.

Dataset terdiri atas:
- **Data latih**: digunakan untuk mengembangkan metode dan melatih model.
- **Data validasi**: digunakan untuk memilih parameter, ambang keputusan (*threshold*), dan konfigurasi model.
- **Data uji yang disertai label, apabila tersedia**: digunakan untuk mengevaluasi metode setelah konfigurasi ditetapkan.
- **Data evaluasi tersembunyi**: digunakan untuk penilaian akhir. Label jawabannya disimpan oleh dosen.

Gambar referensi merupakan contoh desain yang dianggap sempurna. Status sempurna atau cacat ditentukan berdasarkan perubahan yang sengaja ditambahkan terhadap referensi tersebut.

### Struktur Berkas Utama

| Folder atau Berkas | Kegunaan |
| :--- | :--- |
| `references/` | Gambar desain sempurna beserta identitas desain (`designs.json`) |
| `single/` | Citra gingerbread individual untuk Tugas 1 |
| `single_labels.csv` | Identitas desain dan label sempurna/cacat untuk Tugas 1 |
| `cube/` | Citra kemasan kubus untuk Tugas 2 |
| `cube_face_labels.csv` | Jumlah gambar, identitas desain, dan informasi cacat pada setiap sisi |
| Berkas anotasi dan mask | Informasi tambahan (`*_annotations.json`, `*_coco.json`, masks) untuk pengembangan metode, pelatihan, visualisasi, dan analisis kesalahan |

> **Aturan Ketat Evaluasi Data Tersembunyi:**  
> Pada saat melakukan prediksi terhadap data evaluasi tersembunyi, sistem **hanya boleh menggunakan citra RGB dan gambar referensi yang disediakan**. *Mask*, *bounding box*, transformasi geometris, jumlah objek sebenarnya, dan label kualitas dari data evaluasi tidak boleh digunakan sebagai masukan.

---

## 4. Tugas 1: Deteksi Cacat pada Gingerbread Individual

### Deskripsi Masalah
Setiap citra menampilkan satu *gingerbread*. Sistem harus menentukan apakah *gingerbread* tersebut **sempurna** atau **cacat**.

Jenis cacat yang dapat muncul meliputi:

| Jenis Cacat | Deskripsi |
| :--- | :--- |
| Bagian tepi terpotong (`boundary_cut`) | Sebagian bentuk gingerbread hilang dari batas luarnya, misalnya ujung topi, bahu, atau kaki |
| Lubang di bagian dalam (`interior_hole`) | Sebagian material di dalam bentuk gingerbread hilang |
| Hiasan atau fitur hilang (`missing_feature`) | Sebagian icing atau detail visual hilang, sementara bentuk luar tetap dipertahankan |

Luas dan lokasi cacat dapat berbeda antarcitra. Pada cacat hiasan, kerusakan yang besar dapat menghilangkan beberapa detail yang berdekatan. Perubahan posisi, rotasi, ukuran, perspektif, atau pencahayaan merupakan variasi pengambilan gambar. Sistem perlu menangani variasi tersebut agar tetap dapat mengenali kerusakan produk.

### Masukan dan Keluaran
- **Masukan**: citra RGB satu *gingerbread* dan referensi desain yang tersedia.
- **Keluaran utama**: keputusan sempurna atau cacat untuk setiap citra.

Simpan hasil prediksi dalam berkas CSV dengan kolom berikut:

| Kolom | Isi |
| :--- | :--- |
| `image_id` | Identitas citra sesuai dataset |
| `is_defective` | `0` untuk sempurna dan `1` untuk cacat |
| `defect_score` | Skor dugaan cacat; nilai yang lebih tinggi menunjukkan dugaan cacat yang lebih kuat |

> **Catatan:** Skor tidak harus berupa probabilitas yang telah dikalibrasi. Jelaskan cara menghitung skor dan menentukan ambang keputusan dalam laporan. Lokalisasi daerah cacat dapat digunakan untuk mendukung keputusan dan visualisasi. Target utama tugas ini adalah **klasifikasi kualitas pada tingkat citra**.  
> *(Untuk submisi Kaggle Task 1, gunakan format `sample_submission_task1_kaggle.csv` dengan kolom `image_id,is_defective`.)*

---

## 5. Tugas 2: Menghitung Gingerbread, Mendeteksi Cacat, dan Menghitung Jenis Desain pada Kemasan Kubus

### Deskripsi Masalah
Setiap citra menampilkan sebuah kemasan kubus dari satu sudut pandang sehingga terlihat tiga sisi. Pada setiap sisi terdapat satu sampai enam gambar *gingerbread*. Jumlah gambar, jenis desain, susunan, dan orientasi kemasan dapat bervariasi. Desain yang sama dapat muncul lebih dari satu kali, dan satu sisi dapat memuat satu atau beberapa jenis desain.

Pada keseluruhan citra kubus, terdapat **maksimal dua gambar gingerbread yang cacat**, termasuk kemungkinan tidak ada gambar yang cacat. Cacat diterapkan pada gambar gingerbread yang tercetak pada kemasan; bentuk kubus tetap utuh. Seluruh analisis dilakukan dari satu citra RGB.

Sistem harus menghasilkan, untuk setiap sisi:
1. **Jumlah seluruh gambar gingerbread**, termasuk gambar sempurna dan cacat.
2. **Jumlah gambar gingerbread yang cacat**, tanpa membedakan identitas desainnya.
3. **Jumlah jenis desain unik yang muncul**, khusus untuk evaluasi leaderboard.

Penamaan sisi sesuai anotasi dataset:
- `front`: sisi depan.
- `right`: sisi kanan.
- `top`: sisi atas.

Pelajari konvensi penamaan tersebut dari contoh dan anotasi data latih. Penamaan sisi harus konsisten meskipun orientasi kemasan bervariasi. Mahasiswa wajib mencoba **pendekatan *non-deep learning* dan pendekatan *deep learning***, serta membandingkan hasil keduanya.

### Perbedaan Jumlah Gambar dan Jumlah Jenis Desain
Satu *instance* berarti satu kemunculan gambar gingerbread. Sebagai contoh, sebuah sisi menampilkan tiga gambar elf hijau dan satu gambar penyihir. Jika satu gambar elf hijau cacat, maka:
- Jumlah seluruh gambar gingerbread adalah **4**.
- Jumlah gambar gingerbread yang cacat adalah **1**.
- Jumlah jenis desain unik adalah **2**, yaitu elf hijau dan penyihir.

Gambar yang cacat tetap termasuk dalam jumlah seluruh gambar dan tetap dihitung sebagai desain asalnya. Perbedaan rotasi, ukuran, perspektif, atau kondisi cacat tidak menjadikan suatu gambar sebagai desain baru. Sistem tidak diwajibkan melaporkan identitas desain maupun jumlah gambar untuk setiap desain. Namun, untuk menghitung jenis desain unik, sistem harus mampu membedakan gambar dari desain yang sama dan gambar dari desain yang berbeda.

### Masukan dan Keluaran
- **Masukan**: citra RGB kemasan kubus dan referensi desain yang tersedia.
- **Keluaran utama**: berkas CSV dengan **satu baris untuk setiap citra**. Untuk tugas utama, keluaran terdiri atas 7 kolom (1 kolom identitas citra + 6 kolom jumlah gambar). Untuk leaderboard, tambahkan 3 kolom jumlah desain unik sehingga terdapat **10 kolom** termasuk `image_id`.

| Kolom | Isi | Penggunaan |
| :--- | :--- | :--- |
| `image_id` | Identitas citra kubus | Tugas utama dan leaderboard |
| `front_total` | Jumlah seluruh gambar gingerbread pada sisi depan | Tugas utama dan leaderboard |
| `right_total` | Jumlah seluruh gambar gingerbread pada sisi kanan | Tugas utama dan leaderboard |
| `top_total` | Jumlah seluruh gambar gingerbread pada sisi atas | Tugas utama dan leaderboard |
| `front_defective` | Jumlah gambar gingerbread cacat pada sisi depan | Tugas utama dan leaderboard |
| `right_defective` | Jumlah gambar gingerbread cacat pada sisi kanan | Tugas utama dan leaderboard |
| `top_defective` | Jumlah gambar gingerbread cacat pada sisi atas | Tugas utama dan leaderboard |
| `front_design_count` | Jumlah jenis desain unik pada sisi depan | Khusus leaderboard |
| `right_design_count` | Jumlah jenis desain unik pada sisi kanan | Khusus leaderboard |
| `top_design_count` | Jumlah jenis desain unik pada sisi atas | Khusus leaderboard |

Format ini berlaku baik ketika citra hanya memuat satu jenis desain maupun beberapa jenis desain.

### Contoh Keluaran
```csv
image_id,front_total,right_total,top_total,front_defective,right_defective,top_defective,front_design_count,right_design_count,top_design_count
cube_contoh,4,2,3,1,0,1,2,1,3
```
Artinya:
- **Sisi depan**: 4 gambar gingerbread dari 2 jenis desain, dengan 1 gambar cacat.
- **Sisi kanan**: 2 gambar gingerbread dari 1 jenis desain, tanpa gambar cacat.
- **Sisi atas**: 3 gambar gingerbread dari 3 jenis desain, dengan 1 gambar cacat.

Secara keseluruhan citra memuat 9 gambar gingerbread dengan 2 gambar cacat. Jumlah desain unik dihitung **secara terpisah pada setiap sisi**.

### Ketentuan Keluaran
- Setiap `image_id` harus muncul tepat satu kali, dan seluruh citra yang diminta harus memiliki prediksi.
- Semua nilai jumlah harus berupa bilangan bulat nonnegatif.
- Jumlah gambar cacat pada suatu sisi tidak boleh melebihi jumlah seluruh gambar pada sisi tersebut.
- Jumlah jenis desain unik tidak boleh melebihi jumlah gambar pada sisi tersebut maupun jumlah desain yang tersedia.
- Jika seluruh gambar pada suatu sisi berasal dari desain yang sama, jumlah jenis desain uniknya adalah `1`.
- Nilai `0` digunakan untuk menyatakan tidak adanya gambar cacat, bukan kolom kosong.
- Ground truth pada dataset memuat maksimal 2 gambar cacat per citra kubus.
- Berkas leaderboard harus mengikuti nama dan urutan kolom pada `sample_submission.csv` (`sample_submission_task2.csv`) yang disediakan.

---

## 6. Solusi Tugas Kelompok

Kelompok perlu menerapkan pendekatan *non-deep learning* dan *deep learning* untuk Tugas 1 dan Tugas 2.

### A. Pendekatan Non-Deep Learning
Bangun metode yang menggunakan fitur visual yang dirancang secara manual dan pencocokan terhadap referensi. Komponen yang dapat digunakan antara lain:
- Segmentasi berdasarkan warna, intensitas, atau bentuk.
- Deteksi tepi dan operasi morfologi.
- Deskriptor lokal seperti SIFT atau ORB.
- Pencocokan fitur antara citra masukan dan referensi.
- RANSAC untuk menolak pencocokan yang tidak konsisten dan mengestimasi transformasi.
- Perbandingan bentuk atau tampilan setelah penyelarasan.
- Pengelompokan hasil pencocokan untuk membedakan beberapa instansi.

Pilih komponen yang sesuai dengan masalah, dan jelaskan hubungan antara fitur yang digunakan, proses pencocokan, dan keputusan akhir sistem.

### B. Pendekatan Deep Learning
Bangun metode yang menggunakan representasi atau model *neural network* sebagai komponen utama pengenalan atau pemeriksaan kualitas. Pilihan pendekatan antara lain:
- Menggunakan model pralatih sebagai ekstraktor fitur, kemudian mencocokkan fitur citra dengan referensi.
- Melatih atau melakukan *fine-tuning* model klasifikasi kualitas.
- Menggunakan model deteksi atau segmentasi untuk membantu menemukan instansi gingerbread.
- Menggabungkan fitur *neural network* dengan pencocokan dan penyelarasan geometris.

> **Penggunaan Model Pralatih:** Penggunaan fitur dari model *deep learning* pralatih diperbolehkan, termasuk ketika bobot model dibekukan (*frozen*). Pelatihan model dari awal bersifat opsional. Menggunakan *classifier* tradisional di atas fitur *neural network* tetap termasuk pendekatan *deep learning* karena representasi citranya berasal dari *neural network*.

---

## 7. Prosedur Eksperimen

1. Periksa struktur dataset, referensi desain, dan contoh anotasi.
2. Tampilkan contoh citra sempurna dan cacat, termasuk variasi jenis dan luas kerusakan.
3. Kembangkan pendekatan *non-deep learning* dan *deep learning* untuk kedua tugas.
4. Tentukan parameter dan ambang keputusan menggunakan data latih dan validasi.
5. Tetapkan konfigurasi akhir sebelum mengevaluasi data uji atau evaluasi tersembunyi.
6. Simpan prediksi dalam format yang telah ditentukan.
7. Bandingkan kinerja dan lakukan analisis kesalahan.

Gunakan pembagian data yang sama untuk membandingkan kedua pendekatan. Catat model pralatih, sumber bobot, pustaka, *seed*, dan perangkat komputasi yang digunakan.

---

## 8. Evaluasi dan Analisis

Evaluasi dilakukan secara terpisah untuk **pendekatan *non-deep learning* dan pendekatan *deep learning*** menggunakan pembagian data yang sama. Gunakan data validasi untuk memilih parameter dan ambang keputusan. Data uji digunakan untuk evaluasi akhir.

### Tugas 1: Deteksi Gingerbread Cacat
Gunakan **kelas cacat sebagai kelas positif**. Laporkan:
- Confusion matrix
- Precision
- Recall
- F1-score
- Balanced accuracy

Jika data evaluasi mencakup beberapa desain, laporkan hasil keseluruhan dan hasil per desain untuk melihat kemampuan generalisasi metode.

### Tugas 2: Penghitungan pada Kemasan Kubus
Evaluasi mengikuti format keluaran **satu baris per citra**, dengan prediksi untuk sisi `front`, `right`, dan `top`.

| Aspek | Ukuran Evaluasi |
| :--- | :--- |
| Jumlah seluruh gingerbread per sisi | Mean Absolute Error (MAE) untuk `front_total`, `right_total`, dan `top_total` |
| Jumlah gingerbread cacat per sisi | MAE untuk `front_defective`, `right_defective`, dan `top_defective` |
| Jumlah jenis desain unik per sisi — khusus leaderboard | MAE untuk `front_design_count`, `right_design_count`, dan `top_design_count` |
| Ketepatan seluruh jawaban pada suatu sisi | Persentase sisi yang seluruh nilai prediksinya tepat (*exact match per sisi*) |
| Ketepatan seluruh jawaban pada suatu citra | Persentase citra yang seluruh nilai prediksi pada ketiga sisinya tepat (*exact match per citra*) |

- **MAE** menunjukkan rata-rata selisih absolut antara jumlah prediksi dan jumlah sebenarnya (makin kecil makin baik; `0` berarti tepat seluruhnya). Hitung MAE terpisah untuk setiap aspek dengan merata-ratakan kesalahan pada seluruh citra dan ketiga sisi; laporkan pula MAE masing-masing sisi untuk mengetahui apakah perspektif tertentu lebih sulit dianalisis.
- Untuk **tugas utama**: *exact match per sisi* mensyaratkan jumlah total dan cacat sama-sama benar; *exact match per citra* mensyaratkan seluruh **6 nilai prediksi** pada ketiga sisi benar.
- Untuk **leaderboard**: jumlah jenis desain unik juga diperhitungkan; *exact match per sisi* mensyaratkan ketiga nilai pada sisi tersebut benar, sedangkan *exact match per citra* mensyaratkan seluruh **9 nilai prediksi** benar.
- Contoh: jika suatu sisi memiliki 4 gingerbread, 1 cacat, dan 2 jenis desain, prediksi `(4, 1, 3)` benar untuk 2 aspek tugas utama, tetapi belum memenuhi *exact match* sisi untuk leaderboard karena `design_count` salah.

### Analisis Hasil
Selain melaporkan metrik, bandingkan kedua pendekatan dan bahas:
- Pengaruh jenis dan tingkat keparahan cacat terhadap hasil prediksi.
- Pengaruh rotasi, perspektif, ukuran gambar, dan susunan gingerbread.
- Kemampuan metode dalam menangani desain yang tidak digunakan saat pelatihan.
- Contoh keberhasilan dan kegagalan, disertai visualisasi serta penjelasan kemungkinan penyebabnya.
- Waktu komputasi dan kebutuhan sumber daya.

Untuk **Tugas 2**, bedakan kesalahan menghitung objek, kesalahan mendeteksi cacat, dan kesalahan menghitung jenis desain unik. Perhatikan pula kinerja pada sisi yang benar-benar mengandung gambar cacat, karena prediksi selalu nol dapat terlihat baik apabila sebagian besar sisi tidak memiliki cacat.

---

## 9. Luaran yang Dikumpulkan

Kumpulkan:
1. **Kode (semua berkas di-zip):**
   - **Kode atau notebook yang dapat dijalankan**, mencakup pembacaan data, prapemrosesan, kedua pendekatan, inferensi, evaluasi, dan penyimpanan prediksi.
   - **Petunjuk untuk menjalankan kode**, termasuk dependensi, lokasi data, dan **bobot model yang diperlukan**.
2. **File presentasi** berisi laporan eksperimen, uraian metode, pengaturan eksperimen, hasil kuantitatif, visualisasi, dan analisis kesalahan.

Presentasi harus menjelaskan alasan pemilihan metode serta kondisi ketika masing-masing pendekatan berhasil atau gagal. Kesimpulan harus didukung oleh hasil eksperimen.

---

## 10. Template

1. **Dataset Training (`gingerbread_seed2026_students_training.zip`)**:  
   URL dari starter notebook: https://drive.google.com/file/d/1M1IRpFds1VyIkc9IJBehrprcjAEaPXSw/view?usp=sharing
2. **Initial Google Colab**: `Gingerbread_Student_Starter_Colab_ID_v2.ipynb` (tersimpan di `notebooks/Gingerbread_Student_Starter_Colab_ID_v2.ipynb`)

---

## 12. Leaderboard: Evaluasi Menggunakan Dataset Tersembunyi

Leaderboard diselenggarakan menggunakan *hidden test dataset* untuk mengevaluasi kemampuan metode menangani desain dan distribusi kualitas yang lebih beragam.

| Tahap | Data yang Disediakan | Fokus Kegiatan |
| :--- | :--- | :--- |
| Dua minggu pertama | Dataset awal dengan satu desain gingerbread beserta referensi dan labelnya | Mengembangkan, melatih, dan memvalidasi pendekatan *non-deep learning* serta *deep learning* |
| Setelah dua minggu | Enam *template* desain gingerbread tambahan sehingga tersedia tujuh desain referensi | Menyesuaikan metode agar dapat memanfaatkan referensi dari berbagai desain |
| Pembukaan evaluasi tersembunyi | Citra uji yang mencakup seluruh tujuh desain, tanpa label kualitas | Menghasilkan prediksi sempurna/cacat untuk pengumpulan ke leaderboard |

- **Tugas 1:** Dataset tersembunyi memiliki proporsi produk sempurna dan cacat yang dapat berbeda dari dataset awal maupun antar desain. Sistem **tidak boleh mengasumsikan bahwa separuh citra selalu cacat atau bahwa setiap desain memiliki rasio kualitas yang sama**. Kewajiban mengimplementasikan kedua pendekatan (*non-DL* dan *DL*) tetap berlaku; tiap kelompok cukup menggunakan 1 teknik *computer vision* untuk *final leaderboard*, namun presentasi wajib membandingkan keduanya serta membahas generalisasi saat referensi bertambah dari 1 ke 7 desain.
- **Tugas 2:** Hidden dataset memuat variasi desain, jumlah gambar, susunan, orientasi, serta kondisi cacat. Jumlah gambar dan jumlah jenis desain merupakan dua target yang berbeda.

### 12. Dataset Tersembunyi (*Hidden Dataset*) dan *Kaggle Competition Pages*
- **Template semua desain (`template_all`)**: boleh di-augmentasi untuk menyelesaikan baik Tugas 1 maupun Tugas 2 (klik tautan pada dokumen sumber kelas jika ingin mengunduh ulang).
- **Dataset tersembunyi (`hidden_students.zip`)**:  
  https://drive.google.com/file/d/1Jkl9WL75xv_Gxb24kaDKE48Q1q6iYFYJ/view?usp=sharing
- **Kaggle Competition Pages:**
  - **Gingerbread Tugas 1 — Non-Deep Learning:** https://www.kaggle.com/t/bf2e55de99c04fca9eb6b89f12ebbc99
  - **Gingerbread Task 1 — Deep Learning:** https://www.kaggle.com/t/eb7d78511623430da9bedadda6abfc6e
  - **Gingerbread Task 2 — Non-Deep Learning:** https://www.kaggle.com/t/fdc879988e044c91a67e0ec091b489b6
  - **Gingerbread Task 2 — Deep Learning:** https://www.kaggle.com/t/041fcbab6bb24778a9e777417ec302c6

---

## 13. Pembagian Kelompok, Jadwal, dan Bobot Penilaian

1. **Kelompok:** 3–4 orang (boleh *mixed* Kelas A dan Kelas B) melalui spreadsheet `[Shared] Kelompok Mini Project CV 2026/2027`.
2. **Jadwal:**
   - Demo dengan asisten dosen (tidak harus selesai): **12 Nov & 19 Nov 2026** (jadwal diacak)
   - Pengumpulan kode sumber: **20 Nov 2026**
   - Pengumpulan presentasi: **25 Nov 2026**
   - Presentasi: **26 Nov, 1 Des, 3 Des, dan 8 Des 2026**
3. **Bobot Penilaian:**
   - **Mini-Project Development and Demo (15%)**:
     - Demo dengan asisten dosen: 10%
     - *Performance vs baseline*: 5%
   - **Mini-Project Presentation (20%)**:
     - Presentasi: 10%
     - Tanya jawab: 10%

---

## 14. Ketentuan Presentasi

### File Presentasi
- Dibuat dalam format PDF (`ppt`/`canva` -> ekspor PDF) dengan konten minimal:
  - Identitas anggota kelompok
  - Deskripsi Dataset
  - Untuk masing-masing Tugas 1 dan Tugas 2:
    - Penjelasan Task
    - Penjelasan metode
    - Metodologi/desain eksperimen
    - Hasil kuantitatif dan analisis kesalahan
    - Visualisasi
  - Summary
- Submit di slot Scele maksimal **Rabu, 25 November 2026 pukul 23:59** dengan format penamaan **`<Kelas>_<Nama Kelompok>.pdf`** (contoh: `A_Nama kelompokku.pdf`).

### Detail Jadwal & Aturan Hari-H
- Lokasi & penguji sesuai sheet jadwal di `[Shared] Kelompok Mini Project CV 2026/2027`:
  - **Kamis, 26 November 2026**: Pak Febrian (`13.00–13.50`, `A2.51`) / Bu Laksmita (`13.00–14.40`, `A3.05`)
  - **Selasa, 1 Desember 2026**: Pak Febrian (`13.00–13.50`, `A6.02`) / Bu Laksmita (`13.00–14.40`, `A6.08`)
  - **Kamis, 3 Desember 2026**: Pak Febrian (`13.00–13.50`, `A2.51`) / Bu Laksmita (`13.00–14.40`, `A3.05`)
  - **Selasa, 8 Desember 2026**: Pak Febrian (`13.00–13.50`, `A6.02`) / Bu Laksmita (`13.00–14.40`, `A6.08`)
- Kelompok wajib hadir sepanjang kelas pada hari presentasi yang ditentukan.
- Tidak ada toleransi keterlambatan; anggota yang terlambat atau tidak hadir saat waktu presentasi dimulai **tidak mendapat nilai presentasi**.
- Durasi per kelompok **25 menit**: 15 menit presentasi, 8 menit tanya jawab, 2 menit *buffer*/pergantian.

---

## Lampiran Teknis Dataset (`README.txt` dari Paket Data)

Ringkasan catatan teknis dari arsip resmi `gingerbread_seed2026_students_training.zip` dan `hidden_students.zip`:
- **Training zip**:
  - `Design ID` bernilai `1..7`; kualitas (`is_defective`) adalah label terpisah.
  - Mask PNG (`visible`/`reference`/`defect`) bersifat otoritatif untuk lubang dan bagian terpotong; poligon kontur COCO hanya merepresentasikan batas luar (tidak meng-encode lubang interior).
  - Cube mask adalah PNG 16-bit ID instansi (`0` = background, instansi mulai `1`). Face mask: `0` = background, `1` = `front`, `2` = `right`, `3` = `top`.
  - Di satu citra kubus, maksimal hanya **2 instansi** yang cacat di seluruh citra (bukan 2 per sisi).
- **Hidden test zip (`hidden_students.zip`)**:
  - Hanya berisi citra RGB (`single/hidden/images/`, `cube/hidden/images/`), indeks (`test_single.csv`, `test_cube.csv`), dan template CSV (`sample_submission_task1.csv`, `sample_submission_task1_kaggle.csv`, `sample_submission_task2.csv`).
  - Paket *hidden test* **tidak menyertakan folder `references/`** — gunakan `references/` dari paket training / `template_all` secara terpisah.
  - Untuk submisi Kaggle Tugas 1, konversi keluaran 3 kolom starter menjadi 2 kolom (`image_id,is_defective`) sesuai `sample_submission_task1_kaggle.csv`.
