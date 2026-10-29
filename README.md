# Reproduksi paper Fire 2025

Implementasi [A Comparative Analysis of YOLOv9, YOLOv10, YOLOv11 for Smoke and Fire Detection](https://www.mdpi.com/2571-6255/8/1/26) untuk HPC. Notebook dan `fire2025_reproduction.py` harus berada pada folder yang sama. Tidak ada kode untuk memasang library. `YOLO_AUTOINSTALL=False` juga disetel sebelum Ultralytics diimpor; dataset dan bobot tidak diunduh otomatis.

## Yang disiapkan di HPC

Gunakan kernel Python yang **sudah** menyediakan PyTorch dengan CUDA, Ultralytics yang mendukung YOLOv9t/YOLOv10n/YOLO11, PyYAML, NumPy, Matplotlib dan Pillow. Jalankan sel pemeriksaan environment di notebook untuk melihat versi dan GPU yang terdeteksi. Driver yang dilaporkan `nvidia-smi` tidak menjamin bahwa PyTorch di kernel dapat memakai CUDA.

Unduh dan ekstrak dataset secara terpisah:

- [Smoke and Fire — arsip multipart dari dataset asal](https://github.com/vinchole/zzzccc/tree/main/dataset%20for%20papar). Pertahankan folder `train`, `valid`, dan `test`; setiap split berisi `images` dan `labels`. Gabungkan/ekstrak seluruh bagian arsip dengan perangkat yang tersedia di HPC. Arsip tersebut tidak memiliki `data.yaml`; periksa urutan kelas ID 0/1 dengan sumber dataset dan visualisasi notebook.
- [D-Fire — repo resmi dan tautan unduhan](https://github.com/gaia-solutions-on-demand/DFireDataset). Arahkan `DFIRE_ROOT` atau `--dfire-root` ke folder `D-Fire` dengan layout resmi berikut:

  ```text
  D-Fire/
    train/
      images/   # 17.221 gambar
      labels/   # TXT YOLO; file kosong valid untuk gambar negatif
    test/
      images/   # 4.306 gambar
      labels/
  ```

  Dataset resmi tidak memiliki folder validation. Kode membagi train menjadi **15.498 train dan 1.723 validation** dengan seed default **42**, serta mempertahankan seluruh **4.306 test**. Split ditulis sebagai manifest TXT berisi absolute path dan `data.yaml`; gambar tidak disalin atau dipindahkan. Path `.../images/nama.jpg` tetap dipetakan ke `.../labels/nama.txt` pada sumber, termasuk gambar validation yang berasal dari `train/images`.

  Layout lama [versi Kaggle siap YOLO](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo), `data/{train,val,test}/{images,labels}` di bawah root D-Fire, tetap didukung. Untuk layout ini, train dan val digabung sebelum dibagi ulang.

Seluruh manifest dan hasil audit persiapan data tersimpan di `OUTPUT_ROOT/data_manifests/`, termasuk `provenance.json` yang mencatat seed, jumlah split, checksum, dan audit split sumber. `OUTPUT_ROOT` harus berada di luar kedua folder dataset. Persiapan data hanya membaca gambar/label asli; label kosong tetap disertakan sebagai gambar negatif, sedangkan label hilang atau rusak menyebabkan audit gagal. Seed 42 dan split yang dihasilkan adalah pilihan reproduksi, **bukan klaim identik dengan seed atau pembagian asli penulis paper**.

Sediakan bobot awal di satu folder. Berikut aset resmi yang dapat ditransfer ke HPC tanpa memasang library:

- [yolov9t.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9t.pt)
- [yolov10n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10n.pt)
- [yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)
- [yolo11x.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt)

Bobot COCO ini merupakan asumsi inisialisasi; sumber dan versi bobot awal yang benar-benar digunakan penulis belum dipastikan. Versi file dan hash SHA-256 direkam. User bertanggung jawab memilih kernel dan alokasi GPU sesuai kebijakan HPC yang digunakan.

## Cara memakai

1. Buka `Fire2025_Reproduction.ipynb` dengan kernel HPC yang sudah siap.
2. Isi `SMOKE_ROOT`, `DFIRE_ROOT`, `WEIGHTS_DIR`, dan `OUTPUT_ROOT`. Atur `SMOKE_CLASSES` berdasarkan label sebenarnya. Gunakan `SHOW_SMOKE_LABEL_SAMPLES=True` untuk melihat bbox ID 0/1.
3. Mulai dengan `PREPARE_DATA=True` dan `RUN_TRAINING=False`. Audit semua label, periksa jumlah split dan contoh label bermasalah, serta tinjau daftar 13 job.
4. Atur `RUN_TRAINING=True`; notebook menjalankan semua job berurutan. Pilih `RUN_IDS` untuk menjalankan sebagian jika harus melanjutkan antrean dalam sesi terpisah.
5. Hasil per run tersimpan di `outputs/fire2025/training`, `evaluation`, dan `metadata`; `comparison.csv` menggabungkan referensi paper, evaluasi validation dan evaluasi test. Program menggunakan `best.pt` untuk kedua evaluasi.

Notebook tidak memulai training hanya karena dibuka. Run selesai dilewati ketika dijalankan ulang; run parsial dengan `last.pt` dicoba dilanjutkan. Folder parsial tanpa checkpoint dihentikan untuk ditinjau. Output dengan konfigurasi berbeda tidak akan ditimpa; pilih folder output baru untuk eksperimen baru.

## Menjalankan dari terminal dengan nohup

Semua modifikasi kode dilakukan pada laptop; jangan memasang atau menjalankan agentic tools di HPC. Salin kode yang sudah direview ke HPC dan gunakan environment yang tersedia. Ganti setiap contoh path serta urutan `--smoke-classes` sesuai label yang sudah diverifikasi; urutan argumen adalah ID 0 lalu ID 1.

Mode `audit` (default) hanya memeriksa anotasi dan jumlah gambar, lalu menulis manifest serta provenance. Mode ini memakai standard library Python, tanpa GPU, bobot, `tqdm`, atau impor Ultralytics. Tidak ada training, unduhan, atau instalasi otomatis:

```bash
python -u run_hpc.py --mode audit \
  --smoke-root /path/ke/Smoke_and_Fire --dfire-root /path/ke/D-Fire \
  --output-root /path/ke/outputs/fire2025 \
  --smoke-classes fire smoke
```

`--seed` default 42 menentukan split D-Fire dan seed training. Gunakan seed dan `OUTPUT_ROOT` yang sama pada audit dan training; manifest berbeda tidak akan menimpa output lama. Setelah audit dan label diperiksa, mode `train` memerlukan `--weights-dir`, `tqdm`, dan paket training yang sudah tersedia. Jalankan seluruh 13 eksperimen dalam alokasi GPU yang sah:

```bash
nohup python -u run_hpc.py --mode train \
  --smoke-root /path/ke/Smoke_and_Fire --dfire-root /path/ke/D-Fire \
  --weights-dir /path/ke/weights --output-root /path/ke/outputs/fire2025 \
  --smoke-classes fire smoke > fire2025.log 2>&1 &
echo $!
tail -f fire2025.log
```

Contoh `fire smoke` hanya ilustrasi, bukan urutan kelas yang telah diverifikasi. Tambahkan `--run-ids t4_yolo11n_50` untuk menjalankan satu eksperimen lebih dulu. Jalankan ulang perintah yang sama setelah terputus: hasil lengkap dilewati dan checkpoint parsial dicoba dilanjutkan. Jangan jalankan dua proses terhadap folder output yang sama. `tqdm` menunjukkan jumlah eksperimen yang selesai; Ultralytics menampilkan progres epoch/batch sesuai versinya. Dalam log `nohup`, karakter carriage return dapat membuat progress bar tampak berulang. Jika HPC memakai scheduler, jalankan perintah dalam job GPU yang dialokasikan; `nohup` tidak menggantikan scheduler dan proses tetap tunduk pada batas waktu job. `--device` menggunakan indeks GPU yang terlihat dalam `CUDA_VISIBLE_DEVICES`.

## Cakupan dan asumsi

- Tabel 4: YOLOv9t, YOLOv10n, YOLO11n pada 50 dan 100 epoch, batch 16, Adam. Ukuran 640 dan pemakaian dataset Smoke and Fire adalah interpretasi dari konteks paper; Tabel 3 tidak menjabarkan semua parameter.
- Tabel 6: tiga model yang sama, 100 epoch, 640, batch 64, SGD dan parameter Tabel 5.
- Tabel 7 Smoke and Fire: YOLO11n/YOLO11x, 300 epoch, Tabel 5 dan 416 sesuai teks paper.
- Tabel 7 D-Fire: YOLO11n/YOLO11x, 100 epoch. Karena paper tidak merinci semua hyperparameter untuk kedua run ini, contoh kode memakai Tabel 5 dan 640 sebagai asumsi. `TABLE7_DFIRE_OVERRIDES` tersedia untuk menyelaraskan jika ada bukti tambahan.
- Seed 42, `deterministic=True`, `patience=0`, jumlah worker dan batch evaluasi adalah pilihan implementasi dan dicatat. Lihat `args.yaml` yang dihasilkan untuk seluruh default library yang aktif.
- D-Fire resmi memiliki train/test 17.221/4.306 tanpa validation; split reproduksi 15.498/1.723 dibuat dari train dengan seed default 42. Layout lama Kaggle memiliki train/val/test 14.122/3.099/4.306 dan tetap dapat dipakai. Kesamaan seed maupun keanggotaan split reproduksi dengan pembagian asli penulis paper belum diverifikasi.
- Catatan audit versi D-Fire sebelumnya: 18 bbox luas nol serta 379 bbox melewati batas gambar, termasuk 8 bbox berdimensi >1. Angka ini bukan hasil audit otomatis untuk dataset Anda; periksa `audit_original_splits` di `provenance.json`. Dataset asli dipertahankan; perhatikan peringatan dan gambar yang dilewati oleh versi Ultralytics yang digunakan. Audit ini tidak menjamin seluruh gambar benar secara visual.
- Metrik pada paper tidak selalu menjelaskan split dan aturan pemilihan checkpoint secara lengkap. Tabel hasil memisahkan val dan test agar tidak terjadi klaim perbandingan setara secara otomatis.

## Hasil

Setiap run merekam konfigurasi dan versi environment, fingerprint bobot, checksum daftar split dan label (isi gambar belum di-hash), asumsi, checkpoint `best.pt`/`last.pt`, `results.csv`, plot Ultralytics, metrik bbox agregat, dan AP kelas smoke/fire. `comparison.csv` berisi seluruh 13 referensi paper; run yang belum selesai diberi status `not_finished`.

## Pemeriksaan kode tanpa training

Dari folder proyek pada laptop, jalankan `python -m unittest discover -s tests -v`. Uji memakai fixture gambar/label sintetis, path sintetis untuk memeriksa jumlah resmi, dan model tiruan. Pemeriksaan mencakup split deterministik dan tanpa overlap, jumlah train/validation/test, pasangan gambar-label, label negatif kosong, data sumber yang tetap utuh, kompatibilitas layout lama, serta mode audit tanpa bobot/GPU. Uji ini tidak memerlukan dataset sebenarnya atau GPU dan bukan pengganti pengujian training di HPC.
