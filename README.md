# Face Recognition Attendance

## Ketentuan sebelum menjalankan
- Menggunakan python versi <= 3.9 
- Copy `.env.example` dan rubah hasil copy menjadi `.env` dan batalkan komentar 
- Menyiapkan gambar-gambar dataset di folder `/dataset` dengan subdirektori adalah "user_id"/label.

Contoh:
```bash
├── 0
│   ├── 1.jpg
│   └── 2.jpg
├── 1
│   ├── 1.1.jpg
│   ├── 1.2.jpg
```
## Langkah-langkah sebelum menjalankan server
0. *(Opsional)* buat venv terpisah 
1. Jalankan perintah `pip install -r requirments.txt`
2. Jalankan perintah `python generate_db.py` untuk membuat database
3. Jalankan perintah `python cnn.py` untuk membuat model CNN pertama

## Menjalankan server
1. Jalankan perintah `python app.py`
2. Applikasi siap digunakan di link `127.0.0.1:5000`

## Cara Memasukan Absensi dengan Wajah
1. Buka browser, lalu buka link ke `127.0.0.1:5000`
2. Hadapkan muka ke webcam
3. Tunggu beberapa saat hingga aplikasi mengenali wajah anda

## Cara Mendaftarkan Wajah
1. Dari Homepage, tekan tombol "Registrasi"
2. Masukan nama yang ingin didaftarkan
3. Tekan tombol "Submit & Register Muka"
4. Hadapkan muka ke webcam
5. Tunggu sampai counter menunjukan angka 40
6. Tekan tombol "Relearn", tunggu sampai redirect ke Homepage
7. Wajah sudah terdaftar siap digunakan
