# Bot Network Detector (TikTok)

Prototype web untuk deteksi jaringan bot pada kolom komentar video TikTok.

## Fitur v1
- Input URL video TikTok dan parameter (α mention, γ konten, k-NN, T1/T2, MST overlay)
- Ambil komentar via TikTokApi (async) hingga N komentar
- Bangun graf berbobot (mention + konten sederhana), MST backbone, clustering dari MST
- Skor “terduga bot” unsupervised + metrik ringkas (modularity, conductance mean)
- Visualisasi graf interaktif (Cytoscape) dan ringkasan cluster
- Ekspor dapat ditambahkan (CSV/JSON)

## Prasyarat
- ms_token TikTok diset di environment agar TikTokApi bisa membuat sesi
  - Windows PowerShell:
    ```powershell
    $env:ms_token = "your_ms_token_here"
    ```
- (Opsional) set browser untuk TikTokApi (chromium/firefox/webkit):
  ```powershell
  $env:TIKTOK_BROWSER = "chromium"
  ```
- Pastikan repo `TikTok-Api` ada sebagai saudara folder proyek ini (struktur):
  ```
  Prototyping/
    TikTok-Api/
    botnet-web/
  ```
  Kode akan menambahkan path `TikTok-Api` ke `sys.path` secara otomatis.

## Menjalankan secara lokal
1) Install dependencies (disarankan pakai virtualenv)
   ```powershell
   cd "c:\Users\Ola\OneDrive\Documents\Kuliahnya Intan\Kebutuhan Tugas Akhir\Prototyping\botnet-web"
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2) Jalankan server
   ```powershell
   $env:ms_token = "<isi_ms_token>"
   uvicorn app.main:app --reload --port 8000
   ```
3) Buka browser ke `http://localhost:8000/`

## Catatan Algoritma
- Mention edge: dibuat dari deteksi `@username` dalam teks komentar (v1 memetakan ke user yang juga berkomentar)
- Konten: v1 memakai Jaccard token overlap (sederhana) untuk baseline → bisa diupgrade ke TF‑IDF / embedding
- MST: dihitung sebagai minimum spanning tree atas jarak `1/(similarity+eps)`; cluster diperoleh dengan memotong edge MST yang di bawah median bobot per komponen
- Skoring: kombinasi rata-rata bobot internal cluster dan konsentrasi derajat
- Canopy T1/T2: placeholder di UI, implementasi ANN + canopy akan ditambahkan bertahap

## Docker (opsional)
Build image:
```powershell
cd "c:\Users\Ola\OneDrive\Documents\Kuliahnya Intan\Kebutuhan Tugas Akhir\Prototyping\botnet-web"
docker build -t botnet-web:latest .
```
Jalankan:
```powershell
# mount TikTok-Api agar dapat diimport
docker run -it --rm -p 8000:8000 `
  -e ms_token=$env:ms_token `
  -e TIKTOK_BROWSER=chromium `
  -e TIKTOK_API_PATH=/ext/TikTok-Api `
  -v "c:\Users\Ola\OneDrive\Documents\Kuliahnya Intan\Kebutuhan Tugas Akhir\Prototyping\TikTok-Api":/ext/TikTok-Api `
  botnet-web:latest
```

## Next Steps
- Ganti Jaccard dengan TF‑IDF/embedding + hnswlib ANN untuk canopy
- Tambah edges co-thread dari reply traversal
- Ekspor CSV/JSON, simpan run artifacts
- Tooltip edukatif lebih lengkap dan mode dark
- Penanganan rate-limit & retry TikTokApi

## Laporan Kompleksitas Waktu
- Tujuan: dokumentasi teoretis (Big‑O) per tahap, dan bukti empiris dari benchmark.

### Kompleksitas Teoretis (asumsi N komentar, V pengguna, E edges)
- Fetch komentar: tergantung API, kira‑kira `O(N)` untuk parsing respons.
- Parsing teks: tokenisasi sederhana `O(N)`.
- Canopy (ANN + assignment): dengan HNSW `~O(log V)` per lookup; keseluruhan `~O(V log V)` untuk embedding + penempatan canopies (approksimasi).
- Build graf: pembuatan edges bernilai kemiripan (mention + konten) biasanya `O(V·k)` dengan k‑NN lokal, atau `O(V^2)` bila pairwise penuh.
- MST clustering: MST via Prim/Kruskal `O(E log V)`; pemotongan dan komponen `O(E)`.
- Skoring cluster: agregasi metrik lokal `O(E)`.

Catatan: Bila memakai k‑NN (k konstan kecil), maka `E ≈ O(V·k)` dan MST `O(V·k log V)`.

### Benchmark Empiris
- File: `benchmark.py` menjalankan pipeline untuk beberapa ukuran komentar dan menyimpan hasil.
- Output: `data/bench_results.json` dan `data/bench_results.csv` berisi waktu `build_sec`, `cluster_sec`, `score_sec` serta ukuran graf.

Menjalankan benchmark:
```powershell
cd "c:\Users\Ola\OneDrive\Documents\Kuliahnya Intan\Kebutuhan Tugas Akhir\Prototyping\botnet-web"
$env:ms_token = "<isi_ms_token>"; $env:BENCH_VIDEO_URL = "<url_video_tiktok>"
python benchmark.py
```

Interpretasi cepat:
- `build_sec` ≈ biaya konstruksi graf (dipengaruhi k‑NN / pairwise).
- `cluster_sec` ≈ MST + pemotongan.
- `score_sec` ≈ agregasi metrik.

### Menyisipkan ke Laporan
- Ringkas tabel hasil dari `bench_results.csv` (size → waktu tiap tahap).
- Grafik batang sederhana (Chart.js) untuk memvisualisasikan waktu per tahap.
- Tuliskan hubungan teoretis vs empiris, dan faktor dominan (misal build graf).
