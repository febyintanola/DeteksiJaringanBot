# Perbandingan Canopy + Kruskal MST dengan Louvain

Sumber data: `https://vt.tiktok.com/ZSHoa9M6H/`
Dibuat: 2026-06-25 05:10:16 UTC

## Tabel Perbandingan dengan Penelitian Terdahulu / Baseline

| Aspek | Baseline Louvain | Penelitian Ini |
| --- | --- | --- |
| Platform | Graf sosial umum; pada benchmark ini diterapkan ke komentar TikTok yang sama | TikTok, berbasis komentar publik dan interaksi antar pengguna |
| Metode clustering | Louvain Community Detection berbasis optimasi modularity | Canopy Clustering sebagai pre-clustering, dilanjutkan Kruskal MST clustering |
| Metode graph reduction | Tidak memakai reduksi MST; community detection dijalankan pada graf berbobot penuh | Kruskal MST membentuk backbone graf, lalu edge lemah pada MST dipotong |
| Output sistem | Komunitas/cluster pengguna dan skor kecurigaan menggunakan modul scoring yang sama untuk pembanding | Graf pengguna, cluster terdeteksi, skor kecurigaan, alasan cluster, dan daftar akun mencurigakan |
| Kebaruan yang ditonjolkan | Berfokus pada pembentukan komunitas graf berdasarkan modularity. | Menggabungkan pre-clustering berbasis konten, reduksi backbone MST, dan scoring multi-sinyal untuk deteksi jaringan bot. |

## Tabel Benchmark Empiris

| Ukuran | Metode | Edge awal | Edge setelah reduksi | Reduksi | Total detik | Cluster | Modularity | Silhouette | Conductance | Akun mencurigakan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | Canopy + Kruskal MST (Usulan) | 1069 | 49 | 95.4% | 0.780 | 49 | 0.026 | 0.117 | 0.760 | 0 |
| 100 | Louvain Community Detection (Baseline) | 1069 | 1069 | 0.0% | 0.234 | 3 | 0.172 | 0.192 | 0.573 | 24 |
| 200 | Canopy + Kruskal MST (Usulan) | 2189 | 97 | 95.6% | 2.533 | 97 | 0.031 | 0.098 | 0.835 | 2 |
| 200 | Louvain Community Detection (Baseline) | 2191 | 2191 | 0.0% | 0.491 | 3 | 0.238 | 0.104 | 0.529 | 30 |
| 500 | Canopy + Kruskal MST (Usulan) | 5767 | 243 | 95.8% | 3.972 | 238 | 0.048 | 0.054 | 0.775 | 2 |
| 500 | Louvain Community Detection (Baseline) | 5777 | 5777 | 0.0% | 1.091 | 5 | 0.414 | 0.103 | 0.401 | 58 |
| 1000 | Canopy + Kruskal MST (Usulan) | 11920 | 477 | 96.0% | 9.379 | 477 | 0.060 | 0.010 | 0.824 | 2 |
| 1000 | Louvain Community Detection (Baseline) | 12015 | 12015 | 0.0% | 1.731 | 4 | 0.594 | 0.068 | 0.252 | 68 |

Catatan: Louvain dipakai sebagai baseline karena sama-sama bekerja pada graf berbobot, sehingga perbandingannya lebih langsung dibanding baseline non-graf. Nilai silhouette dihitung pada ruang embedding teks yang sama agar kualitas pemisahan cluster dapat dibandingkan.
