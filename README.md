# Studi Komparatif Empiris Model Generatif (AE, Sparse AE, VAE, Vanilla GAN, DCGAN, DDPM) & Pembuktian Batasan MLP Classifier

Repositori ini berisi implementasi PyTorch modular dan pipeline eksperimen terstandarisasi untuk tugas *Take-Home* mata kuliah **Deep Learning (D6421) — Sesi 07: Generative Models** pada dataset benchmark **Fashion-MNIST** (70.000 citra 28×28 grayscale, split terstratifikasi 80% Train / 10% Validation / 10% Test, `seed=42`, `batch_size=128`).

## Struktur Repositori

```text
.
├── main.py                         # Pipeline end-to-end (MLP proof, 6 model, metrik, 5 figur)
├── report_config.json              # Metadata penulis (Nama, NIM, Kelas, URL GitHub) untuk laporan PDF
├── requirements.txt                # Dependensi pustaka Python & PyTorch
├── scripts/
│   └── download_assets.py          # Pra-unduh Fashion-MNIST dan bobot InceptionV3
├── src/
│   ├── config.py                   # Konfigurasi hyperparameter & path terstandarisasi
│   ├── data/fashion_mnist.py       # Loader & stratified 80/10/10 split Fashion-MNIST
│   ├── utils/seed.py               # Penetapan seed deterministik & seleksi perangkat GPU/CPU
│   ├── models/
│   │   ├── mlp_classifier.py       # MLP Classifier (784->256->64->10) & Frozen Latent Decoder
│   │   ├── autoencoders.py         # Vanilla AE, Sparse AE (L1), dan VAE
│   │   ├── gans.py                 # Vanilla MLP GAN dan DCGAN
│   │   └── ddpm.py                 # Time-conditioned 28x28 UNet, EMA, & GaussianDiffusion
│   ├── experiments/
│   │   ├── mlp_proof.py            # Eksperimen pembuktian 5 tahap Bagian 2.1
│   │   ├── train_autoencoders.py   # Pelatihan Vanilla AE, Sparse AE, dan VAE
│   │   ├── train_gans.py           # Pelatihan minimax Vanilla GAN dan DCGAN
│   │   └── train_ddpm.py           # Pelatihan noise-prediction DDPM + EMA
│   └── evaluation/
│       ├── metrics.py              # Test MSE, PSNR, SSIM, InceptionV3 FID, dan latensi batch 64
│       └── visualize.py            # Generator 5 figur komparatif wajib (300 DPI)
├── notebooks/
│   └── comparative_study.ipynb     # Notebook interaktif untuk inspeksi hasil & visualisasi
├── report/
│   └── build_report.py             # Skrip pembuat laporan ilmiah PDF (10-12 halaman)
└── tests/
    └── test_models.py              # Unit test dimensi tensor, loss, dan invarian model
```

## Cara Menjalankan

1. **Instalasi Dependensi**
   ```bash
   pip install -r requirements.txt
   ```

2. **Menjalankan Seluruh Eksperimen, Evaluasi & Pembuatan Figur**
   ```bash
   python main.py
   ```
   Skrip ini akan:
   - Mengunduh Fashion-MNIST (jika belum tersedia) dan membaginya menjadi 56.000 Train / 7.000 Val / 7.000 Test.
   - Menjalankan pembuktian 5 tahap batasan MLP Classifier (`Bagian 2.1`).
   - Melatih keenam model generatif (Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, DDPM).
   - Menghitung Test MSE, Test PSNR, Test SSIM, InceptionV3 FID (7.000 sampel), dan latensi inferensi per batch 64 ke `results/metrics.json`.
   - Menghasilkan 5 figur resolusi tinggi (300 DPI) di folder `figures/`.

3. **Menjalankan Unit Test**
   ```bash
   python -m pytest tests/test_models.py -v
   ```

4. **Mengisi Identitas & Membangun Laporan PDF**
   - Sunting `report_config.json` (`nama_lengkap`, `nim`, `kelas`, `github_url`).
   - Jalankan:
     ```bash
     python report/build_report.py
     ```
   - Berkas PDF akan dihasilkan sesuai konvensi `TUGAS_DL_SESI07_[NIM]_[NAMA_LENGKAP].pdf`.

## Ringkasan Eksperimen & Model

| Model | Dimensi Laten | Objektif Pelatihan | Evaluasi Utama |
| :--- | :---: | :--- | :--- |
| **MLP Classifier (Pendamping)** | 64 (`z_MLP`) | Cross-Entropy Loss | Test Accuracy, `W^T` Projection, Frozen Decoder |
| **Vanilla Autoencoder** | 64 | Pixel-wise MSE | Test MSE, PSNR, SSIM, Interpolasi |
| **Sparse Autoencoder** | 64 (L1) | MSE + $\lambda \|z\|_1$ | Test MSE, PSNR, SSIM, Sparsity, Interpolasi |
| **Variational Autoencoder (VAE)** | 64 ($\mu, \sigma$) | ELBO (BCE + KL Divergence) | Test MSE, PSNR, SSIM, FID, Interpolasi |
| **Vanilla GAN (MLP)** | 64 | Minimax BCEWithLogits | FID, Latensi, Kurva G/D, Interpolasi |
| **DCGAN** | 100 | Minimax BCEWithLogits (Conv + BN) | FID, Latensi, Kurva G/D, Interpolasi |
| **DDPM** | 28×28 ($T=1000$) | Noise Prediction MSE ($\epsilon_\theta$) | FID, Latensi, Trajektori Denoising |
