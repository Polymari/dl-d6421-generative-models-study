"""Reproducible academic PDF report builder (10-12 pages, Indonesian, Bagian I-VI)."""

import base64
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pypdf import PdfReader
from report.diagrams import render_all_architecture_svgs


def _img_to_data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _fmt_opt(val: float | None, decimals: int = 4, na_text: str = "N/A") -> str:
    if val is None:
        return na_text
    return f"{val:.{decimals}f}"


def _find_browser_binary() -> str:
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError("Neither Microsoft Edge nor Google Chrome was found for HTML-to-PDF rendering.")


def build_html(cfg: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """Render full Indonesian scientific report HTML with embedded 300-DPI figures and SVG diagrams."""
    m = metrics["models"]
    mlp = metrics["mlp_proof"]
    ds = metrics["dataset"]
    hist = metrics["histories"]

    fig1_uri = _img_to_data_uri(ROOT / "figures" / "fig1_reconstruction_grid.png")
    fig2_uri = _img_to_data_uri(ROOT / "figures" / "fig2_generative_sampling_4x4.png")
    fig3_uri = _img_to_data_uri(ROOT / "figures" / "fig3_mlp_limitation_proof.png")
    fig4_uri = _img_to_data_uri(ROOT / "figures" / "fig4_latent_interpolation.png")
    fig5_uri = _img_to_data_uri(ROOT / "figures" / "fig5_loss_curves_and_ddpm_trajectory.png")

    svgs_html = render_all_architecture_svgs()

    # Exact observed training curve endpoints
    van_tr_end = hist["vanilla_ae"]["train_losses"][-1]
    spa_tr_end = hist["sparse_ae"]["train_losses"][-1]
    vae_tr_end = hist["vae"]["train_losses"][-1]
    vgan_d_end = hist["vanilla_gan"]["d_losses"][-1]
    vgan_g_end = hist["vanilla_gan"]["g_losses"][-1]
    dc_d_end = hist["dcgan"]["d_losses"][-1]
    dc_g_end = hist["dcgan"]["g_losses"][-1]
    ddpm_start = hist["ddpm"]["train_losses"][0]
    ddpm_end = hist["ddpm"]["train_losses"][-1]
    vgan_g_max = max(hist["vanilla_gan"]["g_losses"])
    vgan_g_max_ep = hist["vanilla_gan"]["g_losses"].index(vgan_g_max) + 1
    mse_ratio = mlp["frozen_decoder_metrics"]["mse"] / m["vanilla_ae"]["mse"]
    ddpm_vs_dcgan_lat = m["ddpm"]["latency_ms_mean"] / m["dcgan"]["latency_ms_mean"]
    ddpm_vs_vae_lat = m["ddpm"]["latency_ms_mean"] / m["vae"]["latency_ms_mean"]
    sae_l1_ratio = m["vanilla_ae"]["latent_l1_mean"] / m["sparse_ae"]["latent_l1_mean"]
    sae_mse_diff = m["sparse_ae"]["mse"] - m["vanilla_ae"]["mse"]
    sae_psnr_diff = m["sparse_ae"]["psnr"] - m["vanilla_ae"]["psnr"]

    github_url = cfg["github_url"]

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8" />
<title>Laporan Tugas Take-Home Deep Learning Sesi 07</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 15mm 15mm 16mm 15mm;
    @top-right {{
      content: "Deep Learning (D6421) — Sesi 07: Generative Models";
      font-size: 7.5pt;
      color: #64748b;
      font-family: "Segoe UI", Arial, sans-serif;
    }}
    @bottom-left {{
      content: "{cfg['nama_lengkap']} | {cfg['nim']}";
      font-size: 7.5pt;
      color: #64748b;
      font-family: "Segoe UI", Arial, sans-serif;
    }}
    @bottom-right {{
      content: "Halaman " counter(page) " dari " counter(pages);
      font-size: 7.5pt;
      color: #475569;
      font-weight: 600;
      font-family: "Segoe UI", Arial, sans-serif;
    }}
  }}
  @page :first {{
    @top-right {{ content: ""; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Georgia", "Times New Roman", serif;
    font-size: 9.6pt;
    line-height: 1.44;
    color: #1e293b;
    margin: 0;
    padding: 0;
  }}
  h1, h2, h3, h4, .sans {{
    font-family: "Segoe UI", "Calibri", Arial, sans-serif;
    color: #0f172a;
  }}
  .title-banner {{
    border-top: 4px solid #1e3a8a;
    border-bottom: 2px solid #cbd5e1;
    padding: 12px 0 10px 0;
    margin-bottom: 12px;
  }}
  .course-tag {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8.5pt;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    color: #1d4ed8;
    margin-bottom: 4px;
  }}
  h1.paper-title {{
    font-size: 14.5pt;
    line-height: 1.25;
    margin: 0 0 8px 0;
    color: #0f172a;
  }}
  .meta-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8.8pt;
    margin-bottom: 12px;
  }}
  .meta-table td {{
    padding: 4px 8px;
    border: 1px solid #cbd5e1;
  }}
  .meta-table td.lbl {{
    width: 24%;
    background: #f1f5f9;
    font-weight: 700;
    color: #334155;
  }}
  h2.section-hdr {{
    font-size: 11.5pt;
    color: #1e3a8a;
    border-bottom: 1.6px solid #1e3a8a;
    padding-bottom: 2px;
    margin: 14px 0 7px 0;
    page-break-after: avoid;
  }}
  h3.sub-hdr {{
    font-size: 10pt;
    color: #0f172a;
    margin: 9px 0 4px 0;
    page-break-after: avoid;
  }}
  p {{
    margin: 0 0 6.5px 0;
    text-align: justify;
  }}
  ul, ol {{
    margin: 2px 0 7px 0;
    padding-left: 18px;
  }}
  li {{
    margin-bottom: 3px;
    text-align: justify;
  }}
  .callout {{
    background: #f8fafc;
    border-left: 3.5px solid #2563eb;
    padding: 7px 10px;
    margin: 8px 0;
    font-size: 9pt;
    break-inside: avoid;
  }}
  .callout-title {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-weight: 700;
    color: #1e40af;
    margin-bottom: 3px;
  }}
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8pt;
    margin: 7px 0 10px 0;
    break-inside: avoid;
  }}
  .data-table th {{
    background: #1e3a8a;
    color: #ffffff;
    font-weight: 700;
    padding: 5px 6px;
    border: 1px solid #1e3a8a;
    text-align: left;
  }}
  .data-table td {{
    padding: 4.5px 6px;
    border: 1px solid #cbd5e1;
    vertical-align: top;
  }}
  .data-table tr:nth-child(even) td {{
    background: #f8fafc;
  }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .eq-box {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 5px 9px;
    margin: 4px 0 6px 0;
    font-family: "Cambria Math", "Georgia", serif;
    font-size: 9.2pt;
    break-inside: avoid;
  }}
  .diagram-grid {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 6px;
    margin: 6px 0 9px 0;
  }}
  .diagram-card {{
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 5px 8px;
    background: #ffffff;
    break-inside: avoid;
  }}
  .diagram-title {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8.2pt;
    font-weight: 700;
    color: #1e3a8a;
    margin-bottom: 2px;
  }}
  .arch-svg {{
    width: 100%;
    height: auto;
    display: block;
  }}
  .figure-block {{
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 6px 8px;
    margin: 8px 0 10px 0;
    background: #ffffff;
    break-inside: avoid;
  }}
  .figure-block img {{
    width: 100%;
    height: auto;
    display: block;
    margin: 0 auto 5px auto;
  }}
  .fig-caption {{
    font-size: 8.5pt;
    color: #334155;
    line-height: 1.38;
    text-align: justify;
  }}
  .qa-card {{
    border-left: 3px solid #0f766e;
    background: #f8fafc;
    padding: 6px 9px;
    margin-bottom: 8px;
    break-inside: avoid;
  }}
  .qa-title {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-weight: 700;
    font-size: 9.3pt;
    color: #0f766e;
    margin-bottom: 3px;
  }}
  .page-break {{ page-break-before: always; }}
  code {{
    font-family: "Consolas", monospace;
    font-size: 8.4pt;
    background: #f1f5f9;
    padding: 0 3px;
    border-radius: 2px;
  }}
</style>
</head>
<body>

<!-- ==================== BAGIAN I ==================== -->
<div class="title-banner">
  <div class="course-tag">{cfg['mata_kuliah']}</div>
  <h1 class="paper-title">Studi Komparatif Empiris Model Generatif: Autoencoder (Vanilla &amp; Sparse), VAE, GAN (Vanilla &amp; DCGAN), dan DDPM disertai Pembuktian Batasan Teoretis Model Klasifikasi untuk Image Generation</h1>
</div>

<h2 class="section-hdr">Bagian I: Identitas Penulis, Tautan Repositori GitHub, dan Protokol Standardisasi</h2>

<table class="meta-table">
  <tr>
    <td class="lbl">Nama Lengkap</td><td><strong>{cfg['nama_lengkap']}</strong></td>
    <td class="lbl">NIM</td><td><strong>{cfg['nim']}</strong></td>
  </tr>
  <tr>
    <td class="lbl">Kelas</td><td>{cfg['kelas']}</td>
    <td class="lbl">Tahun Akademik</td><td>{cfg['tahun_akademik']}</td>
  </tr>
  <tr>
    <td class="lbl">Dosen Pengampu</td><td colspan="3">{cfg['dosen_pengampu']}</td>
  </tr>
  <tr>
    <td class="lbl">Repositori GitHub</td><td colspan="3"><a href="{github_url}">{github_url}</a> (Skrip PyTorch Modular, <code>requirements.txt</code>, <code>README.md</code>, Notebook, &amp; Riwayat Commit Bertahap)</td>
  </tr>
</table>

<p>Laporan ilmiah ini memaparkan investigasi teoretis dan empiris atas enam paradigma pemodelan generatif dari materi Sesi 06 dan Sesi 07—yakni <strong>Vanilla Autoencoder</strong>, <strong>Sparse Autoencoder</strong>, <strong>Variational Autoencoder (VAE)</strong>, <strong>Vanilla GAN</strong> (berbasis MLP), <strong>Deep Convolutional GAN (DCGAN)</strong>, serta <strong>Denoising Diffusion Probabilistic Models (DDPM)</strong>—bersama satu model pendamping <strong>MLP Classifier</strong> untuk membuktikan secara matematis dan eksperimental mengapa representasi fitur diskriminatif tidak dapat langsung difungsikan sebagai generator citra.</p>

<div class="callout">
  <div class="callout-title">Prinsip Evaluasi Terstandarisasi</div>
  Seluruh model dilatih dan dievaluasi secara seragam pada benchmark <strong>Fashion-MNIST</strong> ({ds['name']}, 70.000 citra 28×28 grayscale, 10 kelas pakaian) menggunakan penetapan random seed deterministik <code>seed={ds['seed']}</code> (Python, NumPy, PyTorch CUDA), pembagian data terstratifikasi <strong>80% Train ({ds['train_size']:,} citra) / 10% Validation ({ds['val_size']:,} citra) / 10% Test ({ds['test_size']:,} citra)</strong>, serta <code>batch_size={ds['batch_size']}</code> (<code>shuffle=True</code> pada train, <code>shuffle=False</code> pada validation/test). Seluruh metrik numerik dan visualisasi diukur pada Test Set yang belum pernah dilihat selama pelatihan di GPU NVIDIA GeForce RTX 4060 Laptop (8 GB VRAM).
</div>

<!-- ==================== BAGIAN II ==================== -->
<h2 class="section-hdr">Bagian II: Landasan Teori, Desain Arsitektur, dan Formulasi Matematika</h2>

<p>Perbedaan perilaku empiris antarmodel generatif berakar langsung pada hubungan kausal antara <em>inductive bias</em> arsitektur (lapisan <em>fully-connected</em> versus konvolusional berhierarki spasial) dan bentuk fungsi objektif yang dioptimasi. Diagram skematik arsitektur keenam model generatif beserta model MLP Classifier pendamping disajikan di bawah ini.</p>

{svgs_html}

<h3 class="sub-hdr">2.1 Konfigurasi Parameter &amp; Hiperparameter Terstandarisasi</h3>
<table class="data-table">
  <thead>
    <tr>
      <th>Model</th>
      <th>Lapisan Arsitektur Utama</th>
      <th>Dim Laten</th>
      <th>Aktivasi</th>
      <th>Optimizer &amp; LR</th>
      <th>Epoch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>MLP Classifier</strong></td>
      <td>784 → 256 → 64 → 10</td>
      <td class="num">64 (<em>z</em><sub>MLP</sub>)</td>
      <td>ReLU, Logits</td>
      <td>Adam (1e-3, Cosine)</td>
      <td class="num">20</td>
    </tr>
    <tr>
      <td><strong>Frozen Decoder</strong></td>
      <td>64 → 256 → 784 (pada <em>z</em><sub>MLP</sub> beku)</td>
      <td class="num">64</td>
      <td>ReLU, Sigmoid</td>
      <td>Adam (1e-3)</td>
      <td class="num">30</td>
    </tr>
    <tr>
      <td><strong>Vanilla AE</strong></td>
      <td>Enc: 784→256→64 | Dec: 64→256→784</td>
      <td class="num">64 (Linear)</td>
      <td>ReLU, Sigmoid</td>
      <td>Adam (1e-3)</td>
      <td class="num">30</td>
    </tr>
    <tr>
      <td><strong>Sparse AE</strong></td>
      <td>Enc: 784→256→64 | Dec: 64→256→784</td>
      <td class="num">64 (ReLU+L1)</td>
      <td>ReLU, Sigmoid</td>
      <td>Adam (1e-3, λ=5e-3)</td>
      <td class="num">30</td>
    </tr>
    <tr>
      <td><strong>VAE</strong></td>
      <td>784→256→(μ:64, logσ²:64)→256→784</td>
      <td class="num">64 (Gaussian)</td>
      <td>ReLU, Sigmoid</td>
      <td>Adam (1e-3)</td>
      <td class="num">30</td>
    </tr>
    <tr>
      <td><strong>Vanilla GAN</strong></td>
      <td>G: 64→256→512→784 | D: 784→512→256→1</td>
      <td class="num">64 (Prior)</td>
      <td>LeakyReLU(0.2), Tanh</td>
      <td>Adam (2e-4, β=(0.5,0.999))</td>
      <td class="num">50</td>
    </tr>
    <tr>
      <td><strong>DCGAN</strong></td>
      <td>G: ConvT(100→256→128→1) | D: Conv(1→64→128→256)</td>
      <td class="num">100 (Prior)</td>
      <td>ReLU/LReLU, BN, Tanh</td>
      <td>Adam (2e-4, β=(0.5,0.999))</td>
      <td class="num">50</td>
    </tr>
    <tr>
      <td><strong>DDPM</strong></td>
      <td>UNet28 (64→128→256+SelfAttn 7×7), TimeEmb, EMA</td>
      <td class="num">28×28 (T=1000)</td>
      <td>SiLU, GroupNorm</td>
      <td>AdamW (2e-4, wd=1e-4)</td>
      <td class="num">50</td>
    </tr>
  </tbody>
</table>

<h3 class="sub-hdr">2.2 Formulasi Matematika Fungsi Loss Objektif</h3>
<div class="eq-box">
  <strong>1. MLP Classifier (Cross-Entropy):</strong> &nbsp; ℒ<sub>CE</sub> = −(1/N) ∑<sub>i=1..N</sub> ∑<sub>c=1..10</sub> y<sub>i,c</sub> log softmax(W<sub>3</sub> z<sub>MLP,i</sub> + b<sub>3</sub>)<sub>c</sub> &nbsp;— memodelkan <em>P</em>(<em>y</em>|<em>x</em>).<br/>
  <strong>2. Vanilla Autoencoder (Pixel-wise MSE):</strong> &nbsp; ℒ<sub>AE</sub> = (1/784) ‖<em>x</em> − <em>D</em><sub>θ</sub>(<em>E</em><sub>φ</sub>(<em>x</em>))‖<sub>2</sub><sup>2</sup>.<br/>
  <strong>3. Sparse Autoencoder (MSE + L1 Sparsity):</strong> &nbsp; ℒ<sub>SAE</sub> = (1/784) ‖<em>x</em> − <em>x̂</em>‖<sub>2</sub><sup>2</sup> + λ · (1/64) ∑<sub>j=1..64</sub> |<em>z</em><sub>j</sub>|, dengan λ = 5×10<sup>−3</sup>.<br/>
  <strong>4. Variational Autoencoder (ELBO):</strong> &nbsp; ℒ<sub>VAE</sub> = −𝔼<sub><em>q</em><sub>φ</sub>(<em>z</em>|<em>x</em>)</sub>[log <em>p</em><sub>θ</sub>(<em>x</em>|<em>z</em>)] + <em>D</em><sub>KL</sub>(<em>q</em><sub>φ</sub>(<em>z</em>|<em>x</em>) ∥ 𝒩(0, <strong>I</strong>)), dimana <em>D</em><sub>KL</sub> = −½ ∑(1 + log σ² − μ² − σ²) dan <em>z</em> = μ + σ ⊙ ε.<br/>
  <strong>5. Vanilla GAN &amp; DCGAN (Minimax Game):</strong> &nbsp; min<sub>G</sub> max<sub>D</sub> <em>V</em>(<em>D</em>,<em>G</em>) = 𝔼<sub><em>x</em>~<em>p</em><sub>data</sub></sub>[log <em>D</em>(<em>x</em>)] + 𝔼<sub><em>z</em>~𝒩(0,<strong>I</strong>)</sub>[log(1 − <em>D</em>(<em>G</em>(<em>z</em>)))], dilatih dengan objektif <em>non-saturating</em> Generator −𝔼[log <em>D</em>(<em>G</em>(<em>z</em>))].<br/>
  <strong>6. DDPM (Variational Noise-Prediction MSE):</strong> &nbsp; ℒ<sub>simple</sub>(θ) = 𝔼<sub><em>t</em>~𝒰(1,<em>T</em>), <em>x</em><sub>0</sub>, ε~𝒩(0,<strong>I</strong>)</sub>[‖ε − ε<sub>θ</sub>(√ᾱ<sub><em>t</em></sub> <em>x</em><sub>0</sub> + √(1 − ᾱ<sub><em>t</em></sub>) ε, <em>t</em>)‖<sub>2</sub><sup>2</sup>], dengan jadwal varians linear β<sub><em>t</em></sub> ∈ [10<sup>−4</sup>, 0.02] untuk <em>T</em> = 1000 langkah.
</div>

<!-- ==================== BAGIAN III ==================== -->
<h2 class="section-hdr">Bagian III: Pembuktian Teoretis dan Empiris Batasan MLP Classifier untuk Image Generation</h2>

<p>Meskipun lapisan ekstraktor fitur MLP Classifier (<code>784 → 256 → 64</code>) memiliki topologi identik dengan bagian Encoder pada Autoencoder, bobot terlatih <em>W</em><sup>(<em>l</em>)</sup> dan fitur laten <em>z</em><sub>MLP</sub> ∈ ℝ<sup>64</sup> gagal melakukan rekonstruksi dan sintesis citra karena tiga batasan matematis fundamental:</p>

<ol>
  <li><strong>Information Bottleneck Pemodelan <em>P</em>(<em>y</em>|<em>x</em>) vs. <em>P</em>(<em>x</em>):</strong> Cross-Entropy Loss memaksimalkan separabilitas antar-10 kelas (<em>inter-class margin</em>). Akibatnya, jaringan menekan dan membuang variasi spasial intra-kelas (ketebalan garis, lipatan kain, orientasi kerah, corak tekstur) yang tidak relevan terhadap keputusan kelas, berbeda dengan model generatif yang memodelkan kerapatan data penuh <em>P</em>(<em>x</em>).</li>
  <li><strong>Non-Ortogonalitas Matriks Bobot &amp; Ireversibilitas ReLU:</strong> Matriks bobot persegi panjang <em>W</em><sup>(<em>l</em>)</sup> tidak memenuhi (<em>W</em><sup>(<em>l</em>)</sup>)<sup>T</sup> <em>W</em><sup>(<em>l</em>)</sup> = <strong>I</strong> sehingga <em>W</em><sup>T</sup> ≠ <em>W</em><sup>−1</sup>. Pengukuran langsung pada bobot terlatih menghasilkan deviasi ortogonalitas relatif ‖<em>W</em><sup>T</sup><em>W</em> − <strong>I</strong>‖<sub>F</sub> / ‖<strong>I</strong>‖<sub>F</sub> sebesar <strong>{mlp['orthogonality']['W1_orthogonality_error']:.4f}</strong> (<em>W</em><sub>1</sub>), <strong>{mlp['orthogonality']['W2_orthogonality_error']:.4f}</strong> (<em>W</em><sub>2</sub>), dan <strong>{mlp['orthogonality']['W3_orthogonality_error']:.4f}</strong> (<em>W</em><sub>3</sub>). Selain itu, aktivasi ReLU(<em>u</em>) = max(0, <em>u</em>) memotong komponen negatif secara ireversibel (tercatat <strong>{mlp['z_mlp_zero_fraction']*100:.2f}%</strong> elemen <em>z</em><sub>MLP</sub> bernilai tepat nol).</li>
  <li><strong>Ketiadaan Regularisasi Prior <em>P</em>(<em>z</em>):</strong> Ruang laten <em>z</em><sub>MLP</sub> tidak dikekang ke distribusi prior kontinu seperti 𝒩(0, <strong>I</strong>). Pada Test Set, vektor rata-rata fitur memiliki norma ‖μ<sub><em>z</em></sub>‖<sub>2</sub> = <strong>{mlp['z_mlp_mu_norm']:.2f}</strong> dengan rata-rata simpangan baku σ̄<sub><em>z</em></sub> = <strong>{mlp['z_mlp_sigma_mean']:.2f}</strong> serta sebaran asimetris akibat ReLU.</li>
</ol>

<table class="data-table">
  <thead>
    <tr>
      <th>Tahap Eksperimen Bagian 2.1 (Test Set Fashion-MNIST)</th>
      <th class="num">Test MSE</th>
      <th class="num">Test PSNR (dB)</th>
      <th class="num">Test SSIM</th>
      <th>Observasi Empiris Utama</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Tahap 1:</strong> MLP Classifier (784→256→64→10)</td>
      <td class="num">—</td><td class="num">—</td><td class="num">—</td>
      <td>Test Acc = <strong>{mlp['test_accuracy']*100:.2f}%</strong> (Val = {mlp['val_accuracy']*100:.2f}%, melampaui syarat ≥ 88%)</td>
    </tr>
    <tr>
      <td><strong>Tahap 2:</strong> Proyeksi Balik Bobot <em>W</em><sup>T</sup> dari One-Hot <em>y</em><sub><em>c</em></sub></td>
      <td class="num">{mlp['transpose_onehot_metrics']['mse']:.5f}</td>
      <td class="num">{mlp['transpose_onehot_metrics']['psnr']:.2f}</td>
      <td class="num">{mlp['transpose_onehot_metrics']['ssim']:.4f}</td>
      <td>Pola bercak abu-abu seragam (superposisi template kelas kasar; <em>W</em><sup>T</sup> ≠ <em>W</em><sup>−1</sup>)</td>
    </tr>
    <tr>
      <td><strong>Tahap 3:</strong> Proyeksi Balik <em>W</em><sup>T</sup> dari Latent <em>z</em><sub>MLP</sub></td>
      <td class="num">{mlp['transpose_latent_metrics']['mse']:.5f}</td>
      <td class="num">{mlp['transpose_latent_metrics']['psnr']:.2f}</td>
      <td class="num">{mlp['transpose_latent_metrics']['ssim']:.4f}</td>
      <td>Artefak kontras biner ekstrem; bobot maju gagal berfungsi sebagai dekoder</td>
    </tr>
    <tr>
      <td><strong>Tahap 4a:</strong> Frozen Latent Decoder (<em>z</em><sub>MLP</sub> ∈ ℝ<sup>64</sup>)</td>
      <td class="num">{mlp['frozen_decoder_metrics']['mse']:.5f}</td>
      <td class="num">{mlp['frozen_decoder_metrics']['psnr']:.2f}</td>
      <td class="num">{mlp['frozen_decoder_metrics']['ssim']:.4f}</td>
      <td>Siluet kelas terpulihkan, tetapi tekstur &amp; detail spasial intra-kelas hilang ({mse_ratio:.2f}× MSE Vanilla AE)</td>
    </tr>
    <tr>
      <td><strong>Tahap 4b (Pembanding):</strong> Vanilla AE (<em>z</em><sub>AE</sub> ∈ ℝ<sup>64</sup>)</td>
      <td class="num">{m['vanilla_ae']['mse']:.5f}</td>
      <td class="num">{m['vanilla_ae']['psnr']:.2f}</td>
      <td class="num">{m['vanilla_ae']['ssim']:.4f}</td>
      <td>Mempertahankan variasi intra-kelas karena tanpa penalti bottleneck diskriminatif</td>
    </tr>
    <tr>
      <td><strong>Tahap 5:</strong> Prior Sampling <em>z</em><sub>rand</sub> ~ 𝒩(μ<sub><em>z</em></sub>, diag(σ<sub><em>z</em></sub><sup>2</sup>))</td>
      <td class="num">—</td><td class="num">—</td><td class="num">—</td>
      <td>Gagal membentuk objek utuh; menghasilkan pecahan siluet &amp; bercak inkoheren</td>
    </tr>
  </tbody>
</table>

<!-- ==================== BAGIAN IV ==================== -->
<h2 class="section-hdr">Bagian IV: Tabel Komparasi Metrik Numerik (Kompilasi Kuantitatif)</h2>

<p>Seluruh metrik rekonstruksi (Test MSE, Test PSNR, Test SSIM) dievaluasi pada rentang piksel [0, 1] terhadap 7.000 citra Test Set. Skor <strong>FID (Fréchet Inception Distance)</strong> dihitung menggunakan fitur pool3 (2048-dimensi) dari InceptionV3 terlatih terhadap 7.000 citra sintetis versus 7.000 citra Test Set riil. Latensi inferensi diukur per batch 64 citra dengan <code>torch.cuda.synchronize()</code>.</p>

<table class="data-table">
  <thead>
    <tr>
      <th>Arsitektur Model</th>
      <th>Dim Laten</th>
      <th class="num">Epoch Time (s)</th>
      <th class="num">Latency (ms/b64)</th>
      <th class="num">Test MSE</th>
      <th class="num">Test PSNR (dB)</th>
      <th class="num">Test SSIM</th>
      <th class="num">FID Score</th>
      <th>Kestabilan Training</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Vanilla Autoencoder</strong></td>
      <td>64</td>
      <td class="num">{m['vanilla_ae']['epoch_time_s']:.2f}</td>
      <td class="num">{m['vanilla_ae']['latency_ms_mean']:.2f}</td>
      <td class="num">{m['vanilla_ae']['mse']:.5f}</td>
      <td class="num">{m['vanilla_ae']['psnr']:.2f}</td>
      <td class="num">{m['vanilla_ae']['ssim']:.4f}</td>
      <td class="num">N/A (Latent Voids)</td>
      <td>Stabil / Konvergen (Train MSE {van_tr_end:.5f})</td>
    </tr>
    <tr>
      <td><strong>Sparse Autoencoder</strong></td>
      <td>64 (L1)</td>
      <td class="num">{m['sparse_ae']['epoch_time_s']:.2f}</td>
      <td class="num">{m['sparse_ae']['latency_ms_mean']:.2f}</td>
      <td class="num">{m['sparse_ae']['mse']:.5f}</td>
      <td class="num">{m['sparse_ae']['psnr']:.2f}</td>
      <td class="num">{m['sparse_ae']['ssim']:.4f}</td>
      <td class="num">N/A (Latent Voids)</td>
      <td>Stabil / Konvergen ({m['sparse_ae']['latent_zero_frac']*100:.2f}% nol, L1={m['sparse_ae']['latent_l1_mean']:.2f})</td>
    </tr>
    <tr>
      <td><strong>Variational AE (VAE)</strong></td>
      <td>64</td>
      <td class="num">{m['vae']['epoch_time_s']:.2f}</td>
      <td class="num">{m['vae']['latency_ms_mean']:.2f}</td>
      <td class="num">{m['vae']['mse']:.5f}</td>
      <td class="num">{m['vae']['psnr']:.2f}</td>
      <td class="num">{m['vae']['ssim']:.4f}</td>
      <td class="num">{m['vae']['fid']:.2f}</td>
      <td>Stabil / Konvergen (ELBO {vae_tr_end:.2f})</td>
    </tr>
    <tr>
      <td><strong>Vanilla GAN (MLP)</strong></td>
      <td>64 (Prior)</td>
      <td class="num">{m['vanilla_gan']['epoch_time_s']:.2f}</td>
      <td class="num">{m['vanilla_gan']['latency_ms_mean']:.2f}</td>
      <td class="num">N/A</td>
      <td class="num">N/A</td>
      <td class="num">N/A</td>
      <td class="num">{m['vanilla_gan']['fid']:.2f}</td>
      <td>Rentan Osilasi Awal (D={vgan_d_end:.2f}, G={vgan_g_end:.2f})</td>
    </tr>
    <tr>
      <td><strong>DCGAN</strong></td>
      <td>100 (Prior)</td>
      <td class="num">{m['dcgan']['epoch_time_s']:.2f}</td>
      <td class="num">{m['dcgan']['latency_ms_mean']:.2f}</td>
      <td class="num">N/A</td>
      <td class="num">N/A</td>
      <td class="num">N/A</td>
      <td class="num">{m['dcgan']['fid']:.2f}</td>
      <td>Cukup Stabil (D={dc_d_end:.2f}, G={dc_g_end:.2f} dominan)</td>
    </tr>
    <tr>
      <td><strong>DDPM (UNet28 + EMA)</strong></td>
      <td>28×28</td>
      <td class="num">{m['ddpm']['epoch_time_s']:.2f}</td>
      <td class="num">{m['ddpm']['latency_ms_mean']:.2f}</td>
      <td class="num">N/A</td>
      <td class="num">N/A</td>
      <td class="num">N/A</td>
      <td class="num"><strong>{m['ddpm']['fid']:.2f}</strong></td>
      <td>Sangat Stabil (Noise MSE {ddpm_start:.4f} → {ddpm_end:.4f})</td>
    </tr>
  </tbody>
</table>

<p>Dari sisi representasi laten pada keluarga Autoencoder, penambahan penalti sparsity λ‖<em>z</em>‖<sub>1</sub> (λ = 5×10<sup>−3</sup>) berhasil menurunkan rata-rata norma L1 representasi laten dari <strong>{m['vanilla_ae']['latent_l1_mean']:.2f}</strong> (dengan hanya <strong>{m['vanilla_ae']['latent_zero_frac']*100:.2f}%</strong> unit hampir nol pada Vanilla AE) menjadi <strong>{m['sparse_ae']['latent_l1_mean']:.2f}</strong> dengan <strong>{m['sparse_ae']['latent_zero_frac']*100:.2f}%</strong> unit non-aktif (|<em>z</em>| &lt; 10<sup>−3</sup>) pada Sparse AE. Pada evaluasi generatif, terjadi lompatan kualitas dramatis antar-keluarga: <strong>DDPM</strong> mencatat FID terbaik (<strong>{m['ddpm']['fid']:.2f}</strong>), diikuti ketat oleh <strong>DCGAN</strong> (<strong>{m['dcgan']['fid']:.2f}</strong>), sementara <strong>VAE</strong> (<strong>{m['vae']['fid']:.2f}</strong>) dan <strong>Vanilla GAN</strong> (<strong>{m['vanilla_gan']['fid']:.2f}</strong>) tertinggal jauh akibat keburaman posterior Gaussian dan ketiadaan bias konvolusional.</p>

<!-- ==================== BAGIAN V ==================== -->
<h2 class="section-hdr page-break">Bagian V: Kompilasi Lima Visualisasi Komparatif Utama</h2>

<div class="figure-block">
  <img src="{fig1_uri}" alt="Figur 1 Rekonstruksi Side-by-Side" />
  <div class="fig-caption"><strong>Figur 1 — Grid Rekonstruksi Side-by-Side (Test Set):</strong> Perbandingan 8 kelas Fashion-MNIST antara citra asli dan rekonstruksi Vanilla AE (MSE {m['vanilla_ae']['mse']:.5f}), Sparse AE ({m['sparse_ae']['mse']:.5f}), serta VAE ({m['vae']['mse']:.5f}). Vanilla AE mempertahankan detail lokal paling tajam (pola kotak pada <em>Shirt</em> dan tekstur <em>Coat</em>), Sparse AE menghasilkan tepi sedikit lebih halus akibat pemangkasan {m['sparse_ae']['latent_zero_frac']*100:.2f}% aktivasi laten, sedangkan VAE tampak paling kabur karena regularisasi KL memaksa variasi lokal melebur menuju rata-rata posterior.</div>
</div>

<div class="figure-block">
  <img src="{fig2_uri}" alt="Figur 2 Random Generative Sampling 4x4" />
  <div class="fig-caption"><strong>Figur 2 — Grid Random Generative Sampling (Kisi 4×4 per Model):</strong> Sampel acak dari prior 𝒩(0, <strong>I</strong>). <strong>Panel A (VAE, FID {m['vae']['fid']:.2f}):</strong> struktur global utuh namun tepi buram khas ekspektasi kondisional. <strong>Panel B (Vanilla GAN, FID {m['vanilla_gan']['fid']:.2f}):</strong> tepi kontras tinggi tetapi sarat derau piksel berbintik (<em>checkerboard/speckle noise</em>) karena generator MLP tidak memiliki lokalisasi spasial. <strong>Panel C (DCGAN, FID {m['dcgan']['fid']:.2f}):</strong> siluet pakaian dan sepatu jauh lebih tajam dan koheren berkat <em>strided transposed convolution</em> dan BatchNorm. <strong>Panel D (DDPM, FID {m['ddpm']['fid']:.2f}):</strong> sampel paling bersih, kontur tegas, gradasi tekstil halus, dan keragaman kelas paling merata.</div>
</div>

<div class="figure-block">
  <img src="{fig3_uri}" alt="Figur 3 Bukti Empiris MLP Classifier" />
  <div class="fig-caption"><strong>Figur 3 — Bukti Empiris Batasan Bobot dan Latent Features MLP Classifier:</strong> <strong>(1)</strong> Citra uji asli. <strong>(2)</strong> Proyeksi balik langsung <em>W</em><sup>T</sup> dari <em>y</em><sub><em>c</em></sub> menghasilkan bercak abu-abu nyaris identik lintas kelas, sedangkan dari <em>z</em><sub>MLP</sub> menghasilkan pola biner kacau. <strong>(3)</strong> Frozen Latent Decoder hanya memulihkan siluet kasar kelas (PSNR {mlp['frozen_decoder_metrics']['psnr']:.2f} dB vs Vanilla AE {m['vanilla_ae']['psnr']:.2f} dB; misalnya detail corak <em>Shirt</em> paling kanan hilang total menjadi kemeja polos). <strong>(4)</strong> Sampling prior <em>z</em><sub>rand</sub> dari statistik <em>z</em><sub>MLP</sub> menghasilkan artefak rusak tidak berbentuk objek utuh.</div>
</div>

<div class="figure-block">
  <img src="{fig4_uri}" alt="Figur 4 Interpolasi Ruang Laten Linier" />
  <div class="fig-caption"><strong>Figur 4 — Interpolasi Ruang Laten Linier <em>z</em>(α) = (1−α)<em>z</em><sub>A</sub> + α<em>z</em><sub>B</sub> (α ∈ [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):</strong> Pada Vanilla AE dan Sparse AE, transisi antara <em>Sneaker</em> (α=0) dan <em>Dress</em> (α=1) melewati wilayah kosong ruang laten (<em>latent space voids</em>) pada α=0.4–0.6 sehingga muncul efek tumpang-tindih hantu (<em>ghosting</em>). Sebaliknya, VAE menunjukkan transisi probabilistik yang halus tanpa artefak pecah, Vanilla GAN mengalami transisi berderau, sedangkan DCGAN bertransformasi secara semantik tajam dari jaket → gaun → celana panjang.</div>
</div>

<div class="figure-block">
  <img src="{fig5_uri}" alt="Figur 5 Kurva Training Loss dan DDPM Denoising Trajectory" />
  <div class="fig-caption"><strong>Figur 5 — Kurva Training Loss Keenam Model dan Trajektori Reverse Denoising DDPM:</strong> Vanilla AE, Sparse AE, dan VAE konvergen mulus tanpa gap overfitting antara kurva train dan validation. Pada Vanilla GAN terlihat lonjakan loss Generator di epoch awal ({vgan_g_max:.2f} pada epoch {vgan_g_max_ep}) sebelum stabil, sedangkan pada DCGAN Discriminator mendominasi ringan (D≈{dc_d_end:.2f}, G≈{dc_g_end:.2f}) namun tetap menghasilkan fitur konvolusional tajam. DDPM mencatat penurunan monoton Noise MSE ({ddpm_start:.4f} → {ddpm_end:.4f}) dengan kurva paling stabil; baris bawah memperlihatkan evolusi bertahap dari derau Gaussian <em>t</em>=999 menuju bentuk sepatu dan gaun tajam di <em>t</em>=0.</div>
</div>

<!-- ==================== BAGIAN VI ==================== -->
<h2 class="section-hdr page-break">Bagian VI: Pembahasan Kritis, Jawaban Pertanyaan Penelitian (Q1–Q6), dan Rekomendasi Industri</h2>

<div class="qa-card">
  <div class="qa-title">Q1 — Perbandingan Kuantitatif: Reconstruction Fidelity vs. Generative Quality</div>
  <p>Pada kemampuan rekonstruksi Test Set, <strong>Vanilla AE</strong> mencatat performa terbaik (MSE = <strong>{m['vanilla_ae']['mse']:.5f}</strong>, PSNR = <strong>{m['vanilla_ae']['psnr']:.2f} dB</strong>, SSIM = <strong>{m['vanilla_ae']['ssim']:.4f}</strong>), disusul <strong>Sparse AE</strong> (MSE = <strong>{m['sparse_ae']['mse']:.5f}</strong>, PSNR = <strong>{m['sparse_ae']['psnr']:.2f} dB</strong>, SSIM = <strong>{m['sparse_ae']['ssim']:.4f}</strong>) dan <strong>VAE</strong> (MSE = <strong>{m['vae']['mse']:.5f}</strong>, PSNR = <strong>{m['vae']['psnr']:.2f} dB</strong>, SSIM = <strong>{m['vae']['ssim']:.4f}</strong>). Penambahan penalti sparsity λ‖<em>z</em>‖<sub>1</sub> (λ=5×10<sup>−3</sup>) memaksa {m['sparse_ae']['latent_zero_frac']*100:.2f}% dimensi laten Sparse AE bernilai hampir nol dan memangkas norma L1 hingga {sae_l1_ratio:.2f}× ({m['sparse_ae']['latent_l1_mean']:.2f} vs {m['vanilla_ae']['latent_l1_mean']:.2f}), sehingga kapasitas kanal informasi menyempit dan mengorbankan +{sae_mse_diff:.5f} MSE ({sae_psnr_diff:.2f} dB PSNR) dibanding representasi padat Vanilla AE.</p>
  <p>Sebaliknya, pada sintesis citra baru dari prior acak 𝒩(0, <strong>I</strong>), skor <strong>FID</strong> memperlihatkan hierarki yang jelas: <strong>DDPM ({m['ddpm']['fid']:.2f}) &lt; DCGAN ({m['dcgan']['fid']:.2f}) &lt;&lt; VAE ({m['vae']['fid']:.2f}) &lt; Vanilla GAN ({m['vanilla_gan']['fid']:.2f})</strong>. Hal ini membuktikan bahwa fidelitas rekonstruksi deterministik tidak berkorelasi dengan kapabilitas generatif: Vanilla AE dan Sparse AE tidak dapat disampel dari prior karena ruang latennya tidak diregulasi, sementara VAE mengorbankan ketajaman demi kontinuitas ELBO.</p>
</div>

<div class="qa-card">
  <div class="qa-title">Q2 — Analisis Blurriness vs. Sharp Edges dan Struktur Representasi Fitur</div>
  <p>Kekaburan (<em>blurriness</em>) pada Vanilla AE, Sparse AE, dan VAE adalah konsekuensi matematis langsung dari minimisasi <em>pixel-wise loss</em> (MSE / BCE). Secara teoretis, nilai optimal <em>x̂</em> yang meminimalkan 𝔼[‖<em>x</em> − <em>x̂</em>‖<sub>2</sub><sup>2</sup> | <em>z</em>] adalah <strong>ekspektasi kondisional</strong> 𝔼[<em>x</em>|<em>z</em>]. Apabila beberapa variasi tepi atau tekstur berpeluang muncul dari kode laten <em>z</em> yang berdekatan (<em>multimodal posterior</em>), rata-rata piksel menghasilkan efek kabur yang meredam frekuensi tinggi. Pada VAE, adanya istilah KL divergence memperparah perataan ini karena <em>posterior overlap</em> meningkat.</p>
  <p>Sebaliknya, fungsi objektif minimax GAN/DCGAN meminimalkan <strong>divergensi Jensen-Shannon</strong> <em>D</em><sub>JS</sub>(<em>P</em><sub>data</sub> ∥ <em>P</em><sub><em>g</em></sub>) ketika Discriminator optimal. Karena <em>D</em><sub>JS</sub> menghukum berat generator jika menghasilkan sampel kabur yang mudah dikenali sebagai palsu oleh Discriminator (berada di luar manifold citra tajam), Generator terdorong memilih mode tajam pada distribusi data. Sementara itu, penalti L1 pada Sparse AE mengubah struktur fitur tersembunyi dari representasi padat terdistribusi (pada Vanilla AE) menjadi representasi selektif terlokalisasi di mana hanya subset neuron yang aktif untuk tiap pola pakaian.</p>
</div>

<div class="qa-card">
  <div class="qa-title">Q3 — Stabilitas Training, Dinamika Arsitektur, dan Verifikasi Generative Trilemma</div>
  <p>Pada <strong>Vanilla GAN</strong> (MLP), ketiadaan <em>weight sharing</em> spasial membuat Discriminator mudah mengeksploitasi artefak piksel tunggal atau menyebabkan gradien Generator berosilasi tajam di epoch awal (G loss melonjak ke {vgan_g_max:.2f} pada epoch {vgan_g_max_ep} sebelum turun ke {vgan_g_end:.2f}). <strong>DCGAN</strong> memitigasi ketidakstabilan ini melalui prinsip arsitektur konvolusional penuh (<em>strided convolutions</em> menggantikan pooling, penghapusan lapisan FC tersembunyi, <code>BatchNorm2d</code> untuk menstabilkan skala gradien, serta aktivasi <code>LeakyReLU</code> dan inisialisasi 𝒩(0, 0.02)) sehingga menghasilkan FID {m['dcgan']['fid']:.2f} vs {m['vanilla_gan']['fid']:.2f} pada Vanilla GAN—meskipun tetap sensitif terhadap ketidakseimbangan G/D (D={dc_d_end:.2f} vs G={dc_g_end:.2f}).</p>
  <p>Dibandingkan optimasi <em>adversarial minimax</em> yang non-stasioner, <strong>DDPM</strong> melatih fungsi objektif regresi <em>supervised noise prediction</em> ‖ε − ε<sub>θ</sub>‖<sub>2</sub><sup>2</sup> pada target tetap, menghasilkan kurva training paling stabil dan konvergen monoton ({ddpm_start:.4f} → {ddpm_end:.4f} pada epoch 1–50). Data latensi batch-64 memverifikasi langsung teori <strong>Generative Trilemma</strong> (Sample Quality, Sampling Speed, Mode Coverage):</p>
  <ul>
    <li><strong>VAE</strong> unggul di <em>Sampling Speed</em> (<strong>{m['vae']['latency_ms_mean']:.2f} ms/batch</strong>) dan <em>Mode Coverage</em>, tetapi mengorbankan <em>Sample Quality</em> (FID {m['vae']['fid']:.2f}, buram).</li>
    <li><strong>DCGAN</strong> menyeimbangkan <em>Sample Quality</em> tajam (FID {m['dcgan']['fid']:.2f}) dan <em>Sampling Speed</em> satu-langkah (<strong>{m['dcgan']['latency_ms_mean']:.2f} ms/batch</strong>), namun rentan penurunan keragaman mode / sensitivitas arsitektur.</li>
    <li><strong>DDPM</strong> memuncaki <em>Sample Quality</em> (FID <strong>{m['ddpm']['fid']:.2f}</strong>) dan <em>Mode Coverage</em>, tetapi membayar harga latensi iteratif <em>T</em>=1000 langkah sebesar <strong>{m['ddpm']['latency_ms_mean']:.2f} ms/batch</strong> (~{ddpm_vs_dcgan_lat:,.0f}× lebih lambat dari DCGAN dan ~{ddpm_vs_vae_lat:,.0f}× lebih lambat dari VAE).</li>
  </ul>
</div>

<div class="qa-card">
  <div class="qa-title">Q4 — Struktur Latent Space dan Fenomena Interpolasi</div>
  <p>Dalam eksperimen interpolasi linier <em>z</em>(α) = (1−α)<em>z</em><sub>A</sub> + α<em>z</em><sub>B</sub> (Figur 4), <strong>Vanilla AE</strong> dan <strong>Sparse AE</strong> menampilkan artefak visual rusak berbentuk tumpang-tindih bayangan (<em>ghosting</em>) di titik tengah α ∈ [0.4, 0.6]. Hal ini terjadi karena AE deterministik tidak memiliki regulasi kepadatan ruang laten, sehingga setiap citra dipetakan ke titik-titik terisolasi dan menyisakan wilayah kosong (<em>latent space voids</em>) di antara klaster kelas yang belum pernah dilatih oleh Decoder. Sebaliknya, regularisasi <em>D</em><sub>KL</sub>(<em>q</em>(<em>z</em>|<em>x</em>) ∥ 𝒩(0, <strong>I</strong>)) pada <strong>VAE</strong> memaksa posterior tiap sampel saling bertumpang-tindih mengelilingi asal koordinat, membentuk manifold kontinu yang menghasilkan morfing bentuk mulus dari <em>Sneaker</em> ke <em>Dress</em>. Pada <strong>DCGAN</strong>, pengambilan dari prior Gaussian 𝒩(0, <strong>I</strong>) yang terisi penuh oleh konvolusi menghasilkan transisi semantik yang tetap tajam di setiap α, berbeda dengan <strong>Vanilla GAN</strong> yang memperlihatkan transisi kasar berderau.</p>
</div>

<div class="qa-card">
  <div class="qa-title">Q5 — Sintesis Pembuktian Batasan MLP Classifier untuk Image Generation</div>
  <p>Hasil Bagian 2.1 membuktikan secara konsisten bahwa performa klasifikasi tinggi ({mlp['test_accuracy']*100:.2f}%) tidak berbanding lurus dengan kemampuan generatif. Pemodelan <em>P</em>(<em>y</em>|<em>x</em>) bekerja sebagai <em>information bottleneck</em> yang membuang detail piksel intra-kelas (terbukti dari lonjakan Test MSE Frozen Latent Decoder menjadi <strong>{mlp['frozen_decoder_metrics']['mse']:.5f}</strong> berbanding <strong>{m['vanilla_ae']['mse']:.5f}</strong> pada Vanilla AE berdimensi 64 yang sama). Kedua, proyeksi balik langsung <em>W</em><sup>T</sup> gagal total (MSE {mlp['transpose_onehot_metrics']['mse']:.4f} untuk <em>y</em><sub><em>c</em></sub> dan {mlp['transpose_latent_metrics']['mse']:.4f} untuk <em>z</em><sub>MLP</sub>) akibat matriks bobot non-ortogonal (deviasi hingga {mlp['orthogonality']['W1_orthogonality_error']:.2f} pada <em>W</em><sub>1</sub>) dan pemotongan ireversibel ReLU ({mlp['z_mlp_zero_fraction']*100:.2f}% nol). Ketiga, tidak adanya regularisasi <em>P</em>(<em>z</em>) membuat ruang laten <em>z</em><sub>MLP</sub> terkonsentrasi jauh dari distribusi Gaussian standar (‖μ<sub><em>z</em></sub>‖ = {mlp['z_mlp_mu_norm']:.2f}), sehingga sampling acak menghasilkan citra rusak.</p>
</div>

<div class="qa-card">
  <div class="qa-title">Q6 — Sintesis Rekayasa dan Rekomendasi Deployment Industri</div>
  <ol>
    <li><strong>Deteksi Anomali Tanpa Supervisi pada Citra Inspeksi Manufaktur (Vanilla AE vs. Sparse AE):</strong> Direkomendasikan <strong>Sparse Autoencoder</strong>. Meskipun Vanilla AE memiliki Test MSE sedikit lebih rendah ({m['vanilla_ae']['mse']:.5f} vs {m['sparse_ae']['mse']:.5f}), Vanilla AE berisiko bertindak sebagai <em>identity mapping</em> yang tetap merekonstruksi cacat kecil. Penalti sparsity L1 pada Sparse AE memaksa encoder hanya menyimpan <em>basis fitur normal yang dominan</em> ({m['sparse_ae']['latent_zero_frac']*100:.2f}% unit nol, L1 turun {sae_l1_ratio:.2f}×), sehingga pola anomali asing yang tidak cocok dengan kamus fitur jarang menghasilkan lonjakan <em>reconstruction error</em> yang jauh lebih sensitif dan diskriminatif.</li>
    <li><strong>Filter Avatar Interaktif Waktu-Nyata pada Perangkat Mobile (&lt;30 ms) (Vanilla GAN vs. DCGAN vs. VAE):</strong> Direkomendasikan <strong>DCGAN</strong> (untuk generasi tajam tanpa encoder) atau <strong>VAE</strong> (apabila pipeline membutuhkan modul <em>encode-edit-decode</em> dari citra input pengguna). Dengan latensi <strong>{m['dcgan']['latency_ms_mean']:.2f} ms/batch 64</strong> (&lt;0.1 ms/citra pada GPU) dan FID <strong>{m['dcgan']['fid']:.2f}</strong>, DCGAN berada jauh di bawah anggaran 30 ms sekaligus jauh lebih tajam dibanding VAE (FID {m['vae']['fid']:.2f}) dan lebih koheren dibanding Vanilla GAN (FID {m['vanilla_gan']['fid']:.2f}). Untuk perangkat CPU/NPU mobile ringan, arsitektur konvolusional DCGAN dapat dipangkas menggunakan <em>depthwise separable convolution</em> tanpa mengorbankan anggaran 30 ms.</li>
    <li><strong>Pembuatan Aset Visual Beresolusi &amp; Berfidelitas Tinggi untuk Industri Kreatif (DCGAN vs. DDPM):</strong> Direkomendasikan <strong>DDPM</strong> (dengan akselerasi sampling seperti DDIM/DPM-Solver pada tahap produksi). Secara empiris, DDPM meraih fidelitas terbaik (FID <strong>{m['ddpm']['fid']:.2f}</strong> vs {m['dcgan']['fid']:.2f} pada DCGAN), stabilitas pelatihan superior tanpa risiko <em>mode collapse</em>, serta kontrol detail tekstural halus (Figur 2D &amp; 5). Pada aset kreatif non-real-time (rendering desain katalog/konsep), waktu inferensi ~{m['ddpm']['latency_ms_mean']/1000:.2f} detik per 64 sampel pada 1.000 langkah masih sangat layak—dan dapat dipangkas 10–20× menggunakan sampler deterministik DDIM (50 langkah) sambil mempertahankan keunggulan kualitas difusi.</li>
  </ol>
</div>

</body>
</html>
"""


def main() -> None:
    cfg = json.loads((ROOT / "report_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))

    build_dir = ROOT / "report" / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    html_path = build_dir / "report.html"
    html_path.write_text(build_html(cfg, metrics), encoding="utf-8")

    safe_nim = cfg["nim"].strip().replace(" ", "_")
    safe_name = cfg["nama_lengkap"].strip().replace(" ", "_")
    pdf_name = f"TUGAS_DL_SESI07_{safe_nim}_{safe_name}.pdf"
    pdf_path = ROOT / pdf_name

    browser = _find_browser_binary()
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0 or not pdf_path.exists():
        raise RuntimeError(f"PDF generation failed (code={res.returncode}): {res.stderr}")

    num_pages = len(PdfReader(str(pdf_path)).pages)
    print(f"Generated PDF: {pdf_path} ({num_pages} pages)")
    if not (10 <= num_pages <= 12):
        print(f"WARNING: Page count is {num_pages}, target is 10-12 pages!", file=sys.stderr)


if __name__ == "__main__":
    main()
