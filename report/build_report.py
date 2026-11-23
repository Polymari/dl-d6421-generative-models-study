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
      content: "Deep Learning (D6421): Sesi 07, Generative Models";
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
    font-size: 7.9pt;
    margin: 5px 0 8px 0;
    break-inside: avoid;
  }}
  .data-table th {{
    background: #1e3a8a;
    color: #ffffff;
    font-weight: 700;
    padding: 4.5px 5.5px;
    border: 1px solid #1e3a8a;
    text-align: left;
  }}
  .data-table td {{
    padding: 3.6px 5.5px;
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
    gap: 5px;
    margin: 5px 0 7px 0;
  }}
  .diagram-card {{
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 4px 7px;
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

<p>Laporan ini membandingkan enam arsitektur generatif dari materi Sesi 06 dan Sesi 07 (Vanilla Autoencoder, Sparse Autoencoder, Variational Autoencoder, Vanilla GAN berbasis MLP, Deep Convolutional GAN, dan Denoising Diffusion Probabilistic Model) bersama satu model pendamping MLP Classifier. Pengujian berfokus pada pertukaran antara fidelitas rekonstruksi, ketajaman sampel, kestabilan pelatihan, dan latensi inferensi, sekaligus membuktikan mengapa fitur klasifikasi terawasi tidak dapat langsung dipakai untuk membangkitkan citra.</p>

<div class="callout">
  <div class="callout-title">Prinsip Evaluasi Terstandarisasi</div>
  Seluruh model dilatih dan diuji pada dataset Fashion-MNIST ({ds['name']}, 70.000 citra 28×28 grayscale dalam 10 kelas pakaian) dengan <code>seed={ds['seed']}</code> di Python, NumPy, dan PyTorch CUDA. Pembagian data dilakukan secara terstratifikasi menjadi 80% Train ({ds['train_size']:,} citra), 10% Validation ({ds['val_size']:,} citra), dan 10% Test ({ds['test_size']:,} citra) dengan <code>batch_size={ds['batch_size']}</code> (<code>shuffle=True</code> pada train, <code>shuffle=False</code> pada validation dan test). Semua angka metrik dan visualisasi berasal dari Test Set pada GPU NVIDIA GeForce RTX 4060 Laptop (8 GB VRAM).
</div>

<!-- ==================== BAGIAN II ==================== -->
<h2 class="section-hdr">Bagian II: Landasan Teori, Desain Arsitektur, dan Formulasi Matematika</h2>

<p>Perbedaan hasil antarmodel pada citra pakaian muncul dari dua sumber: bias arsitektur (lapisan fully-connected dibanding konvolusi spasial) dan bentuk fungsi loss yang dioptimasi. Ketujuh arsitektur yang diuji mengikuti rancangan skematik berikut.</p>

{svgs_html}

<h3 class="sub-hdr">2.1 Konfigurasi Parameter, Jumlah Bobot, dan Hiperparameter Terstandarisasi</h3>
<table class="data-table">
  <thead>
    <tr>
      <th>Model</th>
      <th>Lapisan Arsitektur Utama</th>
      <th class="num">Params</th>
      <th>Dim Laten</th>
      <th>Aktivasi</th>
      <th>Optimizer &amp; LR</th>
      <th class="num">Epoch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>MLP Classifier</strong></td>
      <td>784 → 256 → 64 → 10</td>
      <td class="num">218,058</td>
      <td class="num">64 (<em>z</em><sub>MLP</sub>)</td>
      <td>ReLU, Logits</td>
      <td>Adam (1e-3, Cosine)</td>
      <td class="num">20</td>
    </tr>
    <tr>
      <td><strong>Frozen Decoder</strong></td>
      <td>64 → 256 → 784 (pada <em>z</em><sub>MLP</sub> beku)</td>
      <td class="num">218,128</td>
      <td class="num">64</td>
      <td>ReLU, Sigmoid</td>
      <td>Adam (1e-3)</td>
      <td class="num">30</td>
    </tr>
    <tr>
      <td><strong>Vanilla AE</strong></td>
      <td>Enc: 784→256→64 | Dec: 64→256→784</td>
      <td class="num">435,536</td>
      <td class="num">64 (Linear)</td>
      <td>ReLU, Sigmoid</td>
      <td>Adam (1e-3)</td>
      <td class="num">30</td>
    </tr>
    <tr>
      <td><strong>Sparse AE</strong></td>
      <td>Enc: 784→256→64 | Dec: 64→256→784</td>
      <td class="num">435,536</td>
      <td class="num">64 (ReLU+L1)</td>
      <td>ReLU, Sigmoid</td>
      <td>Adam (1e-3, λ=5e-3)</td>
      <td class="num">30</td>
    </tr>
    <tr>
      <td><strong>VAE</strong></td>
      <td>784→256→(μ:64, logσ²:64)→256→784</td>
      <td class="num">451,984</td>
      <td class="num">64 (Gaussian)</td>
      <td>ReLU, Sigmoid</td>
      <td>Adam (1e-3)</td>
      <td class="num">30</td>
    </tr>
    <tr>
      <td><strong>Vanilla GAN</strong></td>
      <td>G: 64→256→512→784 | D: 784→512→256→1</td>
      <td class="num">1,083,921</td>
      <td class="num">64 (Prior)</td>
      <td>LeakyReLU(0.2), Tanh</td>
      <td>Adam (2e-4, β=(0.5,0.999))</td>
      <td class="num">50</td>
    </tr>
    <tr>
      <td><strong>DCGAN</strong></td>
      <td>G: ConvT(100→256→128→1) | D: Conv(1→64→128→256)</td>
      <td class="num">2,213,377</td>
      <td class="num">100 (Prior)</td>
      <td>ReLU/LReLU, BN, Tanh</td>
      <td>Adam (2e-4, β=(0.5,0.999), ls=0.9)</td>
      <td class="num">50</td>
    </tr>
    <tr>
      <td><strong>DDPM</strong></td>
      <td>UNet28 (64→128→256+SelfAttn 7×7), TimeEmb, EMA</td>
      <td class="num">4,577,665</td>
      <td class="num">28×28 (T=1000)</td>
      <td>SiLU, GroupNorm</td>
      <td>AdamW (2e-4, wd=1e-4)</td>
      <td class="num">50</td>
    </tr>
  </tbody>
</table>

<h3 class="sub-hdr">2.2 Formulasi Matematika Fungsi Loss Objektif</h3>
<div class="eq-box">
  1. MLP Classifier (Cross-Entropy): &nbsp; ℒ<sub>CE</sub> = −(1/N) ∑<sub>i=1..N</sub> ∑<sub>c=1..10</sub> y<sub>i,c</sub> log softmax(W<sub>3</sub> z<sub>MLP,i</sub> + b<sub>3</sub>)<sub>c</sub>, yang memodelkan <em>P</em>(<em>y</em>|<em>x</em>).<br/>
  2. Vanilla Autoencoder (Pixel-wise MSE): &nbsp; ℒ<sub>AE</sub> = (1/784) ‖<em>x</em> − <em>D</em><sub>θ</sub>(<em>E</em><sub>φ</sub>(<em>x</em>))‖<sub>2</sub><sup>2</sup>.<br/>
  3. Sparse Autoencoder (MSE + L1 Sparsity): &nbsp; ℒ<sub>SAE</sub> = (1/784) ‖<em>x</em> − <em>x̂</em>‖<sub>2</sub><sup>2</sup> + λ · (1/64) ∑<sub>j=1..64</sub> |<em>z</em><sub>j</sub>|, dengan λ = 5×10<sup>−3</sup>.<br/>
  4. Variational Autoencoder (ELBO): &nbsp; ℒ<sub>VAE</sub> = −𝔼<sub><em>q</em><sub>φ</sub>(<em>z</em>|<em>x</em>)</sub>[log <em>p</em><sub>θ</sub>(<em>x</em>|<em>z</em>)] + <em>D</em><sub>KL</sub>(<em>q</em><sub>φ</sub>(<em>z</em>|<em>x</em>) ∥ 𝒩(0, <strong>I</strong>)), dimana <em>D</em><sub>KL</sub> = −½ ∑(1 + log σ² − μ² − σ²) dan <em>z</em> = μ + σ ⊙ ε.<br/>
  5. Vanilla GAN &amp; DCGAN (Minimax Game): &nbsp; min<sub>G</sub> max<sub>D</sub> <em>V</em>(<em>D</em>,<em>G</em>) = 𝔼<sub><em>x</em>~<em>p</em><sub>data</sub></sub>[log <em>D</em>(<em>x</em>)] + 𝔼<sub><em>z</em>~𝒩(0,<strong>I</strong>)</sub>[log(1 − <em>D</em>(<em>G</em>(<em>z</em>)))], dilatih dengan objektif <em>non-saturating</em> Generator −𝔼[log <em>D</em>(<em>G</em>(<em>z</em>))].<br/>
  6. DDPM (Variational Noise-Prediction MSE): &nbsp; ℒ<sub>simple</sub>(θ) = 𝔼<sub><em>t</em>~𝒰(1,<em>T</em>), <em>x</em><sub>0</sub>, ε~𝒩(0,<strong>I</strong>)</sub>[‖ε − ε<sub>θ</sub>(√ᾱ<sub><em>t</em></sub> <em>x</em><sub>0</sub> + √(1 − ᾱ<sub><em>t</em></sub>) ε, <em>t</em>)‖<sub>2</sub><sup>2</sup>], dengan jadwal varians linear β<sub><em>t</em></sub> ∈ [10<sup>−4</sup>, 0.02] untuk <em>T</em> = 1000 langkah.
</div>

<!-- ==================== BAGIAN III ==================== -->
<h2 class="section-hdr">Bagian III: Pembuktian Teoretis dan Empiris Batasan MLP Classifier untuk Image Generation</h2>

<p>Lapisan ekstraktor fitur MLP Classifier (<code>784 → 256 → 64</code>) punya topologi yang sama persis dengan Encoder pada Autoencoder. Namun, bobot terlatih <em>W</em><sup>(<em>l</em>)</sup> dan fitur laten <em>z</em><sub>MLP</sub> ∈ ℝ<sup>64</sup> gagal merekonstruksi maupun membangkitkan citra karena tiga batasan matematis:</p>

<ol>
  <li>Information bottleneck dari pemodelan <em>P</em>(<em>y</em>|<em>x</em>): Cross-Entropy Loss mendorong pemisahan antar-10 kelas, sehingga jaringan membuang variasi spasial intra-kelas (ketebalan garis, lipatan kain, corak baju) yang tidak dibutuhkan untuk menebak label. Model generatif sebaliknya harus menjaga informasi untuk memodelkan kerapatan data <em>P</em>(<em>x</em>).</li>
  <li>Matriks bobot non-ortogonal dan pemotongan ReLU: Bobot persegi panjang <em>W</em><sup>(<em>l</em>)</sup> tidak memenuhi (<em>W</em><sup>(<em>l</em>)</sup>)<sup>T</sup> <em>W</em><sup>(<em>l</em>)</sup> = <strong>I</strong>, jadi <em>W</em><sup>T</sup> ≠ <em>W</em><sup>−1</sup>. Pengukuran pada model terlatih mencatat deviasi ortogonalitas relatif ‖<em>W</em><sup>T</sup><em>W</em> − <strong>I</strong>‖<sub>F</sub> / ‖<strong>I</strong>‖<sub>F</sub> sebesar {mlp['orthogonality']['W1_orthogonality_error']:.4f} (<em>W</em><sub>1</sub>), {mlp['orthogonality']['W2_orthogonality_error']:.4f} (<em>W</em><sub>2</sub>), dan {mlp['orthogonality']['W3_orthogonality_error']:.4f} (<em>W</em><sub>3</sub>). Selain itu, fungsi ReLU(<em>u</em>) = max(0, <em>u</em>) membuang nilai negatif secara ireversibel, membuat {mlp['z_mlp_zero_fraction']*100:.2f}% komponen <em>z</em><sub>MLP</sub> bernilai tepat nol.</li>
  <li>Tanpa regularisasi prior <em>P</em>(<em>z</em>): Ruang laten <em>z</em><sub>MLP</sub> tidak dikekang ke distribusi prior seperti 𝒩(0, <strong>I</strong>). Pada Test Set, vektor rata-rata fitur memiliki norma ‖μ<sub><em>z</em></sub>‖<sub>2</sub> = {mlp['z_mlp_mu_norm']:.2f} dengan rata-rata simpangan baku σ̄<sub><em>z</em></sub> = {mlp['z_mlp_sigma_mean']:.2f} dan sebaran miring akibat ReLU.</li>
</ol>

<div class="callout">
  <div class="callout-title">Analisis Teoretis <em>P</em>(<em>y</em>|<em>x</em>) vs. <em>P</em>(<em>x</em>) dan Ketidakbalikan Aljabar Linear</div>
  Secara informasi-teoretis, rantai Markov pelatihan terawasi <em>X</em> → <em>H</em><sub>1</sub> → <em>Z</em><sub>MLP</sub> → <em>Y</em> tunduk pada <em>Data Processing Inequality</em>: optimasi ℒ<sub>CE</sub> hanya mempertahankan mutual information <em>I</em>(<em>Z</em><sub>MLP</sub>; <em>Y</em>) ≤ log<sub>2</sub>(10) ≈ 3.32 bit per sampel dan membuang informasi kondisional <em>H</em>(<em>X</em>|<em>Y</em>) yang mencakup ragam tekstur dan siluet di dalam satu kelas. Sebaliknya, Autoencoder memaksimalkan <em>I</em>(<em>X</em>; <em>Z</em><sub>AE</sub>) agar <em>X</em> dapat dipulihkan dari <em>Z</em><sub>AE</sub>. Dari sisi aljabar linear, proyeksi balik <em>x̂</em> = σ(<em>W</em><sub>1</sub><sup>T</sup> ReLU(<em>W</em><sub>2</sub><sup>T</sup> <em>z</em><sub>MLP</sub>)) mengasumsikan pemetaan ortonormal (<em>W</em><sup>T</sup><em>W</em> = <strong>I</strong>) serta aktivasi bijektif; karena <em>W</em><sub>1</sub> ∈ ℝ<sup>256×784</sup> tidak berkolom ortogonal dan ReLU memangkas setengah ruang vektor, rekonstruksi aljabar langsung merosot menjadi superposisi bobot kasar.
</div>

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
      <td>Tahap 1: MLP Classifier (784→256→64→10)</td>
      <td class="num">-</td><td class="num">-</td><td class="num">-</td>
      <td>Test Acc = {mlp['test_accuracy']*100:.2f}% (Val = {mlp['val_accuracy']*100:.2f}%, di atas batas minimum 88%)</td>
    </tr>
    <tr>
      <td>Tahap 2: Proyeksi Balik Bobot <em>W</em><sup>T</sup> dari One-Hot <em>y</em><sub><em>c</em></sub></td>
      <td class="num">{mlp['transpose_onehot_metrics']['mse']:.5f}</td>
      <td class="num">{mlp['transpose_onehot_metrics']['psnr']:.2f}</td>
      <td class="num">{mlp['transpose_onehot_metrics']['ssim']:.4f}</td>
      <td>Pola bercak abu-abu seragam akibat superposisi template kasar (<em>W</em><sup>T</sup> ≠ <em>W</em><sup>−1</sup>)</td>
    </tr>
    <tr>
      <td>Tahap 3: Proyeksi Balik <em>W</em><sup>T</sup> dari Latent <em>z</em><sub>MLP</sub></td>
      <td class="num">{mlp['transpose_latent_metrics']['mse']:.5f}</td>
      <td class="num">{mlp['transpose_latent_metrics']['psnr']:.2f}</td>
      <td class="num">{mlp['transpose_latent_metrics']['ssim']:.4f}</td>
      <td>Artefak kontras biner; bobot maju tidak dapat dibalik menjadi dekoder</td>
    </tr>
    <tr>
      <td>Tahap 4a: Frozen Latent Decoder (<em>z</em><sub>MLP</sub> ∈ ℝ<sup>64</sup>)</td>
      <td class="num">{mlp['frozen_decoder_metrics']['mse']:.5f}</td>
      <td class="num">{mlp['frozen_decoder_metrics']['psnr']:.2f}</td>
      <td class="num">{mlp['frozen_decoder_metrics']['ssim']:.4f}</td>
      <td>Siluet kelas kembali terlihat, namun tekstur intra-kelas hilang ({mse_ratio:.2f}× MSE Vanilla AE)</td>
    </tr>
    <tr>
      <td>Tahap 4b (Pembanding): Vanilla AE (<em>z</em><sub>AE</sub> ∈ ℝ<sup>64</sup>)</td>
      <td class="num">{m['vanilla_ae']['mse']:.5f}</td>
      <td class="num">{m['vanilla_ae']['psnr']:.2f}</td>
      <td class="num">{m['vanilla_ae']['ssim']:.4f}</td>
      <td>Menjaga detail intra-kelas karena dilatih untuk rekonstruksi penuh</td>
    </tr>
    <tr>
      <td>Tahap 5: Prior Sampling <em>z</em><sub>rand</sub> ~ 𝒩(μ<sub><em>z</em></sub>, diag(σ<sub><em>z</em></sub><sup>2</sup>))</td>
      <td class="num">-</td><td class="num">-</td><td class="num">-</td>
      <td>Menghasilkan pecahan siluet dan bercak yang tidak membentuk pakaian utuh</td>
    </tr>
  </tbody>
</table>

<!-- ==================== BAGIAN IV ==================== -->
<h2 class="section-hdr">Bagian IV: Tabel Komparasi Metrik Numerik (Kompilasi Kuantitatif)</h2>

<p>Metrik rekonstruksi (Test MSE, Test PSNR, Test SSIM) dihitung pada rentang piksel [0, 1] atas 7.000 citra Test Set. Skor FID (Fréchet Inception Distance) membandingkan fitur pool3 (2048 dimensi) dari InceptionV3 antara 7.000 citra sintetis dan 7.000 citra Test Set riil. Latensi inferensi diukur per batch 64 citra dengan <code>torch.cuda.synchronize()</code>.</p>

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
      <td class="num">{m['vanilla_ae']['latency_ms_mean']:.2f} ± {m['vanilla_ae']['latency_ms_std']:.2f}</td>
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
      <td class="num">{m['sparse_ae']['latency_ms_mean']:.2f} ± {m['sparse_ae']['latency_ms_std']:.2f}</td>
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
      <td class="num">{m['vae']['latency_ms_mean']:.2f} ± {m['vae']['latency_ms_std']:.2f}</td>
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
      <td class="num">{m['vanilla_gan']['latency_ms_mean']:.2f} ± {m['vanilla_gan']['latency_ms_std']:.2f}</td>
      <td class="num">N/A (Tanpa Encoder)</td>
      <td class="num">N/A</td>
      <td class="num">N/A</td>
      <td class="num">{m['vanilla_gan']['fid']:.2f}</td>
      <td>Rentan Osilasi Awal (D={vgan_d_end:.2f}, G={vgan_g_end:.2f})</td>
    </tr>
    <tr>
      <td><strong>DCGAN</strong></td>
      <td>100 (Prior)</td>
      <td class="num">{m['dcgan']['epoch_time_s']:.2f}</td>
      <td class="num">{m['dcgan']['latency_ms_mean']:.2f} ± {m['dcgan']['latency_ms_std']:.2f}</td>
      <td class="num">N/A (Tanpa Encoder)</td>
      <td class="num">N/A</td>
      <td class="num">N/A</td>
      <td class="num">{m['dcgan']['fid']:.2f}</td>
      <td>Cukup Stabil (D={dc_d_end:.2f}, G={dc_g_end:.2f})</td>
    </tr>
    <tr>
      <td><strong>DDPM (UNet28 + EMA)</strong></td>
      <td>28×28</td>
      <td class="num">{m['ddpm']['epoch_time_s']:.2f}</td>
      <td class="num">{m['ddpm']['latency_ms_mean']:.2f} ± {m['ddpm']['latency_ms_std']:.2f}</td>
      <td class="num">N/A (Denoising Model)</td>
      <td class="num">N/A</td>
      <td class="num">N/A</td>
      <td class="num"><strong>{m['ddpm']['fid']:.2f}</strong></td>
      <td>Sangat Stabil (Noise MSE {ddpm_start:.4f} → {ddpm_end:.4f})</td>
    </tr>
  </tbody>
</table>

<p>Penambahan penalti sparsity λ‖<em>z</em>‖<sub>1</sub> (λ = 5×10<sup>−3</sup>) menurunkan rata-rata norma L1 representasi laten dari {m['vanilla_ae']['latent_l1_mean']:.2f} pada Vanilla AE (hanya {m['vanilla_ae']['latent_zero_frac']*100:.2f}% unit mendekati nol) menjadi {m['sparse_ae']['latent_l1_mean']:.2f} pada Sparse AE dengan {m['sparse_ae']['latent_zero_frac']*100:.2f}% unit non-aktif (|<em>z</em>| &lt; 10<sup>−3</sup>). Pada pengujian kualitas generasi, skor FID terbagi jelas antarkeluarga: DDPM mencatat FID terendah ({m['ddpm']['fid']:.2f}), disusul DCGAN ({m['dcgan']['fid']:.2f}), sedangkan VAE ({m['vae']['fid']:.2f}) dan Vanilla GAN ({m['vanilla_gan']['fid']:.2f}) tertinggal jauh karena keburaman posterior Gaussian serta ketiadaan lapisan konvolusi.</p>

<!-- ==================== BAGIAN V ==================== -->
<h2 class="section-hdr page-break">Bagian V: Kompilasi Lima Visualisasi Komparatif Utama</h2>

<div class="figure-block">
  <img src="{fig1_uri}" alt="Figur 1 Rekonstruksi Side-by-Side" />
  <div class="fig-caption"><strong>Figur 1: Grid Rekonstruksi Side-by-Side (Test Set).</strong> Perbandingan 8 kelas Fashion-MNIST antara citra asli dan rekonstruksi Vanilla AE (MSE {m['vanilla_ae']['mse']:.5f}), Sparse AE ({m['sparse_ae']['mse']:.5f}), dan VAE ({m['vae']['mse']:.5f}). Vanilla AE menjaga detail lokal paling jelas (seperti motif kotak pada <em>Shirt</em> dan tekstur <em>Coat</em>), Sparse AE agak menghaluskan tepi karena memangkas {m['sparse_ae']['latent_zero_frac']*100:.2f}% aktivasi laten, dan VAE terlihat paling kabur karena regularisasi KL meratakan variasi lokal.</div>
</div>

<div class="figure-block">
  <img src="{fig2_uri}" alt="Figur 2 Random Generative Sampling 4x4" />
  <div class="fig-caption"><strong>Figur 2: Grid Random Generative Sampling (Kisi 4×4 per Model).</strong> Sampel acak dari prior 𝒩(0, <strong>I</strong>). Panel A (VAE, FID {m['vae']['fid']:.2f}) membentuk struktur pakaian yang dapat dikenali tetapi tepinya buram. Panel B (Vanilla GAN, FID {m['vanilla_gan']['fid']:.2f}) menghasilkan kontras tinggi yang dipenuhi bintik derau piksel karena generator MLP tidak punya bias spasial. Panel C (DCGAN, FID {m['dcgan']['fid']:.2f}) menajamkan siluet berkat <em>strided transposed convolution</em> dan BatchNorm. Panel D (DDPM, FID {m['ddpm']['fid']:.2f}) memberi kontur paling bersih, gradasi kain halus, dan sebaran kelas merata.</div>
</div>

<div class="figure-block">
  <img src="{fig3_uri}" alt="Figur 3 Bukti Empiris MLP Classifier" />
  <div class="fig-caption"><strong>Figur 3: Bukti Empiris Batasan Bobot dan Latent Features MLP Classifier.</strong> (1) Citra uji asli. (2) Proyeksi balik langsung <em>W</em><sup>T</sup> dari <em>y</em><sub><em>c</em></sub> menghasilkan bercak abu-abu yang mirip di semua kelas, sementara dari <em>z</em><sub>MLP</sub> muncul pola biner acak. (3) Frozen Latent Decoder hanya memulihkan siluet kasar kelas (PSNR {mlp['frozen_decoder_metrics']['psnr']:.2f} dB dibanding Vanilla AE {m['vanilla_ae']['psnr']:.2f} dB; corak <em>Shirt</em> paling kanan hilang menjadi kemeja polos). (4) Sampling prior <em>z</em><sub>rand</sub> dari statistik <em>z</em><sub>MLP</sub> menghasilkan bentuk yang pecah dan tidak utuh.</div>
</div>

<div class="figure-block">
  <img src="{fig4_uri}" alt="Figur 4 Interpolasi Ruang Laten Linier" />
  <div class="fig-caption"><strong>Figur 4: Interpolasi Ruang Laten Linier <em>z</em>(α) = (1−α)<em>z</em><sub>A</sub> + α<em>z</em><sub>B</sub> (α ∈ [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).</strong> Pada Vanilla AE, Sparse AE, dan VAE, interpolasi dilakukan pada pasangan uji <em>Sneaker</em> (α=0) dan <em>Dress</em> (α=1); pada GAN dan DCGAN (tanpa encoder), interpolasi dilakukan antara dua vektor prior acak. Vanilla AE dan Sparse AE melewati ruang kosong (<em>latent space voids</em>) di α=0.4 sampai 0.6 sehingga memunculkan bayangan ganda. VAE bertransisi halus tanpa gambar pecah, Vanilla GAN tetap berderau di seluruh langkah, dan DCGAN beralih tajam dari atasan berlengan panjang terang menjadi kaus berlengan pendek gelap.</div>
</div>

<div class="figure-block">
  <img src="{fig5_uri}" alt="Figur 5 Kurva Training Loss dan DDPM Denoising Trajectory" />
  <div class="fig-caption"><strong>Figur 5: Kurva Training Loss Keenam Model dan Trajektori Reverse Denoising DDPM.</strong> Vanilla AE, Sparse AE, dan VAE turun mulus dengan kurva train dan validation yang berimpit. Pada Vanilla GAN, loss Generator sempat melonjak ke {vgan_g_max:.2f} di epoch {vgan_g_max_ep} sebelum turun stabil, sedangkan pada DCGAN Discriminator memimpin cukup jauh (D≈{dc_d_end:.2f}, G≈{dc_g_end:.2f}) tetapi tetap menghasilkan bentuk konvolusional yang tajam. DDPM turun monoton pada Noise MSE ({ddpm_start:.4f} ke {ddpm_end:.4f}); baris bawah memperlihatkan perubahan bertahap dari derau Gaussian di <em>t</em>=999 menjadi sepatu dan gaun bersih di <em>t</em>=0.</div>
</div>

<!-- ==================== BAGIAN VI ==================== -->
<h2 class="section-hdr page-break">Bagian VI: Pembahasan Kritis, Jawaban Pertanyaan Penelitian (Q1 sampai Q6), dan Rekomendasi Industri</h2>

<div class="qa-card">
  <div class="qa-title">Q1: Perbandingan Kuantitatif Reconstruction Fidelity vs. Generative Quality</div>
  <p>Untuk rekonstruksi Test Set, Vanilla AE mencapai hasil terbaik (MSE = {m['vanilla_ae']['mse']:.5f}, PSNR = {m['vanilla_ae']['psnr']:.2f} dB, SSIM = {m['vanilla_ae']['ssim']:.4f}), diikuti Sparse AE (MSE = {m['sparse_ae']['mse']:.5f}, PSNR = {m['sparse_ae']['psnr']:.2f} dB, SSIM = {m['sparse_ae']['ssim']:.4f}) dan VAE (MSE = {m['vae']['mse']:.5f}, PSNR = {m['vae']['psnr']:.2f} dB, SSIM = {m['vae']['ssim']:.4f}). Penalti L1 λ‖<em>z</em>‖<sub>1</sub> (λ=5×10<sup>−3</sup>) memaksa {m['sparse_ae']['latent_zero_frac']*100:.2f}% dimensi laten Sparse AE bernilai mendekati nol dan memangkas norma L1 sebesar {sae_l1_ratio:.2f}× ({m['sparse_ae']['latent_l1_mean']:.2f} dibanding {m['vanilla_ae']['latent_l1_mean']:.2f}). Penyempitan kapasitas laten ini menaikkan MSE sebesar +{sae_mse_diff:.5f} ({sae_psnr_diff:.2f} dB PSNR) dibanding representasi padat Vanilla AE.</p>
  <p>Pada sintesis sampel baru dari prior 𝒩(0, <strong>I</strong>), urutan skor FID dari yang paling rendah adalah DDPM ({m['ddpm']['fid']:.2f}), DCGAN ({m['dcgan']['fid']:.2f}), VAE ({m['vae']['fid']:.2f}), dan Vanilla GAN ({m['vanilla_gan']['fid']:.2f}). Angka ini menunjukkan bahwa galat rekonstruksi yang rendah tidak menjamin kemampuan membangkitkan data baru: Vanilla AE dan Sparse AE tidak memiliki prior yang teratur untuk disampel, sementara VAE menukar ketajaman citra dengan ruang laten yang kontinu.</p>
</div>

<div class="qa-card">
  <div class="qa-title">Q2: Analisis Blurriness vs. Sharp Edges dan Struktur Representasi Fitur</div>
  <p>Efek buram pada Vanilla AE, Sparse AE, dan VAE berasal dari sifat <em>mode-averaging</em> pada minimisasi <em>pixel-wise loss</em> (MSE atau BCE). Nilai <em>x̂</em> yang meminimalkan 𝔼[‖<em>x</em> − <em>x̂</em>‖<sub>2</sub><sup>2</sup> | <em>z</em>] secara analitik adalah rata-rata kondisional 𝔼[<em>x</em>|<em>z</em>]. Saat beberapa variasi kontur atau tekstur berpeluang muncul dari kode laten <em>z</em> yang berdekatan, perataan piksel meredam frekuensi tinggi sehingga tepi pakaian tampak kabur. Pada VAE, suku KL divergence menambah tumpang-tindih antar-posterior dan mendorong perilaku <em>zero-avoiding</em> (menutupi seluruh sebaran data termasuk ruang di antara mode), sehingga perataan makin terasa.</p>
  <p>Sebaliknya, pelatihan minimax pada GAN dan DCGAN bersifat <em>mode-seeking</em> karena meminimalkan divergensi Jensen-Shannon <em>D</em><sub>JS</sub>(<em>P</em><sub>data</sub> ∥ <em>P</em><sub><em>g</em></sub>) ketika Discriminator optimal. Citra kabur berada di daerah kerapatan rendah antara dua mode tajam, sehingga langsung ditolak sebagai sampel palsu oleh Discriminator dan memaksa Generator memilih kontur yang tegas. Pada DDPM, regresi ε<sub>θ</sub>(<em>x</em><sub><em>t</em></sub>, <em>t</em>) mengestimasi fungsi skor ∇<sub><em>x</em></sub> log <em>p</em><sub><em>t</em></sub>(<em>x</em>) di berbagai skala derau, sehingga struktur global dan detail tajam terbentuk bertahap tanpa kolaps mode maupun keburaman rata-rata. Pada sisi representasi fitur, penalti L1 membuat hidden layer Sparse AE hanya mengaktifkan sebagian kecil unit untuk tiap jenis pakaian ({m['sparse_ae']['latent_zero_frac']*100:.2f}% unit nol), berbeda dari representasi padat pada Vanilla AE di mana hampir seluruh unit aktif bersamaan.</p>
</div>

<div class="qa-card">
  <div class="qa-title">Q3: Stabilitas Training, Dinamika Arsitektur, dan Verifikasi Generative Trilemma</div>
  <p>Pada Vanilla GAN berbasis MLP, ketiadaan <em>weight sharing</em> spasial membuat Discriminator mudah menangkap ketidakkonsistenan piksel lokal, sehingga loss Generator sempat melonjak ke {vgan_g_max:.2f} di epoch {vgan_g_max_ep} sebelum turun ke {vgan_g_end:.2f}. DCGAN meredam masalah ini lewat konvolusi berstrided tanpa lapisan pooling, penghapusan lapisan fully-connected tersembunyi, <code>BatchNorm2d</code> untuk menjaga skala gradien, aktivasi <code>LeakyReLU</code>, serta inisialisasi bobot 𝒩(0, 0.02). Perubahan arsitektur itu menurunkan FID dari {m['vanilla_gan']['fid']:.2f} menjadi {m['dcgan']['fid']:.2f}, walau pelatihan adversarial tetap sensitif terhadap dominasi Discriminator (D={dc_d_end:.2f}, G={dc_g_end:.2f}).</p>
  <p>Berbeda dari permainan minimax yang non-stasioner, DDPM melatih regresi <em>noise prediction</em> ‖ε − ε<sub>θ</sub>‖<sub>2</sub><sup>2</sup> pada target derau yang stasioner, sehingga kurvanya turun paling mulus ({ddpm_start:.4f} ke {ddpm_end:.4f} selama 50 epoch). Pengukuran latensi per batch 64 citra mengonfirmasi trade-off Generative Trilemma (kualitas sampel, kecepatan sampling, dan cakupan mode):</p>
  <ul>
    <li>VAE menghasilkan sampel paling cepat ({m['vae']['latency_ms_mean']:.2f} ms/batch) dengan cakupan kelas luas, tetapi kualitas visualnya kabur (FID {m['vae']['fid']:.2f}).</li>
    <li>DCGAN menjaga sampling satu langkah tetap cepat ({m['dcgan']['latency_ms_mean']:.2f} ms/batch) dengan kontur tajam (FID {m['dcgan']['fid']:.2f}), namun arsitekturnya rentan terhadap ketidakseimbangan G dan D.</li>
    <li>DDPM meraih kualitas sampel terbaik (FID {m['ddpm']['fid']:.2f}) dan cakupan ragam pakaian penuh, tetapi proses denoising iteratif <em>T</em>=1000 langkah membutuhkan {m['ddpm']['latency_ms_mean']:.2f} ms/batch (~{ddpm_vs_dcgan_lat:,.0f}× lebih lambat dari DCGAN dan ~{ddpm_vs_vae_lat:,.0f}× lebih lambat dari VAE).</li>
  </ul>
  <table class="data-table" style="margin-top:5px;">
    <thead>
      <tr>
        <th>Keluarga Model</th>
        <th>Sample Quality (FID)</th>
        <th>Sampling Speed (ms/b64)</th>
        <th>Mode Coverage</th>
        <th>Stabilitas Training</th>
      </tr>
    </thead>
    <tbody>
      <tr><td><strong>VAE</strong></td><td>Rendah (buram, {m['vae']['fid']:.2f})</td><td>Sangat Cepat ({m['vae']['latency_ms_mean']:.2f} ms)</td><td>Tinggi (ELBO covering)</td><td>Tinggi (stasioner)</td></tr>
      <tr><td><strong>Vanilla GAN</strong></td><td>Rendah (derau MLP, {m['vanilla_gan']['fid']:.2f})</td><td>Sangat Cepat ({m['vanilla_gan']['latency_ms_mean']:.2f} ms)</td><td>Rendah-Sedang</td><td>Rendah (osilasi awal)</td></tr>
      <tr><td><strong>DCGAN</strong></td><td>Tinggi (tajam, {m['dcgan']['fid']:.2f})</td><td>Cepat ({m['dcgan']['latency_ms_mean']:.2f} ms)</td><td>Sedang (sensitif)</td><td>Sedang (minimax)</td></tr>
      <tr><td><strong>DDPM</strong></td><td>Tertinggi ({m['ddpm']['fid']:.2f})</td><td>Lambat ({m['ddpm']['latency_ms_mean']:.2f} ms)</td><td>Tinggi (penuh)</td><td>Tertinggi (noise MSE)</td></tr>
    </tbody>
  </table>
</div>

<div class="qa-card">
  <div class="qa-title">Q4: Struktur Latent Space dan Fenomena Interpolasi</div>
  <p>Saat dilakukan interpolasi linier <em>z</em>(α) = (1−α)<em>z</em><sub>A</sub> + α<em>z</em><sub>B</sub> (Figur 4), Vanilla AE dan Sparse AE memunculkan artefak bayangan ganda pada α=0.4 dan 0.6. Penyebabnya adalah ketiadaan regulasi kepadatan ruang laten pada autoencoder deterministik: titik data terkumpul dalam klaster terpisah dan meninggalkan wilayah kosong (<em>latent space voids</em>) yang belum pernah dipelajari Decoder. Regularisasi <em>D</em><sub>KL</sub>(<em>q</em>(<em>z</em>|<em>x</em>) ∥ 𝒩(0, <strong>I</strong>)) pada VAE menarik sebaran laten tiap sampel agar saling bertemu di sekitar titik asal, sehingga transisi dari <em>Sneaker</em> ke <em>Dress</em> berlangsung mulus. Pada DCGAN, seluruh ruang prior 𝒩(0, <strong>I</strong>) terpetakan oleh lapisan konvolusi menjadi objek tajam di setiap α, sedangkan Vanilla GAN tetap memunculkan derau piksel sepanjang lintasan.</p>
</div>

<div class="qa-card">
  <div class="qa-title">Q5: Sintesis Pembuktian Batasan MLP Classifier untuk Image Generation</div>
  <p>Akurasi klasifikasi sebesar {mlp['test_accuracy']*100:.2f}% pada Test Set tidak menghasilkan representasi yang mampu merekonstruksi atau membangkitkan citra. Pertama, objektif <em>P</em>(<em>y</em>|<em>x</em>) bertindak sebagai <em>information bottleneck</em> yang membuang detail intra-kelas, terlihat dari naiknya Test MSE Frozen Latent Decoder menjadi {mlp['frozen_decoder_metrics']['mse']:.5f} dibanding {m['vanilla_ae']['mse']:.5f} pada Vanilla AE berdimensi sama. Kedua, proyeksi balik langsung <em>W</em><sup>T</sup> rusak total (MSE {mlp['transpose_onehot_metrics']['mse']:.4f} dari <em>y</em><sub><em>c</em></sub> dan {mlp['transpose_latent_metrics']['mse']:.4f} dari <em>z</em><sub>MLP</sub>) karena bobot non-ortogonal (deviasi hingga {mlp['orthogonality']['W1_orthogonality_error']:.2f} pada <em>W</em><sub>1</sub>) serta pemotongan ireversibel ReLU ({mlp['z_mlp_zero_fraction']*100:.2f}% nilai nol). Ketiga, tanpa regularisasi <em>P</em>(<em>z</em>), fitur <em>z</em><sub>MLP</sub> terkumpul jauh dari distribusi standar (‖μ<sub><em>z</em></sub>‖ = {mlp['z_mlp_mu_norm']:.2f}), sehingga sampling acak menghasilkan citra pecah.</p>
</div>

<div class="qa-card">
  <div class="qa-title">Q6: Sintesis Rekayasa dan Rekomendasi Deployment Industri</div>
  <ol>
    <li>Deteksi anomali tanpa supervisi pada inspeksi manufaktur (Vanilla AE vs. Sparse AE): Pilih Sparse Autoencoder (435,536 parameter, {m['sparse_ae']['latency_ms_mean']:.2f} ms/batch 64). Meski Vanilla AE mencatat MSE Test lebih kecil ({m['vanilla_ae']['mse']:.5f} berbanding {m['sparse_ae']['mse']:.5f}), kapasitas padatnya rentan meniru pola cacat kecil seolah citra normal. Penalti sparsity L1 pada Sparse AE memaksa encoder hanya menyimpan basis fitur normal yang dominan ({m['sparse_ae']['latent_zero_frac']*100:.2f}% unit nol, norma L1 turun {sae_l1_ratio:.2f}×), sehingga citra cacat yang menyimpang dari basis tersebut langsung memicu galat rekonstruksi lebih tinggi.</li>
    <li>Filter avatar interaktif waktu-nyata pada perangkat mobile (&lt;30 ms) (Vanilla GAN vs. DCGAN vs. VAE): Pilih DCGAN bila filter hanya membangkitkan atau memodifikasi fitur secara tajam tanpa encoder input, atau pilih VAE jika pipeline wajib melakukan <em>encode-edit-decode</em> langsung dari kamera pengguna. Generator DCGAN (1,781,504 parameter) mencatat latensi {m['dcgan']['latency_ms_mean']:.2f} ms per batch 64 (~{m['dcgan']['latency_ms_mean']/64:.3f} ms per citra di GPU) dan FID {m['dcgan']['fid']:.2f}, jauh di bawah batas 30 ms sekaligus jauh lebih tajam dibanding VAE (451,984 parameter, {m['vae']['latency_ms_mean']:.2f} ms/b64, FID {m['vae']['fid']:.2f}) dan Vanilla GAN (Generator 550,416 parameter, FID {m['vanilla_gan']['fid']:.2f}). Pada NPU/CPU ponsel, blok konvolusi DCGAN juga ringan diganti dengan <em>depthwise separable convolution</em>.</li>
    <li>Pembuatan aset visual beresolusi dan berfidelitas tinggi untuk industri kreatif (DCGAN vs. DDPM): Pilih DDPM (UNet28 4,577,665 parameter), dengan akselerator sampling seperti DDIM atau DPM-Solver saat deployment. DDPM menghasilkan fidelitas terbaik (FID {m['ddpm']['fid']:.2f} berbanding {m['dcgan']['fid']:.2f} pada DCGAN), pelatihan paling stabil tanpa risiko <em>mode collapse</em>, serta tekstur kain paling bersih (Figur 2D dan 5). Untuk pembuatan aset kreatif offline yang tidak menuntut respons instan, waktu inferensi {m['ddpm']['latency_ms_mean']/1000:.2f} detik per 64 sampel pada 1.000 langkah masih memadai dan dapat dipangkas 20× (menjadi ~{(m['ddpm']['latency_ms_mean']/1000)/20:.2f} detik/batch) lewat sampler deterministik DDIM berjumlah 50 langkah.</li>
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
