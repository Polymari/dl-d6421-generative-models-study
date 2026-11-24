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
    """Render formal Indonesian academic paper HTML with booktabs tables and embedded figures."""
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

    van_tr_end = hist["vanilla_ae"]["train_losses"][-1]
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
<title>Laporan Ilmiah Deep Learning (D6421) Sesi 07</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 18mm 17mm 18mm 17mm;
    @top-left {{
      content: "{cfg['mata_kuliah']}";
      font-size: 7.8pt;
      color: #475569;
      font-family: "Times New Roman", Georgia, serif;
      font-style: italic;
    }}
    @top-right {{
      content: "Studi Komparatif Model Generatif";
      font-size: 7.8pt;
      color: #475569;
      font-family: "Times New Roman", Georgia, serif;
      font-style: italic;
    }}
    @bottom-left {{
      content: "{cfg['nama_lengkap']} ({cfg['nim']}, {cfg['kelas']})";
      font-size: 7.8pt;
      color: #475569;
      font-family: "Times New Roman", Georgia, serif;
    }}
    @bottom-right {{
      content: "Halaman " counter(page) " dari " counter(pages);
      font-size: 7.8pt;
      color: #1e293b;
      font-family: "Times New Roman", Georgia, serif;
    }}
  }}
  @page :first {{
    @top-left {{ content: ""; }}
    @top-right {{ content: ""; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Times New Roman", Georgia, serif;
    font-size: 9.8pt;
    line-height: 1.42;
    color: #111827;
    margin: 0;
    padding: 0;
  }}
  .paper-header {{
    text-align: center;
    padding-bottom: 10px;
    border-bottom: 1.5pt solid #111827;
    margin-bottom: 10px;
  }}
  .course-line {{
    font-size: 8.5pt;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #334155;
    margin-bottom: 5px;
  }}
  h1.paper-title {{
    font-size: 13.5pt;
    line-height: 1.28;
    font-weight: 700;
    margin: 0 0 8px 0;
    color: #000000;
  }}
  .author-block {{
    font-size: 9.2pt;
    line-height: 1.38;
    color: #1f2937;
  }}
  .author-name {{
    font-weight: 700;
    font-size: 10pt;
  }}
  .repo-line {{
    font-size: 8.6pt;
    margin-top: 3px;
  }}
  .repo-line a {{
    color: #1e3a8a;
    text-decoration: none;
  }}
  .abstract-block {{
    margin: 8px 14px 10px 14px;
    padding: 7px 10px;
    border-top: 0.6pt solid #94a3b8;
    border-bottom: 0.6pt solid #94a3b8;
    font-size: 9pt;
    line-height: 1.38;
    text-align: justify;
  }}
  .abstract-title {{
    font-weight: 700;
    text-transform: uppercase;
    font-size: 8.4pt;
    letter-spacing: 0.4px;
    margin-bottom: 2px;
  }}
  .keywords {{
    margin-top: 4px;
    font-size: 8.6pt;
  }}
  h2.section-hdr {{
    font-size: 10.8pt;
    font-weight: 700;
    color: #000000;
    border-bottom: 0.9pt solid #334155;
    padding-bottom: 2px;
    margin: 12px 0 6px 0;
    page-break-after: avoid;
  }}
  h3.sub-hdr {{
    font-size: 9.8pt;
    font-weight: 700;
    color: #111827;
    margin: 8px 0 4px 0;
    page-break-after: avoid;
  }}
  p {{
    margin: 0 0 5.5px 0;
    text-align: justify;
  }}
  ol, ul {{
    margin: 2px 0 6px 0;
    padding-left: 18px;
  }}
  li {{
    margin-bottom: 2.5px;
    text-align: justify;
  }}
  .theory-note {{
    margin: 6px 0 8px 0;
    padding: 6px 10px;
    border-left: 2pt solid #334155;
    background: #f8fafc;
    font-size: 9.1pt;
    break-inside: avoid;
  }}
  .theory-note-title {{
    font-weight: 700;
    margin-bottom: 2px;
  }}
  .table-caption {{
    font-size: 8.8pt;
    margin: 7px 0 3px 0;
    text-align: left;
    page-break-after: avoid;
  }}
  .booktabs {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.1pt;
    line-height: 1.3;
    margin: 0 0 8px 0;
    border-top: 1.4pt solid #111827;
    border-bottom: 1.4pt solid #111827;
    break-inside: avoid;
  }}
  .booktabs thead th {{
    font-weight: 700;
    color: #000000;
    padding: 4px 5px;
    border-bottom: 0.8pt solid #111827;
    text-align: left;
    vertical-align: bottom;
  }}
  .booktabs tbody td {{
    padding: 3.3px 5px;
    border-bottom: 0.4pt solid #e2e8f0;
    vertical-align: top;
  }}
  .booktabs tbody tr:last-child td {{
    border-bottom: none;
  }}
  .num {{ text-align: right !important; font-variant-numeric: tabular-nums; }}
  .eq-list {{
    margin: 4px 0 8px 0;
    break-inside: avoid;
  }}
  .eq-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 2.5px 8px;
    font-size: 9.2pt;
  }}
  .eq-body {{
    flex: 1;
    text-align: left;
  }}
  .eq-label {{
    font-weight: 700;
    font-size: 8.6pt;
    color: #334155;
    min-width: 150px;
    display: inline-block;
  }}
  .eq-num {{
    margin-left: 12px;
    color: #334155;
    font-size: 8.8pt;
  }}
  .diagram-grid {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 4px;
    margin: 4px 0 7px 0;
  }}
  .diagram-card {{
    border: 0.6pt solid #cbd5e1;
    padding: 3px 6px;
    background: #ffffff;
    break-inside: avoid;
  }}
  .diagram-title {{
    font-size: 8pt;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 1px;
  }}
  .arch-svg {{
    width: 100%;
    height: auto;
    display: block;
  }}
  .figure-wrap {{
    margin: 7px 0 9px 0;
    text-align: center;
    break-inside: avoid;
  }}
  .figure-wrap img {{
    width: 92%;
    height: auto;
    display: block;
    margin: 0 auto 4px auto;
  }}
  .figure-wrap.compact img {{
    width: 82%;
  }}
  .fig-caption {{
    font-size: 8.6pt;
    color: #1f2937;
    line-height: 1.36;
    text-align: justify;
    margin: 0 4px;
  }}
  .ref-list {{
    font-size: 8.5pt;
    line-height: 1.36;
    padding-left: 18px;
    margin: 4px 0 0 0;
  }}
  .ref-list li {{
    margin-bottom: 3px;
  }}
  code {{
    font-family: "Consolas", monospace;
    font-size: 8.4pt;
    background: #f1f5f9;
    padding: 0 2.5px;
  }}
</style>
</head>
<body>

<div class="paper-header">
  <div class="course-line">{cfg['mata_kuliah']} | Tahun Akademik {cfg['tahun_akademik']}</div>
  <h1 class="paper-title">Studi Komparatif Empiris Model Generatif: Autoencoder (Vanilla &amp; Sparse), VAE, GAN (Vanilla &amp; DCGAN), dan DDPM disertai Pembuktian Batasan Teoretis Model Klasifikasi untuk Image Generation</h1>
  <div class="author-block">
    <div class="author-name">{cfg['nama_lengkap']}</div>
    <div>NIM: {cfg['nim']} &nbsp;|&nbsp; Kelas: {cfg['kelas']} &nbsp;|&nbsp; Dosen Pengampu: {cfg['dosen_pengampu']}</div>
    <div class="repo-line">Repositori Publik GitHub: <a href="{github_url}">{github_url}</a></div>
  </div>
</div>

<div class="abstract-block">
  <div class="abstract-title">Abstrak</div>
  Penelitian ini mengevaluasi secara terstandarisasi enam arsitektur pemodelan generatif dari Sesi 06 dan Sesi 07, yaitu Vanilla Autoencoder, Sparse Autoencoder, Variational Autoencoder (VAE), Vanilla GAN berbasis Multi-Layer Perceptron (MLP), Deep Convolutional GAN (DCGAN), dan Denoising Diffusion Probabilistic Model (DDPM), serta menguji batasan teoretis MLP Classifier sebagai generator citra pada dataset Fashion-MNIST (70.000 citra, pembagian terstratifikasi 80/10/10, <code>seed={ds['seed']}</code>). Eksperimen pembuktian menunjukkan bahwa MLP Classifier dengan akurasi uji {mlp['test_accuracy']*100:.2f}% tetap gagal merekonstruksi dan menyintesis citra baru akibat <em>information bottleneck</em> pemodelan <em>P</em>(<em>y</em>|<em>x</em>), ketidakbalikan bobot non-ortogonal (deviasi <em>W</em><sub>1</sub> sebesar {mlp['orthogonality']['W1_orthogonality_error']:.4f}), pemotongan ireversibel ReLU ({mlp['z_mlp_zero_fraction']*100:.2f}% elemen nol), dan ketiadaan regularisasi prior <em>P</em>(<em>z</em>). Pada evaluasi rekonstruksi Test Set, Vanilla AE mencapai hasil terbaik (MSE {m['vanilla_ae']['mse']:.5f}, PSNR {m['vanilla_ae']['psnr']:.2f} dB, SSIM {m['vanilla_ae']['ssim']:.4f}), sementara Sparse AE memangkas norma L1 laten sebesar {sae_l1_ratio:.2f}× dengan konsekuensi MSE {m['sparse_ae']['mse']:.5f}. Sebaliknya, pada sintesis sampel dari prior acak, DDPM meraih Fréchet Inception Distance (FID) terbaik sebesar {m['ddpm']['fid']:.2f} dengan latensi inferensi {m['ddpm']['latency_ms_mean']:.2f} ms/batch 64, diikuti DCGAN (FID {m['dcgan']['fid']:.2f}, {m['dcgan']['latency_ms_mean']:.2f} ms/batch), VAE (FID {m['vae']['fid']:.2f}, {m['vae']['latency_ms_mean']:.2f} ms/batch), dan Vanilla GAN (FID {m['vanilla_gan']['fid']:.2f}, {m['vanilla_gan']['latency_ms_mean']:.2f} ms/batch).
  <div class="keywords"><strong>Kata Kunci:</strong> <em>Generative Models, Autoencoder, Variational Autoencoder, DCGAN, DDPM, Generative Trilemma, Information Bottleneck, Fashion-MNIST.</em></div>
</div>

<h2 class="section-hdr">Bagian I: Identitas Penulis, Repositori Kode, dan Protokol Standardisasi</h2>

<p>Seluruh implementasi eksperimen disusun secara modular dalam bahasa Python dan PyTorch pada repositori publik <a href="{github_url}">{github_url}</a> yang memuat modul arsitektur (<code>src/models/</code>), skrip pelatihan dan pembuktian (<code>src/experiments/</code>), modul evaluasi metrik dan visualisasi (<code>src/evaluation/</code>), notebook interaktif (<code>notebooks/comparative_study.ipynb</code>), berkas dependensi <code>requirements.txt</code>, dokumentasi <code>README.md</code>, serta riwayat commit bertahap.</p>

<p>Untuk menjamin komparabilitas empiris antarmodel, seluruh pengujian dijalankan di GPU NVIDIA GeForce RTX 4060 Laptop (8 GB VRAM) dengan penetapan <em>random seed</em> deterministik <code>seed={ds['seed']}</code> pada pustaka Python, NumPy, dan PyTorch CUDA. Dataset <strong>Fashion-MNIST</strong> ({ds['name']}, 70.000 citra grayscale 28×28 piksel pada 10 kategori pakaian) digabung dan dibagi secara terstratifikasi menjadi 80% Train ({ds['train_size']:,} citra), 10% Validation ({ds['val_size']:,} citra), dan 10% Test ({ds['test_size']:,} citra) menggunakan <code>batch_size={ds['batch_size']}</code> (<code>shuffle=True</code> pada himpunan latih; <code>shuffle=False</code> pada validasi dan uji). Seluruh metrik kuantitatif dan sampel visual diambil dari Test Set yang belum pernah dilihat model selama pelatihan.</p>

<h2 class="section-hdr">Bagian II: Landasan Teori, Desain Arsitektur, dan Formulasi Matematika</h2>

<p>Perbedaan karakteristik antara rekonstruksi kabur, ketajaman tepi, kestabilan konvergensi, serta biaya komputasi antarmodel bersumber dari interaksi antara bias induktif arsitektur (lapisan <em>fully-connected</em> dibandingkan dengan konvolusi spasial berhierarki) dan bentuk fungsi objektif yang diminimalkan. Rancangan skematik ketujuh model yang diteliti ditunjukkan pada skema arsitektur di bawah ini.</p>

{svgs_html}

<h3 class="sub-hdr">2.1 Konfigurasi Parameter, Jumlah Bobot, dan Hiperparameter</h3>
<div class="table-caption"><strong>Tabel 1.</strong> Spesifikasi arsitektur, jumlah parameter terlatih, dimensi laten, fungsi aktivasi, optimizer, dan epoch pelatihan seluruh model.</div>
<table class="booktabs">
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
      <td>MLP Classifier</td>
      <td>784 → 256 → 64 → 10</td>
      <td class="num">218,058</td>
      <td class="num">64 (<em>z</em><sub>MLP</sub>)</td>
      <td>ReLU, Logits</td>
      <td>Adam (1e-3, Cosine)</td>
      <td class="num">20</td>
    </tr>
    <tr>
      <td>Frozen Decoder</td>
      <td>64 → 256 → 784 (pada <em>z</em><sub>MLP</sub> beku)</td>
      <td class="num">218,128</td>
      <td class="num">64</td>
      <td>ReLU, Sigmoid</td>
      <td>Adam (1e-3)</td>
      <td class="num">30</td>
    </tr>
    <tr>
      <td>Vanilla AE</td>
      <td>Enc: 784→256→64 | Dec: 64→256→784</td>
      <td class="num">435,536</td>
      <td class="num">64 (Linear)</td>
      <td>ReLU, Sigmoid</td>
      <td>Adam (1e-3)</td>
      <td class="num">30</td>
    </tr>
    <tr>
      <td>Sparse AE</td>
      <td>Enc: 784→256→64 | Dec: 64→256→784</td>
      <td class="num">435,536</td>
      <td class="num">64 (ReLU+L1)</td>
      <td>ReLU, Sigmoid</td>
      <td>Adam (1e-3, λ=5e-3)</td>
      <td class="num">30</td>
    </tr>
    <tr>
      <td>VAE</td>
      <td>784→256→(μ:64, logσ²:64)→256→784</td>
      <td class="num">451,984</td>
      <td class="num">64 (Gaussian)</td>
      <td>ReLU, Sigmoid</td>
      <td>Adam (1e-3)</td>
      <td class="num">30</td>
    </tr>
    <tr>
      <td>Vanilla GAN</td>
      <td>G: 64→256→512→784 | D: 784→512→256→1</td>
      <td class="num">1,083,921</td>
      <td class="num">64 (Prior)</td>
      <td>LeakyReLU(0.2), Tanh</td>
      <td>Adam (2e-4, β=(0.5,0.999))</td>
      <td class="num">50</td>
    </tr>
    <tr>
      <td>DCGAN</td>
      <td>G: ConvT(100→256→128→1) | D: Conv(1→64→128→256)</td>
      <td class="num">2,213,377</td>
      <td class="num">100 (Prior)</td>
      <td>ReLU/LReLU, BN, Tanh</td>
      <td>Adam (2e-4, β=(0.5,0.999), ls=0.9)</td>
      <td class="num">50</td>
    </tr>
    <tr>
      <td>DDPM</td>
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
<div class="eq-list">
  <div class="eq-row">
    <div class="eq-body"><span class="eq-label">MLP Classifier (CE):</span> ℒ<sub>CE</sub> = −(1/N) ∑<sub>i=1..N</sub> ∑<sub>c=1..10</sub> y<sub>i,c</sub> log softmax(W<sub>3</sub> z<sub>MLP,i</sub> + b<sub>3</sub>)<sub>c</sub></div>
    <div class="eq-num">(1)</div>
  </div>
  <div class="eq-row">
    <div class="eq-body"><span class="eq-label">Vanilla AE (MSE):</span> ℒ<sub>AE</sub> = (1/784) ‖<em>x</em> − <em>D</em><sub>θ</sub>(<em>E</em><sub>φ</sub>(<em>x</em>))‖<sub>2</sub><sup>2</sup></div>
    <div class="eq-num">(2)</div>
  </div>
  <div class="eq-row">
    <div class="eq-body"><span class="eq-label">Sparse AE (MSE+L1):</span> ℒ<sub>SAE</sub> = (1/784) ‖<em>x</em> − <em>x̂</em>‖<sub>2</sub><sup>2</sup> + λ · (1/64) ∑<sub>j=1..64</sub> |<em>z</em><sub>j</sub>|, &nbsp; λ = 5×10<sup>−3</sup></div>
    <div class="eq-num">(3)</div>
  </div>
  <div class="eq-row">
    <div class="eq-body"><span class="eq-label">VAE (ELBO):</span> ℒ<sub>VAE</sub> = −𝔼<sub><em>q</em><sub>φ</sub>(<em>z</em>|<em>x</em>)</sub>[log <em>p</em><sub>θ</sub>(<em>x</em>|<em>z</em>)] + <em>D</em><sub>KL</sub>(<em>q</em><sub>φ</sub>(<em>z</em>|<em>x</em>) ∥ 𝒩(0, <strong>I</strong>)), &nbsp; <em>z</em> = μ + σ ⊙ ε</div>
    <div class="eq-num">(4)</div>
  </div>
  <div class="eq-row">
    <div class="eq-body"><span class="eq-label">GAN / DCGAN (Minimax):</span> min<sub>G</sub> max<sub>D</sub> <em>V</em>(<em>D</em>,<em>G</em>) = 𝔼<sub><em>x</em>~<em>p</em><sub>data</sub></sub>[log <em>D</em>(<em>x</em>)] + 𝔼<sub><em>z</em>~𝒩(0,<strong>I</strong>)</sub>[log(1 − <em>D</em>(<em>G</em>(<em>z</em>)))]</div>
    <div class="eq-num">(5)</div>
  </div>
  <div class="eq-row">
    <div class="eq-body"><span class="eq-label">DDPM (Noise MSE):</span> ℒ<sub>simple</sub>(θ) = 𝔼<sub><em>t</em>~𝒰(1,<em>T</em>), <em>x</em><sub>0</sub>, ε~𝒩(0,<strong>I</strong>)</sub>[‖ε − ε<sub>θ</sub>(√ᾱ<sub><em>t</em></sub> <em>x</em><sub>0</sub> + √(1 − ᾱ<sub><em>t</em></sub>) ε, <em>t</em>)‖<sub>2</sub><sup>2</sup>], &nbsp; <em>T</em> = 1000</div>
    <div class="eq-num">(6)</div>
  </div>
</div>

<h2 class="section-hdr">Bagian III: Pembuktian Teoretis dan Empiris Batasan MLP Classifier untuk Image Generation</h2>

<p>Meskipun lapisan ekstraktor fitur pada MLP Classifier (<code>784 → 256 → 64</code>) memiliki topologi struktural yang identik dengan bagian Encoder pada Vanilla Autoencoder, matriks bobot terlatih <em>W</em><sup>(<em>l</em>)</sup> serta representasi laten <em>z</em><sub>MLP</sub> ∈ ℝ<sup>64</sup> tidak dapat difungsikan untuk rekonstruksi ataupun sintesis citra karena tiga batasan matematis mendasar:</p>

<ol>
  <li><strong>Information Bottleneck pada Pemodelan <em>P</em>(<em>y</em>|<em>x</em>):</strong> Minimisasi Cross-Entropy Loss (Persamaan 1) berfokus pada probabilitas kondisional kelas <em>P</em>(<em>y</em>|<em>x</em>). Jaringan mempertahankan fitur pembeda antarkelas dan membuang variasi spasial intra-kelas (seperti ketebalan lipatan, tekstur kain, dan orientasi detail pakaian) yang tidak memengaruhi batas keputusan klasifikasi.</li>
  <li><strong>Ketidakbalikan Matriks Bobot Non-Ortogonal dan ReLU:</strong> Matriks bobot persegi panjang <em>W</em><sup>(<em>l</em>)</sup> tidak bersifat ortogonal ((<em>W</em><sup>(<em>l</em>)</sup>)<sup>T</sup> <em>W</em><sup>(<em>l</em>)</sup> ≠ <strong>I</strong>), sehingga operasi transpos <em>W</em><sup>T</sup> bukan merupakan invers <em>W</em><sup>−1</sup>. Pengukuran langsung pada bobot terlatih menghasilkan galat ortogonalitas relatif ‖<em>W</em><sup>T</sup><em>W</em> − <strong>I</strong>‖<sub>F</sub> / ‖<strong>I</strong>‖<sub>F</sub> sebesar {mlp['orthogonality']['W1_orthogonality_error']:.4f} (<em>W</em><sub>1</sub>), {mlp['orthogonality']['W2_orthogonality_error']:.4f} (<em>W</em><sub>2</sub>), dan {mlp['orthogonality']['W3_orthogonality_error']:.4f} (<em>W</em><sub>3</sub>). Selain itu, aktivasi ReLU(<em>u</em>) = max(0, <em>u</em>) memotong nilai negatif secara ireversibel dan menyebabkan {mlp['z_mlp_zero_fraction']*100:.2f}% komponen <em>z</em><sub>MLP</sub> pada Test Set bernilai nol.</li>
  <li><strong>Ketiadaan Regularisasi Kerapatan Prior <em>P</em>(<em>z</em>):</strong> Ruang fitur <em>z</em><sub>MLP</sub> tidak diregulasi agar mengikuti distribusi prior kontinu seperti 𝒩(0, <strong>I</strong>). Pada Test Set, vektor rata-rata fitur memiliki norma ‖μ<sub><em>z</em></sub>‖<sub>2</sub> = {mlp['z_mlp_mu_norm']:.2f} dengan rata-rata simpangan baku σ̄<sub><em>z</em></sub> = {mlp['z_mlp_sigma_mean']:.2f} serta distribusi miring akibat aktivasi ReLU.</li>
</ol>

<div class="theory-note">
  <div class="theory-note-title">Implikasi Teori Informasi: <em>Data Processing Inequality</em> pada Representasi Diskriminatif</div>
  Pada rantai Markov pelatihan terawasi <em>X</em> → <em>H</em><sub>1</sub> → <em>Z</em><sub>MLP</sub> → <em>Y</em>, berlaku ketaksamaan <em>I</em>(<em>Z</em><sub>MLP</sub>; <em>Y</em>) ≤ <em>H</em>(<em>Y</em>) ≤ log<sub>2</sub>(10) ≈ 3.32 bit per citra. Karena ℒ<sub>CE</sub> hanya memberi gradien untuk memisahkan 10 label diskret, entropi kondisional <em>H</em>(<em>X</em>|<em>Y</em>) yang merepresentasikan variasi lokal di dalam masing-masing kelas terbuang sepanjang ekstraksi fitur. Sebaliknya, Autoencoder melatih pemetaan yang memaksimalkan <em>I</em>(<em>X</em>; <em>Z</em><sub>AE</sub>) agar sinyal masukan <em>X</em> tetap dapat dipulihkan oleh dekoder.
</div>

<div class="table-caption"><strong>Tabel 2.</strong> Hasil kuantitatif lima tahap eksperimen pembuktian batasan MLP Classifier pada Test Set Fashion-MNIST.</div>
<table class="booktabs">
  <thead>
    <tr>
      <th>Tahap Eksperimen Bagian 2.1 (Test Set)</th>
      <th class="num">Test MSE</th>
      <th class="num">Test PSNR (dB)</th>
      <th class="num">Test SSIM</th>
      <th>Temuan Empiris Utama</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Tahap 1: MLP Classifier (784→256→64→10)</td>
      <td class="num">-</td><td class="num">-</td><td class="num">-</td>
      <td>Test Acc = {mlp['test_accuracy']*100:.2f}% (Val = {mlp['val_accuracy']*100:.2f}%, memenuhi syarat ≥ 88%)</td>
    </tr>
    <tr>
      <td>Tahap 2: Proyeksi Balik Bobot <em>W</em><sup>T</sup> dari One-Hot <em>y</em><sub><em>c</em></sub></td>
      <td class="num">{mlp['transpose_onehot_metrics']['mse']:.5f}</td>
      <td class="num">{mlp['transpose_onehot_metrics']['psnr']:.2f}</td>
      <td class="num">{mlp['transpose_onehot_metrics']['ssim']:.4f}</td>
      <td>Bercak abu-abu seragam akibat superposisi template bobot kasar (<em>W</em><sup>T</sup> ≠ <em>W</em><sup>−1</sup>)</td>
    </tr>
    <tr>
      <td>Tahap 3: Proyeksi Balik <em>W</em><sup>T</sup> dari Latent <em>z</em><sub>MLP</sub></td>
      <td class="num">{mlp['transpose_latent_metrics']['mse']:.5f}</td>
      <td class="num">{mlp['transpose_latent_metrics']['psnr']:.2f}</td>
      <td class="num">{mlp['transpose_latent_metrics']['ssim']:.4f}</td>
      <td>Artefak kontras biner acak; bobot maju tidak dapat dibalik menjadi dekoder</td>
    </tr>
    <tr>
      <td>Tahap 4a: Frozen Latent Decoder (<em>z</em><sub>MLP</sub> ∈ ℝ<sup>64</sup>)</td>
      <td class="num">{mlp['frozen_decoder_metrics']['mse']:.5f}</td>
      <td class="num">{mlp['frozen_decoder_metrics']['psnr']:.2f}</td>
      <td class="num">{mlp['frozen_decoder_metrics']['ssim']:.4f}</td>
      <td>Siluet kelas pulih secara kasar, tetapi detail intra-kelas hilang ({mse_ratio:.2f}× MSE Vanilla AE)</td>
    </tr>
    <tr>
      <td>Tahap 4b (Pembanding): Vanilla AE (<em>z</em><sub>AE</sub> ∈ ℝ<sup>64</sup>)</td>
      <td class="num">{m['vanilla_ae']['mse']:.5f}</td>
      <td class="num">{m['vanilla_ae']['psnr']:.2f}</td>
      <td class="num">{m['vanilla_ae']['ssim']:.4f}</td>
      <td>Mempertahankan variasi spasial intra-kelas tanpa bottleneck diskriminatif</td>
    </tr>
    <tr>
      <td>Tahap 5: Prior Sampling <em>z</em><sub>rand</sub> ~ 𝒩(μ<sub><em>z</em></sub>, diag(σ<sub><em>z</em></sub><sup>2</sup>))</td>
      <td class="num">-</td><td class="num">-</td><td class="num">-</td>
      <td>Menghasilkan pecahan bercak tak berstruktur akibat ketiadaan manifold kontinu</td>
    </tr>
  </tbody>
</table>

<h2 class="section-hdr">Bagian IV: Komparasi Metrik Numerik Terstandarisasi</h2>

<p>Evaluasi kuantitatif pada Test Set (7.000 citra) dirangkum pada Tabel 3. Nilai Test MSE, Test PSNR, dan Test SSIM dihitung pada rentang piksel [0, 1]. Skor FID dihitung melalui ekstraksi fitur pool3 (2048 dimensi) dari InceptionV3 terlatih dengan membandingkan 7.000 sampel sintetis terhadap seluruh 7.000 citra Test Set. Latensi inferensi diukur untuk satu batch berisi 64 sampel pada GPU menggunakan sinkronisasi eksplisit <code>torch.cuda.synchronize()</code>.</p>

<div class="table-caption"><strong>Tabel 3.</strong> Komparasi metrik kuantitatif seluruh enam arsitektur generatif pada Test Set Fashion-MNIST.</div>
<table class="booktabs">
  <thead>
    <tr>
      <th>Arsitektur Model</th>
      <th>Dim Laten</th>
      <th class="num">Epoch Time (s)</th>
      <th class="num">Latency (ms/b64)</th>
      <th class="num">Test MSE</th>
      <th class="num">PSNR (dB)</th>
      <th class="num">SSIM</th>
      <th class="num">FID Score</th>
      <th>Kestabilan Training</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Vanilla Autoencoder</td>
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
      <td>Sparse Autoencoder</td>
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
      <td>Variational AE (VAE)</td>
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
      <td>Vanilla GAN (MLP)</td>
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
      <td>DCGAN</td>
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
      <td>DDPM (UNet28 + EMA)</td>
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

<p>Pada keluarga Autoencoder, penambahan penalti sparsity λ‖<em>z</em>‖<sub>1</sub> (λ = 5×10<sup>−3</sup>) menurunkan rata-rata norma L1 representasi laten dari {m['vanilla_ae']['latent_l1_mean']:.2f} pada Vanilla AE (dengan {m['vanilla_ae']['latent_zero_frac']*100:.2f}% unit mendekati nol) menjadi {m['sparse_ae']['latent_l1_mean']:.2f} pada Sparse AE dengan {m['sparse_ae']['latent_zero_frac']*100:.2f}% unit non-aktif (|<em>z</em>| &lt; 10<sup>−3</sup>). Pada evaluasi kualitas generatif, DDPM mencatat skor FID terendah ({m['ddpm']['fid']:.2f}), diikuti oleh DCGAN ({m['dcgan']['fid']:.2f}), sedangkan VAE ({m['vae']['fid']:.2f}) dan Vanilla GAN ({m['vanilla_gan']['fid']:.2f}) menghasilkan FID yang jauh lebih tinggi akibat keburaman posterior Gaussian serta ketiadaan bias induktif konvolusional.</p>

<h2 class="section-hdr">Bagian V: Visualisasi Komparatif Utama dan Analisis Kualitatif</h2>

<div class="figure-wrap compact">
  <img src="{fig1_uri}" alt="Figur 1 Rekonstruksi Side-by-Side" />
  <div class="fig-caption"><strong>Figur 1. Grid Rekonstruksi Side-by-Side pada Test Set.</strong> Perbandingan delapan kategori Fashion-MNIST antara citra asli dan hasil rekonstruksi Vanilla AE (MSE {m['vanilla_ae']['mse']:.5f}), Sparse AE ({m['sparse_ae']['mse']:.5f}), serta VAE ({m['vae']['mse']:.5f}). Vanilla AE mempertahankan detail lokal paling jelas (misalnya corak kotak pada <em>Shirt</em> dan tekstur <em>Coat</em>), Sparse AE cenderung menghaluskan tepi setelah memangkas {m['sparse_ae']['latent_zero_frac']*100:.2f}% aktivasi laten, sedangkan VAE tampak paling kabur karena regularisasi KL meratakan variasi berfrekuensi tinggi.</div>
</div>

<div class="figure-wrap compact">
  <img src="{fig2_uri}" alt="Figur 2 Random Generative Sampling 4x4" />
  <div class="fig-caption"><strong>Figur 2. Grid Random Generative Sampling (Kisi 4×4 per Model).</strong> Sampel acak dari prior 𝒩(0, <strong>I</strong>). Panel A (VAE, FID {m['vae']['fid']:.2f}) membentuk struktur global yang dapat dikenali namun bertepi kabur. Panel B (Vanilla GAN, FID {m['vanilla_gan']['fid']:.2f}) menghasilkan kontras tajam yang disertai bintik derau piksel karena generator MLP tidak memiliki lokalitas spasial. Panel C (DCGAN, FID {m['dcgan']['fid']:.2f}) menajamkan siluet melalui <em>strided transposed convolution</em> dan BatchNorm. Panel D (DDPM, FID {m['ddpm']['fid']:.2f}) menghasilkan kontur paling bersih, gradasi kain halus, serta keragaman kelas merata.</div>
</div>

<div class="figure-wrap">
  <img src="{fig3_uri}" alt="Figur 3 Bukti Empiris MLP Classifier" />
  <div class="fig-caption"><strong>Figur 3. Bukti Empiris Batasan Bobot dan Latent Features MLP Classifier.</strong> (1) Citra uji asli. (2) Proyeksi balik langsung <em>W</em><sup>T</sup> dari vektor kelas <em>y</em><sub><em>c</em></sub> menghasilkan bercak abu-abu seragam antarkelas, sedangkan proyeksi dari <em>z</em><sub>MLP</sub> memunculkan pola biner terdistorsi. (3) Frozen Latent Decoder hanya memulihkan siluet umum kelas (PSNR {mlp['frozen_decoder_metrics']['psnr']:.2f} dB dibandingkan dengan Vanilla AE {m['vanilla_ae']['psnr']:.2f} dB; corak <em>Shirt</em> paling kanan hilang menjadi kemeja polos). (4) Sampling prior <em>z</em><sub>rand</sub> dari statistik <em>z</em><sub>MLP</sub> menghasilkan bentuk pecah yang tidak membentuk objek utuh.</div>
</div>

<div class="figure-wrap compact">
  <img src="{fig4_uri}" alt="Figur 4 Interpolasi Ruang Laten Linier" />
  <div class="fig-caption"><strong>Figur 4. Interpolasi Ruang Laten Linier <em>z</em>(α) = (1−α)<em>z</em><sub>A</sub> + α<em>z</em><sub>B</sub> (α ∈ [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).</strong> Pada Vanilla AE, Sparse AE, dan VAE, interpolasi menggunakan pasangan uji <em>Sneaker</em> (α=0) dan <em>Dress</em> (α=1); pada GAN dan DCGAN, interpolasi menggunakan dua vektor prior acak. Vanilla AE dan Sparse AE melintasi wilayah kosong (<em>latent space voids</em>) pada α=0.4 sampai 0.6 sehingga memunculkan bayangan ganda. VAE bertransisi halus tanpa gambar pecah, Vanilla GAN tetap berderau sepanjang lintasan, dan DCGAN bertransformasi secara tajam dari atasan berlengan panjang terang menjadi kaus berlengan pendek gelap.</div>
</div>

<div class="figure-wrap">
  <img src="{fig5_uri}" alt="Figur 5 Kurva Training Loss dan DDPM Denoising Trajectory" />
  <div class="fig-caption"><strong>Figur 5. Kurva Training Loss Keenam Model dan Trajektori Reverse Denoising DDPM.</strong> Vanilla AE, Sparse AE, dan VAE turun secara mulus dengan selisih train dan validation yang tipis. Pada Vanilla GAN, loss Generator melonjak hingga {vgan_g_max:.2f} pada epoch {vgan_g_max_ep} sebelum turun stabil, sedangkan pada DCGAN Discriminator bertahan di sekitar {dc_d_end:.2f} (dengan G sebesar {dc_g_end:.2f}) dan menghasilkan fitur konvolusional tajam. DDPM turun secara monoton pada Noise MSE ({ddpm_start:.4f} menjadi {ddpm_end:.4f}); panel bawah menunjukkan perubahan bertahap dari derau Gaussian pada <em>t</em>=999 menuju bentuk sepatu dan gaun bersih pada <em>t</em>=0.</div>
</div>

<h2 class="section-hdr">Bagian VI: Pembahasan Kritis, Jawaban Pertanyaan Penelitian, dan Rekomendasi Industri</h2>

<h3 class="sub-hdr">6.1 Perbandingan Kuantitatif: Reconstruction Fidelity vs. Generative Quality (Q1)</h3>
<p>Pada rekonstruksi Test Set, Vanilla AE mencapai fidelitas tertinggi (MSE = {m['vanilla_ae']['mse']:.5f}, PSNR = {m['vanilla_ae']['psnr']:.2f} dB, SSIM = {m['vanilla_ae']['ssim']:.4f}), diikuti Sparse AE (MSE = {m['sparse_ae']['mse']:.5f}, PSNR = {m['sparse_ae']['psnr']:.2f} dB, SSIM = {m['sparse_ae']['ssim']:.4f}) dan VAE (MSE = {m['vae']['mse']:.5f}, PSNR = {m['vae']['psnr']:.2f} dB, SSIM = {m['vae']['ssim']:.4f}). Penambahan penalti L1 λ‖<em>z</em>‖<sub>1</sub> (λ = 5×10<sup>−3</sup>) memaksa {m['sparse_ae']['latent_zero_frac']*100:.2f}% dimensi laten Sparse AE bernilai mendekati nol dan memangkas norma L1 sebesar {sae_l1_ratio:.2f}× ({m['sparse_ae']['latent_l1_mean']:.2f} dibandingkan dengan {m['vanilla_ae']['latent_l1_mean']:.2f}). Penyempitan kapasitas efektif ruang laten tersebut menaikkan galat MSE sebesar +{sae_mse_diff:.5f} ({sae_psnr_diff:.2f} dB PSNR) relatif terhadap representasi padat Vanilla AE.</p>
<p>Pada sintesis citra baru dari prior acak 𝒩(0, <strong>I</strong>), peringkat FID dari yang terbaik adalah DDPM ({m['ddpm']['fid']:.2f}), DCGAN ({m['dcgan']['fid']:.2f}), VAE ({m['vae']['fid']:.2f}), dan Vanilla GAN ({m['vanilla_gan']['fid']:.2f}). Perbedaan peringkat antara tugas rekonstruksi dan generasi tersebut menunjukkan bahwa kemampuan kompresi-rekonstruksi deterministik tidak memadai untuk sintesis prior: Vanilla AE dan Sparse AE tidak memiliki distribusi prior teratur yang dapat disampel, sedangkan VAE menukar ketajaman tepi dengan kelancaran manifold ELBO.</p>

<h3 class="sub-hdr">6.2 Analisis Blurriness vs. Sharp Edges dan Struktur Representasi Fitur (Q2)</h3>
<p>Efek keburaman pada Vanilla AE, Sparse AE, dan VAE bersumber dari sifat <em>mode-averaging</em> pada minimisasi <em>pixel-wise loss</em> (MSE atau BCE). Secara analitik, penaksir <em>x̂</em> yang meminimalkan 𝔼[‖<em>x</em> − <em>x̂</em>‖<sub>2</sub><sup>2</sup> | <em>z</em>] adalah ekspektasi kondisional 𝔼[<em>x</em>|<em>z</em>]. Apabila beberapa variasi tepi atau tekstur berpeluang muncul di sekitar kode laten <em>z</em> yang sama, perataan piksel meredam komponen frekuensi tinggi sehingga tepi pakaian tampak kabur. Pada VAE, suku divergensi KL memperbesar tumpang-tindih antar-posterior dan mendorong perilaku <em>zero-avoiding</em> yang merentangkan kerapatan di antara mode, sehingga keburaman semakin menonjol.</p>
<p>Sebaliknya, objektif minimax pada GAN dan DCGAN bersifat <em>mode-seeking</em> karena ekuivalen dengan minimisasi divergensi Jensen-Shannon <em>D</em><sub>JS</sub>(<em>P</em><sub>data</sub> ∥ <em>P</em><sub><em>g</em></sub>) ketika Discriminator mencapai titik optimal. Sampel kabur berada di wilayah kerapatan rendah di antara mode data riil sehingga langsung dikenali sebagai citra palsu oleh Discriminator, memaksa Generator memproduksi kontur tegas. Pada DDPM, regresi ε<sub>θ</sub>(<em>x</em><sub><em>t</em></sub>, <em>t</em>) mengestimasi fungsi skor ∇<sub><em>x</em></sub> log <em>p</em><sub><em>t</em></sub>(<em>x</em>) di berbagai skala derau, memungkinkan rekonstruksi detail tajam secara bertahap tanpa perataan maupun kolaps mode. Di sisi lain, penalti L1 mengubah representasi hidden layer Sparse AE menjadi aktivasi selektif terlokalisasi ({m['sparse_ae']['latent_zero_frac']*100:.2f}% unit nol), berbeda dari representasi padat terdistribusi pada Vanilla AE.</p>

<h3 class="sub-hdr">6.3 Stabilitas Training, Dinamika Arsitektur, dan Verifikasi Generative Trilemma (Q3)</h3>
<p>Pada Vanilla GAN berbasis MLP, ketiadaan <em>weight sharing</em> spasial memudahkan Discriminator mengeksploitasi ketidakkonsistenan piksel lokal sehingga kurva loss Generator melonjak ke {vgan_g_max:.2f} pada epoch {vgan_g_max_ep} sebelum turun ke {vgan_g_end:.2f}. DCGAN memitigasi ketidakstabilan tersebut melalui arsitektur konvolusional penuh tanpa lapisan pooling maupun lapisan fully-connected tersembunyi, disertai <code>BatchNorm2d</code> untuk menstabilkan skala gradien, aktivasi <code>LeakyReLU</code>, serta inisialisasi bobot 𝒩(0, 0.02). Perubahan arsitektur tersebut menurunkan skor FID dari {m['vanilla_gan']['fid']:.2f} menjadi {m['dcgan']['fid']:.2f}, meskipun optimasi minimax tetap sensitif terhadap ketidakseimbangan D dan G (D={dc_d_end:.2f}, G={dc_g_end:.2f}).</p>
<p>Berbeda dari optimasi adversarial yang non-stasioner, DDPM menggunakan regresi <em>supervised noise prediction</em> ‖ε − ε<sub>θ</sub>‖<sub>2</sub><sup>2</sup> dengan target stasioner sehingga kurva loss turun secara monoton ({ddpm_start:.4f} menjadi {ddpm_end:.4f} dalam 50 epoch). Pengukuran latensi per batch 64 citra mengonfirmasi trade-off Generative Trilemma sebagaimana dirangkum pada Tabel 4: VAE tercepat ({m['vae']['latency_ms_mean']:.2f} ms/batch) dengan cakupan mode luas tetapi citra kabur (FID {m['vae']['fid']:.2f}); DCGAN menyeimbangkan ketajaman (FID {m['dcgan']['fid']:.2f}) dan kecepatan single-pass ({m['dcgan']['latency_ms_mean']:.2f} ms/batch) dengan sensitivitas arsitektur; sedangkan DDPM mencapai kualitas terbaik (FID {m['ddpm']['fid']:.2f}) dan cakupan mode penuh tetapi membutuhkan {m['ddpm']['latency_ms_mean']:.2f} ms/batch (~{ddpm_vs_dcgan_lat:,.0f}× lebih lambat dibandingkan DCGAN dan ~{ddpm_vs_vae_lat:,.0f}× lebih lambat dibandingkan VAE) akibat iterasi denoising <em>T</em>=1000 langkah.</p>

<div class="table-caption"><strong>Tabel 4.</strong> Matriks verifikasi empiris Generative Trilemma pada keluarga model generatif.</div>
<table class="booktabs">
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
    <tr><td>VAE</td><td>Rendah (buram, {m['vae']['fid']:.2f})</td><td>Sangat Cepat ({m['vae']['latency_ms_mean']:.2f} ms)</td><td>Tinggi (ELBO covering)</td><td>Tinggi (stasioner)</td></tr>
    <tr><td>Vanilla GAN</td><td>Rendah (derau MLP, {m['vanilla_gan']['fid']:.2f})</td><td>Sangat Cepat ({m['vanilla_gan']['latency_ms_mean']:.2f} ms)</td><td>Rendah-Sedang</td><td>Rendah (osilasi awal)</td></tr>
    <tr><td>DCGAN</td><td>Tinggi (tajam, {m['dcgan']['fid']:.2f})</td><td>Cepat ({m['dcgan']['latency_ms_mean']:.2f} ms)</td><td>Sedang (sensitif)</td><td>Sedang (minimax)</td></tr>
    <tr><td>DDPM</td><td>Tertinggi ({m['ddpm']['fid']:.2f})</td><td>Lambat ({m['ddpm']['latency_ms_mean']:.2f} ms)</td><td>Tinggi (penuh)</td><td>Tertinggi (noise MSE)</td></tr>
  </tbody>
</table>

<h3 class="sub-hdr">6.4 Struktur Latent Space dan Fenomena Interpolasi (Q4)</h3>
<p>Pada eksperimen interpolasi linier <em>z</em>(α) = (1−α)<em>z</em><sub>A</sub> + α<em>z</em><sub>B</sub> (Figur 4), Vanilla AE dan Sparse AE memperlihatkan artefak tumpang-tindih bayangan pada α=0.4 dan 0.6. Fenomena ini terjadi karena autoencoder deterministik memetakan data ke klaster terisolasi tanpa regulasi kepadatan prior, sehingga menyisakan wilayah kosong (<em>latent space voids</em>) yang belum pernah dilatih oleh Decoder. Sebaliknya, penalti <em>D</em><sub>KL</sub>(<em>q</em>(<em>z</em>|<em>x</em>) ∥ 𝒩(0, <strong>I</strong>)) pada VAE menarik posterior setiap citra menuju pusat asal, membentuk manifold kontinu yang menghasilkan morfing bentuk mulus dari <em>Sneaker</em> ke <em>Dress</em>. Pada DCGAN, pengambilan dari prior Gaussian 𝒩(0, <strong>I</strong>) yang dipetakan oleh lapisan konvolusi menjaga ketajaman objek di setiap nilai α, sedangkan Vanilla GAN tetap memunculkan derau piksel sepanjang lintasan.</p>

<h3 class="sub-hdr">6.5 Sintesis Pembuktian Batasan MLP Classifier untuk Image Generation (Q5)</h3>
<p>Capaian akurasi klasifikasi sebesar {mlp['test_accuracy']*100:.2f}% pada Test Set tidak menghasilkan bobot ataupun fitur laten yang mampu merekonstruksi citra. Pemodelan <em>P</em>(<em>y</em>|<em>x</em>) bertindak sebagai <em>information bottleneck</em> yang menghilangkan variasi intra-kelas, terbukti dari naiknya Test MSE Frozen Latent Decoder menjadi {mlp['frozen_decoder_metrics']['mse']:.5f} dibandingkan dengan {m['vanilla_ae']['mse']:.5f} pada Vanilla AE berdimensi 64. Selanjutnya, proyeksi balik langsung menggunakan transpos <em>W</em><sup>T</sup> gagal merekonstruksi citra (MSE {mlp['transpose_onehot_metrics']['mse']:.4f} dari <em>y</em><sub><em>c</em></sub> dan {mlp['transpose_latent_metrics']['mse']:.4f} dari <em>z</em><sub>MLP</sub>) akibat matriks bobot non-ortogonal (deviasi {mlp['orthogonality']['W1_orthogonality_error']:.2f} pada <em>W</em><sub>1</sub>) serta pemotongan ireversibel ReLU ({mlp['z_mlp_zero_fraction']*100:.2f}% nol). Tanpa regularisasi <em>P</em>(<em>z</em>), sebaran <em>z</em><sub>MLP</sub> terpusat jauh dari distribusi Gaussian baku (‖μ<sub><em>z</em></sub>‖ = {mlp['z_mlp_mu_norm']:.2f}), sehingga sampling acak menghasilkan citra yang pecah.</p>

<h3 class="sub-hdr">6.6 Sintesis Rekayasa dan Rekomendasi Deployment Industri (Q6)</h3>
<ol>
  <li><strong>Deteksi Anomali Tanpa Supervisi pada Inspeksi Manufaktur (Vanilla AE vs. Sparse AE):</strong> Direkomendasikan <strong>Sparse Autoencoder</strong> (435,536 parameter, {m['sparse_ae']['latency_ms_mean']:.2f} ms/batch 64). Meskipun Vanilla AE memiliki Test MSE lebih rendah ({m['vanilla_ae']['mse']:.5f} dibandingkan dengan {m['sparse_ae']['mse']:.5f}), representasi padatnya berisiko bertindak mendekati pemetaan identitas yang tetap merekonstruksi cacat kecil. Penalti L1 memaksa Sparse AE hanya menyimpan basis fitur normal dominan ({m['sparse_ae']['latent_zero_frac']*100:.2f}% unit nol, norma L1 turun {sae_l1_ratio:.2f}×), sehingga pola cacat asing yang menyimpang dari basis tersebut memicu lonjakan galat rekonstruksi yang lebih diskriminatif.</li>
  <li><strong>Filter Avatar Interaktif Waktu-Nyata pada Perangkat Mobile (&lt;30 ms) (Vanilla GAN vs. DCGAN vs. VAE):</strong> Direkomendasikan <strong>DCGAN</strong> apabila pipeline hanya membutuhkan pembangkitan fitur tajam tanpa modul encoder, atau <strong>VAE</strong> apabila aplikasi wajib melakukan <em>encode-edit-decode</em> langsung dari citra kamera. Generator DCGAN (1,781,504 parameter) mencatat latensi {m['dcgan']['latency_ms_mean']:.2f} ms per batch 64 (~{m['dcgan']['latency_ms_mean']/64:.3f} ms/citra di GPU) dengan FID {m['dcgan']['fid']:.2f}, jauh di bawah batas 30 ms sekaligus jauh lebih tajam dibandingkan dengan VAE (451,984 parameter, {m['vae']['latency_ms_mean']:.2f} ms/b64, FID {m['vae']['fid']:.2f}) dan Vanilla GAN (Generator 550,416 parameter, FID {m['vanilla_gan']['fid']:.2f}). Untuk efisiensi CPU/NPU ponsel, blok konvolusi DCGAN dapat dikonversi menggunakan <em>depthwise separable convolution</em>.</li>
  <li><strong>Pembuatan Aset Visual Beresolusi dan Berfidelitas Tinggi untuk Industri Kreatif (DCGAN vs. DDPM):</strong> Direkomendasikan <strong>DDPM</strong> (UNet28 4,577,665 parameter) yang dipadukan dengan akselerator sampling seperti DDIM atau DPM-Solver pada tahap produksi. Secara empiris, DDPM meraih fidelitas terbaik (FID {m['ddpm']['fid']:.2f} dibandingkan dengan {m['dcgan']['fid']:.2f} pada DCGAN), pelatihan paling stabil tanpa risiko <em>mode collapse</em>, serta tekstur kain paling bersih (Figur 2D dan 5). Untuk pembuatan aset kreatif non-real-time, waktu inferensi {m['ddpm']['latency_ms_mean']/1000:.2f} detik per 64 sampel pada 1.000 langkah masih memadai dan dapat dipangkas 20× (menjadi sekitar {(m['ddpm']['latency_ms_mean']/1000)/20:.2f} detik/batch) melalui sampler deterministik DDIM 50 langkah.</li>
</ol>

<h3 class="sub-hdr">6.7 Kesimpulan</h3>
<p>Studi komparatif pada Fashion-MNIST membuktikan keterkaitan langsung antara formulasi fungsi objektif, bias arsitektur, dan performa empiris model generatif. Model diskriminatif tidak dapat digunakan sebagai generator karena <em>information bottleneck</em> pemodelan <em>P</em>(<em>y</em>|<em>x</em>) serta matriks bobot non-ortogonal. Di antara model generatif, Autoencoder deterministik unggul pada rekonstruksi tetapi memiliki <em>latent space voids</em>; VAE membentuk manifold kontinu dengan konsekuensi keburaman ekspektasi kondisional; DCGAN menjembatani ketajaman kontur dan kecepatan inferensi satu-langkah; dan DDPM memberikan fidelitas serta kestabilan tertinggi dengan kompensasi biaya iterasi sampling.</p>

<h2 class="section-hdr">Daftar Pustaka</h2>
<ol class="ref-list">
  <li>Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., &amp; Bengio, Y. (2014). Generative Adversarial Nets. <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, 27.</li>
  <li>Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., &amp; Hochreiter, S. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, 30.</li>
  <li>Ho, J., Jain, A., &amp; Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, 33, 6840-6851.</li>
  <li>Kingma, D. P., &amp; Welling, M. (2014). Auto-Encoding Variational Bayes. <em>International Conference on Learning Representations (ICLR)</em>.</li>
  <li>Radford, A., Metz, L., &amp; Chintala, S. (2016). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. <em>International Conference on Learning Representations (ICLR)</em>.</li>
  <li>Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., &amp; Chen, X. (2016). Improved Techniques for Training GANs. <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, 29.</li>
  <li>Wang, Z., Bovik, A. C., Sheikh, H. R., &amp; Simoncelli, E. P. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. <em>IEEE Transactions on Image Processing</em>, 13(4), 600-612.</li>
  <li>Xiao, H., Rasul, K., &amp; Vollgraf, R. (2017). Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms. <em>arXiv preprint arXiv:1708.07747</em>.</li>
</ol>

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
