"""Inline SVG schematic architecture diagrams for Bagian II of the scientific report."""


def _box(x: int, y: int, w: int, h: int, title: str, sub: str, fill: str = "#eef4ff", stroke: str = "#2563eb") -> str:
    cx = x + w // 2
    ty = y + h // 2 - (4 if sub else -4)
    sub_html = (
        f'<text x="{cx}" y="{y + h // 2 + 11}" text-anchor="middle" font-size="8.5" fill="#475569">{sub}</text>'
        if sub
        else ""
    )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>'
        f'<text x="{cx}" y="{ty}" text-anchor="middle" font-size="9.5" font-weight="700" fill="#0f172a">{title}</text>'
        f"{sub_html}"
    )


def _arrow(x1: int, y1: int, x2: int, y2: int, label: str = "") -> str:
    line = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#475569" stroke-width="1.3" marker-end="url(#arr)"/>'
    if label:
        mx = (x1 + x2) // 2
        my = (y1 + y2) // 2 - 5
        line += f'<text x="{mx}" y="{my}" text-anchor="middle" font-size="8" fill="#64748b">{label}</text>'
    return line


def render_all_architecture_svgs() -> str:
    """Return responsive grid of clean SVG schematic diagrams for all 7 models."""
    defs = (
        '<defs><marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6 Z" fill="#475569"/></marker></defs>'
    )

    # 1. MLP Classifier + Frozen Decoder
    svg_mlp = (
        f'<svg viewBox="0 0 660 88" class="arch-svg">{defs}'
        + _box(6, 20, 82, 42, "Input x", "784 (28x28)", "#f8fafc", "#64748b")
        + _arrow(88, 41, 114, 41, "W1")
        + _box(114, 20, 96, 42, "Hidden 1", "256, ReLU", "#eef4ff", "#2563eb")
        + _arrow(210, 41, 236, 41, "W2")
        + _box(236, 20, 96, 42, "z_MLP", "64, ReLU", "#fef3c7", "#d97706")
        + _arrow(332, 41, 358, 41, "W3")
        + _box(358, 20, 98, 42, "Logits y", "10 Kelas (CE)", "#fee2e2", "#dc2626")
        + _arrow(284, 62, 284, 72)
        + _box(476, 20, 178, 42, "Frozen Decoder (Tahap 4)", "64 -> 256 -> 784 (MSE)", "#ecfdf5", "#059669")
        + _arrow(332, 41, 476, 41)
        + '</svg>'
    )

    # Fix clean non-overlapping MLP layout
    svg_mlp = (
        f'<svg viewBox="0 0 680 96" class="arch-svg">{defs}'
        + _box(8, 12, 84, 38, "Input x", "784 (28x28)", "#f8fafc", "#64748b")
        + _arrow(92, 31, 122, 31, "W1")
        + _box(122, 12, 92, 38, "Hidden 1", "256, ReLU", "#eef4ff", "#2563eb")
        + _arrow(214, 31, 244, 31, "W2")
        + _box(244, 12, 94, 38, "z_MLP", "64, ReLU", "#fef3c7", "#d97706")
        + _arrow(338, 31, 372, 31, "W3")
        + _box(372, 12, 108, 38, "Klasifikasi", "10 Kelas (CE)", "#fee2e2", "#dc2626")
        + _arrow(291, 50, 291, 68)
        + _box(205, 68, 210, 24, "Frozen Decoder: 64 -> 256 -> 784 (MSE)", "", "#ecfdf5", "#059669")
        + '</svg>'
    )

    # 2. Vanilla AE & Sparse AE
    svg_ae = (
        f'<svg viewBox="0 0 680 70" class="arch-svg">{defs}'
        + _box(6, 14, 84, 42, "Input x", "784", "#f8fafc", "#64748b")
        + _arrow(90, 35, 114, 35)
        + _box(114, 14, 96, 42, "Enc Hidden", "256, ReLU", "#eef4ff", "#2563eb")
        + _arrow(210, 35, 234, 35)
        + _box(234, 14, 120, 42, "Latent z (64)", "Linear (AE) / ReLU+L1", "#fef3c7", "#d97706")
        + _arrow(354, 35, 378, 35)
        + _box(378, 14, 96, 42, "Dec Hidden", "256, ReLU", "#ecfdf5", "#059669")
        + _arrow(474, 35, 498, 35)
        + _box(498, 14, 102, 42, "Output x_hat", "784, Sigmoid", "#f8fafc", "#64748b")
        + '</svg>'
    )

    # 3. VAE
    svg_vae = (
        f'<svg viewBox="0 0 680 86" class="arch-svg">{defs}'
        + _box(6, 22, 74, 42, "Input x", "784", "#f8fafc", "#64748b")
        + _arrow(80, 43, 100, 43)
        + _box(100, 22, 86, 42, "Encoder", "256, ReLU", "#eef4ff", "#2563eb")
        + _arrow(186, 33, 212, 20)
        + _arrow(186, 53, 212, 66)
        + _box(212, 4, 82, 32, "mu (64)", "Linear", "#fef3c7", "#d97706")
        + _box(212, 50, 82, 32, "log var (64)", "Linear", "#fef3c7", "#d97706")
        + _arrow(294, 20, 322, 36)
        + _arrow(294, 66, 322, 50)
        + _box(322, 22, 114, 42, "Reparam. z", "mu + sigma * eps", "#f3e8ff", "#7c3aed")
        + _arrow(436, 43, 458, 43)
        + _box(458, 22, 92, 42, "Decoder", "256, ReLU", "#ecfdf5", "#059669")
        + _arrow(550, 43, 572, 43)
        + _box(572, 22, 98, 42, "Output x_hat", "784, Sigmoid", "#f8fafc", "#64748b")
        + '</svg>'
    )

    # 4. Vanilla GAN
    svg_vgan = (
        f'<svg viewBox="0 0 680 72" class="arch-svg">{defs}'
        + _box(6, 15, 86, 42, "Prior z", "64 ~ N(0,I)", "#f3e8ff", "#7c3aed")
        + _arrow(92, 36, 116, 36)
        + _box(116, 15, 134, 42, "Generator (MLP)", "256 -> 512 -> 784, Tanh", "#ecfdf5", "#059669")
        + _arrow(250, 36, 276, 36)
        + _box(276, 15, 92, 42, "Fake / Real", "28x28 [-1,1]", "#f8fafc", "#64748b")
        + _arrow(368, 36, 394, 36)
        + _box(394, 15, 146, 42, "Discriminator (MLP)", "784->512->256 (Drop 0.3)", "#fee2e2", "#dc2626")
        + _arrow(540, 36, 566, 36)
        + _box(566, 15, 102, 42, "Logit D(x)", "BCEWithLogits", "#fef3c7", "#d97706")
        + '</svg>'
    )

    # 5. DCGAN
    svg_dcgan = (
        f'<svg viewBox="0 0 680 72" class="arch-svg">{defs}'
        + _box(6, 15, 86, 42, "Prior z", "100x1x1", "#f3e8ff", "#7c3aed")
        + _arrow(92, 36, 114, 36)
        + _box(114, 15, 166, 42, "G (ConvTranspose2d)", "256(7x7)->128(14x14)->1(28x28)", "#ecfdf5", "#059669")
        + _arrow(280, 36, 302, 36)
        + _box(302, 15, 82, 42, "Citra", "28x28 [-1,1]", "#f8fafc", "#64748b")
        + _arrow(384, 36, 406, 36)
        + _box(406, 15, 162, 42, "D (Strided Conv2d)", "64(14x14)->128(7x7)->256(4x4)", "#fee2e2", "#dc2626")
        + _arrow(568, 36, 590, 36)
        + _box(590, 15, 82, 42, "Logit D(x)", "Linear->1", "#fef3c7", "#d97706")
        + '</svg>'
    )

    # 6. DDPM
    svg_ddpm = (
        f'<svg viewBox="0 0 680 92" class="arch-svg">{defs}'
        + _box(6, 28, 84, 42, "x_0 + t", "q(x_t|x_0)", "#f8fafc", "#64748b")
        + _arrow(90, 49, 112, 49)
        + _box(112, 28, 102, 42, "UNet Down", "64(28)->128(14)", "#eef4ff", "#2563eb")
        + _arrow(214, 49, 236, 49)
        + _box(236, 28, 122, 42, "Bottleneck 7x7", "256ch + SelfAttn", "#f3e8ff", "#7c3aed")
        + _arrow(358, 49, 384, 49, "skip")
        + _box(384, 28, 102, 42, "UNet Up", "128(14)->64(28)", "#ecfdf5", "#059669")
        + _arrow(486, 49, 512, 49)
        + _box(512, 28, 152, 42, "Pred eps_theta(x_t, t)", "Noise MSE + EMA", "#fef3c7", "#d97706")
        + _box(226, 2, 142, 18, "Sinusoidal Time Emb t", "", "#fee2e2", "#dc2626")
        + _arrow(297, 20, 297, 28)
        + _arrow(226, 11, 163, 28)
        + _arrow(368, 11, 435, 28)
        + '</svg>'
    )

    cards = [
        ("1. MLP Classifier Pendamping & Frozen Latent Decoder (Bagian 2.1)", svg_mlp),
        ("2 & 3. Vanilla Autoencoder & Sparse Autoencoder (L1 Penalty)", svg_ae),
        ("4. Variational Autoencoder (VAE — Reparameterization & ELBO)", svg_vae),
        ("5. Vanilla GAN (Fully-Connected Generator & Discriminator)", svg_vgan),
        ("6. Deep Convolutional GAN (DCGAN — Strided Conv & BatchNorm)", svg_dcgan),
        ("7. Denoising Diffusion Probabilistic Model (DDPM — UNet28 + Time Emb + EMA)", svg_ddpm),
    ]
    blocks = "".join(
        f'<div class="diagram-card"><div class="diagram-title">{title}</div>{svg}</div>'
        for title, svg in cards
    )
    return f'<div class="diagram-grid">{blocks}</div>'
