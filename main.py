"""End-to-end training, evaluation, and figure generation pipeline."""

import json
from typing import Any, Dict, List

import torch

from src.config import ExperimentConfig
from src.data.fashion_mnist import create_dataloaders
from src.evaluation.metrics import (
    InceptionFeatureExtractor,
    compute_fid,
    compute_reconstruction_metrics,
    evaluate_autoencoder_on_test,
    measure_inference_latency,
)
from src.evaluation.visualize import (
    plot_fig1_reconstructions,
    plot_fig2_random_sampling,
    plot_fig3_mlp_proof,
    plot_fig4_latent_interpolation,
    plot_fig5_losses_and_ddpm_trajectory,
)
from src.experiments.mlp_proof import run_mlp_proof
from src.experiments.train_autoencoders import train_all_autoencoders
from src.experiments.train_ddpm import train_ddpm
from src.experiments.train_gans import train_all_gans
from src.utils.seed import get_device, set_seed


def _select_distinct_test_samples(loaders: Dict[str, Any], num_classes_to_pick: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick one clean test image per distinct class for Figure 1."""
    picked_x, picked_y = [], []
    seen = set()
    for x, y in loaders["test"]:
        for i in range(x.size(0)):
            lbl = int(y[i])
            if lbl not in seen:
                seen.add(lbl)
                picked_x.append(x[i : i + 1])
                picked_y.append(y[i : i + 1])
                if len(seen) == num_classes_to_pick:
                    return torch.cat(picked_x, dim=0), torch.cat(picked_y, dim=0)
    raise RuntimeError("Could not find enough distinct classes in test loader.")


def _select_contrasting_pair(loaders: Dict[str, Any], class_a: int = 7, class_b: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one test image from class_a (Sneaker) and one from class_b (Dress) for interpolation."""
    img_a, img_b = None, None
    for x, y in loaders["test"]:
        for i in range(x.size(0)):
            lbl = int(y[i])
            if lbl == class_a and img_a is None:
                img_a = x[i : i + 1]
            elif lbl == class_b and img_b is None:
                img_b = x[i : i + 1]
            if img_a is not None and img_b is not None:
                return img_a, img_b
    raise RuntimeError("Could not locate contrasting class pair.")


@torch.no_grad()
def _generate_samples_batched(gen_fn, total: int, batch_size: int) -> torch.Tensor:
    batches = []
    remaining = total
    while remaining > 0:
        cur = min(batch_size, remaining)
        batches.append(gen_fn(cur).cpu())
        remaining -= cur
    return torch.cat(batches, dim=0)


def main() -> None:
    cfg = ExperimentConfig()
    set_seed(cfg.seed)
    device = get_device()
    for d in [cfg.data_dir, cfg.checkpoint_dir, cfg.results_dir, cfg.figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Running Comparative Study on device: {device}")
    loaders = create_dataloaders(cfg)
    print(
        f"Dataset splits -> Train: {len(loaders['train'].dataset)}, "
        f"Val: {len(loaders['val'].dataset)}, Test: {len(loaders['test'].dataset)}"
    )

    # 1. Section 2.1 MLP Classifier Limitation Proof
    mlp_res = run_mlp_proof(cfg, loaders, device)

    # 2. Train all 6 Generative Architectures
    ae_res = train_all_autoencoders(cfg, loaders, device)
    gan_res = train_all_gans(cfg, loaders, device)
    ddpm_res = train_ddpm(cfg, loaders, device)

    # 3. Quantitative Evaluation on Test Set
    print("\n=== Quantitative Evaluation on Test Set ===")
    test_images = torch.cat([x for x, _ in loaders["test"]], dim=0)

    frozen_metrics = compute_reconstruction_metrics(
        mlp_res["recons_frozen_all"], mlp_res["targets_all"], device
    )
    transpose_onehot_metrics = compute_reconstruction_metrics(
        mlp_res["sample_onehot_transpose"], mlp_res["sample_x"], device
    )
    transpose_latent_metrics = compute_reconstruction_metrics(
        mlp_res["sample_latent_transpose"], mlp_res["sample_x"], device
    )

    eval_vanilla_ae = evaluate_autoencoder_on_test(ae_res["vanilla_ae"]["model"], loaders["test"], device)
    eval_sparse_ae = evaluate_autoencoder_on_test(ae_res["sparse_ae"]["model"], loaders["test"], device)
    eval_vae = evaluate_autoencoder_on_test(ae_res["vae"]["model"], loaders["test"], device, is_vae=True)

    # Latency measurements (ms / batch 64)
    dummy_x64 = torch.rand(cfg.latency_batch_size, 1, 28, 28, device=device)
    dummy_z_vae = torch.randn(cfg.latency_batch_size, cfg.vae_latent_dim, device=device)
    dummy_z_gan = torch.randn(cfg.latency_batch_size, cfg.gan_latent_dim, device=device)
    dummy_z_dcgan = torch.randn(cfg.latency_batch_size, cfg.dcgan_latent_dim, device=device)

    vae_model = ae_res["vae"]["model"].eval()
    vgan_g = gan_res["vanilla_gan"]["generator"].eval()
    dcgan_g = gan_res["dcgan"]["generator"].eval()
    diffusion = ddpm_res["diffusion"]
    ema_unet = ddpm_res["ema_model"].eval()

    lat_vanilla_ae = measure_inference_latency(
        lambda: ae_res["vanilla_ae"]["model"](dummy_x64)[0], device, cfg.latency_warmup_runs, cfg.latency_measure_runs
    )
    lat_sparse_ae = measure_inference_latency(
        lambda: ae_res["sparse_ae"]["model"](dummy_x64)[0], device, cfg.latency_warmup_runs, cfg.latency_measure_runs
    )
    lat_vae = measure_inference_latency(
        lambda: vae_model.decode(dummy_z_vae), device, cfg.latency_warmup_runs, cfg.latency_measure_runs
    )
    lat_vgan = measure_inference_latency(
        lambda: (vgan_g(dummy_z_gan) + 1.0) / 2.0, device, cfg.latency_warmup_runs, cfg.latency_measure_runs
    )
    lat_dcgan = measure_inference_latency(
        lambda: (dcgan_g(dummy_z_dcgan) + 1.0) / 2.0, device, cfg.latency_warmup_runs, cfg.latency_measure_runs
    )
    lat_ddpm = measure_inference_latency(
        lambda: diffusion.sample(cfg.latency_batch_size, device, denoiser=ema_unet)[0],
        device,
        warmup=2,
        runs=5,
    )

    # FID computation on 7,000 generated samples vs 7,000 Test Set images
    print("Computing InceptionV3 FID scores (7,000 test vs 7,000 generated samples)...")
    extractor = InceptionFeatureExtractor(device)

    set_seed(cfg.seed)
    gen_vae_all = _generate_samples_batched(
        lambda n: vae_model.decode(torch.randn(n, cfg.vae_latent_dim, device=device)),
        cfg.fid_num_samples,
        batch_size=256,
    )
    gen_vgan_all = _generate_samples_batched(
        lambda n: (vgan_g(torch.randn(n, cfg.gan_latent_dim, device=device)) + 1.0) / 2.0,
        cfg.fid_num_samples,
        batch_size=256,
    )
    gen_dcgan_all = _generate_samples_batched(
        lambda n: (dcgan_g(torch.randn(n, cfg.dcgan_latent_dim, device=device)) + 1.0) / 2.0,
        cfg.fid_num_samples,
        batch_size=256,
    )
    gen_ddpm_all = _generate_samples_batched(
        lambda n: diffusion.sample(n, device, denoiser=ema_unet)[0],
        cfg.fid_num_samples,
        batch_size=500,
    )

    fid_vae = compute_fid(test_images, gen_vae_all, extractor, device)
    fid_vgan = compute_fid(test_images, gen_vgan_all, extractor, device)
    fid_dcgan = compute_fid(test_images, gen_dcgan_all, extractor, device)
    fid_ddpm = compute_fid(test_images, gen_ddpm_all, extractor, device)

    # 4. Generate 5 Mandatory High-Res Figures
    print("\n=== Generating 5 Mandatory Figures (300 DPI) ===")
    distinct_x, distinct_y = _select_distinct_test_samples(loaders, 8)
    with torch.no_grad():
        dx = distinct_x.to(device)
        fig1_van = ae_res["vanilla_ae"]["model"](dx)[0].cpu()
        fig1_spa = ae_res["sparse_ae"]["model"](dx)[0].cpu()
        fig1_vae = vae_model.decode(vae_model.encode(dx)).cpu()
    plot_fig1_reconstructions(
        distinct_x, distinct_y, fig1_van, fig1_spa, fig1_vae,
        cfg.figures_dir / "fig1_reconstruction_grid.png",
    )

    plot_fig2_random_sampling(
        {
            "vae": gen_vae_all[:16],
            "vanilla_gan": gen_vgan_all[:16],
            "dcgan": gen_dcgan_all[:16],
            "ddpm": gen_ddpm_all[:16],
        },
        cfg.figures_dir / "fig2_generative_sampling_4x4.png",
    )

    with torch.no_grad():
        van_recon_for_mlp = ae_res["vanilla_ae"]["model"](mlp_res["sample_x"].to(device))[0].cpu()
    plot_fig3_mlp_proof(mlp_res, van_recon_for_mlp, cfg.figures_dir / "fig3_mlp_limitation_proof.png")

    # Figure 4: Linear Latent Interpolation
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    img_a, img_b = _select_contrasting_pair(loaders, class_a=7, class_b=3)
    img_a, img_b = img_a.to(device), img_b.to(device)

    def _interp_encoder_model(m, is_vae=False):
        with torch.no_grad():
            za = m.encode(img_a)
            zb = m.encode(img_b)
            zs = torch.cat([(1.0 - a) * za + a * zb for a in alphas], dim=0)
            return m.decode(zs).cpu()

    def _interp_generator_model(g, z_dim: int):
        with torch.no_grad():
            gen = torch.Generator(device=device).manual_seed(cfg.seed + 99)
            za = torch.randn(1, z_dim, device=device, generator=gen)
            zb = torch.randn(1, z_dim, device=device, generator=gen)
            zs = torch.cat([(1.0 - a) * za + a * zb for a in alphas], dim=0)
            return ((g(zs) + 1.0) / 2.0).cpu()

    interp_rows = [
        ("Vanilla AE", _interp_encoder_model(ae_res["vanilla_ae"]["model"])),
        ("Sparse AE", _interp_encoder_model(ae_res["sparse_ae"]["model"])),
        ("VAE", _interp_encoder_model(vae_model, is_vae=True)),
        ("Vanilla GAN", _interp_generator_model(vgan_g, cfg.gan_latent_dim)),
        ("DCGAN", _interp_generator_model(dcgan_g, cfg.dcgan_latent_dim)),
    ]
    plot_fig4_latent_interpolation(interp_rows, alphas, cfg.figures_dir / "fig4_latent_interpolation.png")

    # Figure 5: Loss curves & DDPM denoising trajectory
    record_steps = [999, 900, 800, 700, 600, 450, 300, 150, 50, 0]
    set_seed(cfg.seed + 7)
    _, ddpm_traj = diffusion.sample(4, device, denoiser=ema_unet, record_steps=record_steps)
    histories = {
        "vanilla_ae": {"train_losses": ae_res["vanilla_ae"]["train_losses"], "val_losses": ae_res["vanilla_ae"]["val_losses"]},
        "sparse_ae": {"train_losses": ae_res["sparse_ae"]["train_losses"], "val_losses": ae_res["sparse_ae"]["val_losses"]},
        "vae": {"train_losses": ae_res["vae"]["train_losses"], "val_losses": ae_res["vae"]["val_losses"]},
        "vanilla_gan": {"g_losses": gan_res["vanilla_gan"]["g_losses"], "d_losses": gan_res["vanilla_gan"]["d_losses"]},
        "dcgan": {"g_losses": gan_res["dcgan"]["g_losses"], "d_losses": gan_res["dcgan"]["d_losses"]},
        "ddpm": {"train_losses": ddpm_res["train_losses"]},
    }
    plot_fig5_losses_and_ddpm_trajectory(
        histories, ddpm_traj, cfg.figures_dir / "fig5_loss_curves_and_ddpm_trajectory.png"
    )

    # 5. Save comprehensive JSON metrics
    summary = {
        "dataset": {
            "name": cfg.dataset_name,
            "train_size": len(loaders["train"].dataset),
            "val_size": len(loaders["val"].dataset),
            "test_size": len(loaders["test"].dataset),
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
        },
        "mlp_proof": {
            "test_accuracy": mlp_res["test_accuracy"],
            "val_accuracy": mlp_res["val_accuracy"],
            "orthogonality": mlp_res["orthogonality"],
            "z_mlp_zero_fraction": mlp_res["z_mlp_zero_fraction"],
            "z_mlp_mu_norm": mlp_res["z_mlp_mu_norm"],
            "z_mlp_sigma_mean": mlp_res["z_mlp_sigma_mean"],
            "transpose_onehot_metrics": transpose_onehot_metrics,
            "transpose_latent_metrics": transpose_latent_metrics,
            "frozen_decoder_metrics": frozen_metrics,
        },
        "models": {
            "vanilla_ae": {
                "latent_dim": cfg.ae_latent_dim,
                "epoch_time_s": ae_res["vanilla_ae"]["mean_epoch_time"],
                "latency_ms_mean": lat_vanilla_ae[0],
                "latency_ms_std": lat_vanilla_ae[1],
                **eval_vanilla_ae["metrics"],
                "fid": None,
            },
            "sparse_ae": {
                "latent_dim": cfg.sparse_latent_dim,
                "l1_lambda": cfg.sparse_l1_lambda,
                "epoch_time_s": ae_res["sparse_ae"]["mean_epoch_time"],
                "latency_ms_mean": lat_sparse_ae[0],
                "latency_ms_std": lat_sparse_ae[1],
                **eval_sparse_ae["metrics"],
                "fid": None,
            },
            "vae": {
                "latent_dim": cfg.vae_latent_dim,
                "epoch_time_s": ae_res["vae"]["mean_epoch_time"],
                "latency_ms_mean": lat_vae[0],
                "latency_ms_std": lat_vae[1],
                **eval_vae["metrics"],
                "fid": fid_vae,
            },
            "vanilla_gan": {
                "latent_dim": cfg.gan_latent_dim,
                "epoch_time_s": gan_res["vanilla_gan"]["mean_epoch_time"],
                "latency_ms_mean": lat_vgan[0],
                "latency_ms_std": lat_vgan[1],
                "mse": None,
                "psnr": None,
                "ssim": None,
                "fid": fid_vgan,
            },
            "dcgan": {
                "latent_dim": cfg.dcgan_latent_dim,
                "epoch_time_s": gan_res["dcgan"]["mean_epoch_time"],
                "latency_ms_mean": lat_dcgan[0],
                "latency_ms_std": lat_dcgan[1],
                "mse": None,
                "psnr": None,
                "ssim": None,
                "fid": fid_dcgan,
            },
            "ddpm": {
                "latent_dim": "28x28",
                "timesteps": cfg.ddpm_timesteps,
                "epoch_time_s": ddpm_res["mean_epoch_time"],
                "latency_ms_mean": lat_ddpm[0],
                "latency_ms_std": lat_ddpm[1],
                "mse": None,
                "psnr": None,
                "ssim": None,
                "fid": fid_ddpm,
            },
        },
        "histories": histories,
    }
    metrics_path = cfg.results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved complete quantitative summary to {metrics_path}")
    print(json.dumps({k: v for k, v in summary.items() if k != "histories"}, indent=2))


if __name__ == "__main__":
    main()
