import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterator

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from dataset.youtube_vis_colorization import YouTubeVISColorizationDataset
from models.ddcolor_wrapper import build_ddcolor_tracktention_wrapper


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infinite_loader(loader: DataLoader) -> Iterator[Dict]:
    while True:
        yield from loader


def prepare_ddcolor_imports(ddcolor_root: str):
    ddcolor_root = Path(ddcolor_root).resolve()
    if not ddcolor_root.exists():
        raise FileNotFoundError(f"DDColor root not found: {ddcolor_root}")
    if str(ddcolor_root) not in sys.path:
        sys.path.insert(0, str(ddcolor_root))

    from basicsr.archs.ddcolor_arch import DDColor, DynamicUNetDiscriminator
    from basicsr.losses import build_loss
    from basicsr.utils.img_util import tensor_lab2rgb
    from basicsr.data.transforms import rgb2lab

    return DDColor, DynamicUNetDiscriminator, build_loss, tensor_lab2rgb, rgb2lab


def build_ddcolor_generator(cfg):
    DDColor, *_ = prepare_ddcolor_imports(cfg["ddcolor"]["root"])
    dc = cfg["ddcolor"]
    img_size = cfg["data"]["image_size"]

    model = DDColor(
        encoder_name=dc["encoder_name"],
        decoder_name="MultiScaleColorDecoder",
        num_input_channels=3,
        input_size=(img_size, img_size),
        nf=dc["nf"],
        num_output_channels=2,
        last_norm=dc.get("last_norm", "Spectral"),
        do_normalize=False,
        num_queries=dc["num_queries"],
        num_scales=dc["num_scales"],
        dec_layers=dc["dec_layers"],
        encoder_from_pretrain=dc["encoder_from_pretrain"],
    )

    if dc.get("checkpoint") is not None:
        ckpt_path = Path(dc["checkpoint"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"DDColor checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("params_ema") or ckpt.get("params") or ckpt.get("state_dict") or ckpt

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print("[ddcolor] missing keys:\n" + "\n".join(f"  {k}" for k in missing))
        if unexpected:
            print("[ddcolor] unexpected keys:\n" + "\n".join(f"  {k}" for k in unexpected))
        print(f"[ddcolor] loaded checkpoint from {ckpt_path}")

    return model


def build_ddcolor_discriminator(cfg):
    _, DynamicUNetDiscriminator, *_ = prepare_ddcolor_imports(cfg["ddcolor"]["root"])
    return DynamicUNetDiscriminator(n_channels=3, nf=cfg["ddcolor"]["disc_nf"])


def build_ddcolor_losses(cfg, device):
    _, _, build_loss, *_ = prepare_ddcolor_imports(cfg["ddcolor"]["root"])

    loss_opts = {
        "pixel": {"type": "L1Loss", "loss_weight": 0.1, "reduction": "mean"},
        "perceptual": {
            "type": "PerceptualLoss",
            "layer_weights": {"conv1_1": 0.0625, "conv2_1": 0.125, "conv3_1": 0.25, "conv4_1": 0.5, "conv5_1": 1.0},
            "vgg_type": "vgg16_bn",
            "use_input_norm": True,
            "range_norm": False,
            "perceptual_weight": 5.0,
            "style_weight": 0.0,
            "criterion": "l1",
        },
        "gan": {"type": "GANLoss", "gan_type": "vanilla", "real_label_val": 1.0, "fake_label_val": 0.0, "loss_weight": 1.0},
        "colorfulness": {"type": "ColorfulnessLoss", "loss_weight": 0.5},
    }
    return {name: build_loss(opt).to(device) for name, opt in loss_opts.items()}


def freeze_ddcolor_backbone(wrapper_model: nn.Module) -> None:
    for p in wrapper_model.model.parameters():
        p.requires_grad = False
    for p in wrapper_model.tracktentions.parameters():
        p.requires_grad = True


def save_checkpoint(
    out_dir: Path,
    step: int,
    model_g: nn.Module,
    model_d: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scheduler_g: torch.optim.lr_scheduler._LRScheduler,
    scheduler_d: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    cfg: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model_g": model_g.state_dict(),
        "model_d": model_d.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "scheduler_g": scheduler_g.state_dict(),
        "scheduler_d": scheduler_d.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": cfg,
    }
    torch.save(ckpt, out_dir / f"checkpoint_{step:06d}.pt")
    torch.save(ckpt, out_dir / "last.pt")


def load_checkpoint(
    ckpt_path: str,
    model_g: nn.Module,
    model_d: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scheduler_g: torch.optim.lr_scheduler._LRScheduler,
    scheduler_d: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
) -> int:
    ckpt = torch.load(ckpt_path, map_location=device)
    model_g.load_state_dict(ckpt["model_g"], strict=True)
    model_d.load_state_dict(ckpt["model_d"], strict=True)
    optimizer_g.load_state_dict(ckpt["optimizer_g"])
    optimizer_d.load_state_dict(ckpt["optimizer_d"])
    scheduler_g.load_state_dict(ckpt["scheduler_g"])
    scheduler_d.load_state_dict(ckpt["scheduler_d"])
    scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["step"])


def build_scheduler(optimizer: torch.optim.Optimizer) -> MultiStepLR:
    return MultiStepLR(optimizer, milestones=list(range(8000, 40000, 4000)), gamma=0.5)


def maybe_to_device(x, device):
    return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x


def flatten_bt(x: torch.Tensor) -> torch.Tensor:
    B, T = x.shape[:2]
    return x.view(B * T, *x.shape[2:])


def rgb_to_lab_ddcolor_batch(rgb: torch.Tensor, rgb2lab_fn) -> torch.Tensor:
    """Convert (B, T, 3, H, W) RGB in [0,1] to LAB via DDColor's rgb2lab."""
    B, T = rgb.shape[:2]
    rgb_np = rgb.detach().cpu().numpy()

    lab_frames = []
    for b in range(B):
        lab_clip = []
        for t in range(T):
            img_l, img_ab = rgb2lab_fn(np.transpose(rgb_np[b, t], (1, 2, 0)))
            lab_clip.append(
                torch.from_numpy(np.concatenate([img_l, img_ab], axis=-1)).permute(2, 0, 1).float()
            )
        lab_frames.append(torch.stack(lab_clip, dim=0))

    return torch.stack(lab_frames, dim=0).to(rgb.device)


def train_one_step(
    model_g: nn.Module,
    model_d: nn.Module,
    losses: Dict[str, nn.Module],
    tensor_lab2rgb,
    rgb2lab_fn,
    batch: Dict,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
):
    rgb = maybe_to_device(batch["rgb"], device)
    tracks = maybe_to_device(batch["tracks"], device)

    lab = rgb_to_lab_ddcolor_batch(rgb, rgb2lab_fn)
    l_chan = lab[:, :, 0:1]
    gt_ab = lab[:, :, 1:3]
    lq_rgb = tensor_lab2rgb(flatten_bt(torch.cat([l_chan, torch.zeros_like(gt_ab)], dim=2))).view_as(rgb)

    # generator step
    for p in model_d.parameters():
        p.requires_grad = False

    optimizer_g.zero_grad(set_to_none=True)

    with autocast(enabled=use_amp):
        pred_ab = model_g(lq_rgb, tracks)
        pred_rgb = tensor_lab2rgb(flatten_bt(torch.cat([l_chan, pred_ab], dim=2))).view_as(rgb)

        pred_ab_flat = flatten_bt(pred_ab)
        gt_ab_flat = flatten_bt(gt_ab)
        pred_rgb_flat = flatten_bt(pred_rgb)
        gt_rgb_flat = flatten_bt(rgb)

        l_g_pix = losses["pixel"](pred_ab_flat, gt_ab_flat)
        l_g_percep, l_g_style = losses["perceptual"](pred_rgb_flat, gt_rgb_flat)
        l_g_percep = l_g_percep or 0.0
        l_g_style = l_g_style or 0.0
        l_g_gan = losses["gan"](model_d(pred_rgb_flat), target_is_real=True, is_disc=False)
        l_g_color = losses["colorfulness"](pred_rgb_flat)
        l_g_total = l_g_pix + l_g_percep + l_g_style + l_g_gan + l_g_color

    scaler.scale(l_g_total).backward()
    scaler.step(optimizer_g)

    # discriminator step
    for p in model_d.parameters():
        p.requires_grad = True

    optimizer_d.zero_grad(set_to_none=True)

    with autocast(enabled=use_amp):
        real_d_pred = model_d(gt_rgb_flat)
        fake_d_pred = model_d(pred_rgb_flat.detach())
        l_d = losses["gan"](real_d_pred, target_is_real=True, is_disc=True) + \
              losses["gan"](fake_d_pred, target_is_real=False, is_disc=True)

    scaler.scale(l_d).backward()
    scaler.step(optimizer_d)
    scaler.update()

    def scalar(x):
        return float(x) if isinstance(x, float) else float(x.item())

    return {
        "l_g_total": scalar(l_g_total),
        "l_g_pix": scalar(l_g_pix),
        "l_g_percep": scalar(l_g_percep),
        "l_g_style": scalar(l_g_style),
        "l_g_gan": scalar(l_g_gan),
        "l_g_color": scalar(l_g_color),
        "l_d": scalar(l_d),
        "real_score": float(real_d_pred.detach().mean()),
        "fake_score": float(fake_d_pred.detach().mean()),
    }


@torch.no_grad()
def evaluate(
    model_g: nn.Module,
    tensor_lab2rgb,
    rgb2lab_fn,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    max_batches: int = 20,
):
    model_g.eval()
    batch_losses = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        rgb = maybe_to_device(batch["rgb"], device)
        tracks = maybe_to_device(batch["tracks"], device)

        lab = rgb_to_lab_ddcolor_batch(rgb, rgb2lab_fn)
        l_chan = lab[:, :, 0:1]
        gt_ab = lab[:, :, 1:3]
        lq_rgb = tensor_lab2rgb(flatten_bt(torch.cat([l_chan, torch.zeros_like(gt_ab)], dim=2))).view_as(rgb)

        with autocast(enabled=use_amp):
            pred_ab = model_g(lq_rgb, tracks)
            batch_losses.append(torch.nn.functional.l1_loss(pred_ab, gt_ab).item())

    model_g.train()
    return {"val_l1_ab": float(np.mean(batch_losses)) if batch_losses else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Override resume checkpoint path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.resume is not None:
        cfg["system"]["resume"] = args.resume

    data_cfg = cfg["data"]
    optim_cfg = cfg["optim"]
    sys_cfg = cfg["system"]
    tt_cfg = cfg["tracktention"]

    seed_everything(sys_cfg["seed"])

    device = torch.device(sys_cfg["device"] if torch.cuda.is_available() else "cpu")
    output_dir = Path(sys_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.yaml").write_text(Path(args.config).read_text())

    train_dataset = YouTubeVISColorizationDataset(
        frames_root=data_cfg["frames_root"],
        tracks_root=data_cfg["tracks_root"],
        split=data_cfg["train_split"],
        clip_len=data_cfg["clip_len"],
        image_size=data_cfg["image_size"],
        stride=data_cfg["train_stride"],
        require_tracks=True,
    )
    valid_dataset = YouTubeVISColorizationDataset(
        frames_root=data_cfg["frames_root"],
        tracks_root=data_cfg["tracks_root"],
        split=data_cfg["valid_split"],
        clip_len=data_cfg["clip_len"],
        image_size=data_cfg["image_size"],
        stride=data_cfg["valid_stride"],
        require_tracks=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=optim_cfg["batch_size"], shuffle=True,
                              num_workers=data_cfg["num_workers"], pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=optim_cfg["batch_size"], shuffle=False,
                              num_workers=data_cfg["num_workers"], pin_memory=True, drop_last=False)

    _, _, _, tensor_lab2rgb, rgb2lab_fn = prepare_ddcolor_imports(cfg["ddcolor"]["root"])

    model_g = build_ddcolor_tracktention_wrapper(
        ddcolor_model=build_ddcolor_generator(cfg),
        stage_names=tt_cfg["stage_names"],
        feature_dims=tt_cfg["feature_dims"],
        num_heads=tt_cfg["num_heads"],
        num_layers=tt_cfg["num_layers"],
        sigma=tt_cfg["sigma"],
    ).to(device)

    model_d = build_ddcolor_discriminator(cfg).to(device)

    if tt_cfg.get("freeze_backbone", False):
        freeze_ddcolor_backbone(model_g)

    losses = build_ddcolor_losses(cfg, device)

    optimizer_g = AdamW(
        [p for p in model_g.parameters() if p.requires_grad],
        lr=optim_cfg["lr_g"], weight_decay=optim_cfg["weight_decay_g"], betas=(0.9, 0.99),
    )
    optimizer_d = AdamW(
        model_d.parameters(),
        lr=optim_cfg["lr_d"], weight_decay=optim_cfg["weight_decay_d"], betas=(0.9, 0.99),
    )

    scheduler_g = build_scheduler(optimizer_g)
    scheduler_d = build_scheduler(optimizer_d)
    scaler = GradScaler(enabled=optim_cfg["use_amp"])

    start_step = 0
    if sys_cfg.get("resume") is not None:
        start_step = load_checkpoint(
            sys_cfg["resume"], model_g, model_d,
            optimizer_g, optimizer_d,
            scheduler_g, scheduler_d,
            scaler, device,
        )
        print(f"[resume] loaded checkpoint from {sys_cfg['resume']} at step {start_step}")

    model_g.train()
    model_d.train()
    train_iter = infinite_loader(train_loader)

    for step in range(start_step, optim_cfg["total_iters"]):
        logs = train_one_step(
            model_g=model_g, model_d=model_d, losses=losses,
            tensor_lab2rgb=tensor_lab2rgb, rgb2lab_fn=rgb2lab_fn,
            batch=next(train_iter),
            optimizer_g=optimizer_g, optimizer_d=optimizer_d,
            scaler=scaler, device=device, use_amp=optim_cfg["use_amp"],
        )

        scheduler_g.step()
        scheduler_d.step()

        if (step + 1) % 50 == 0:
            lr_g = optimizer_g.param_groups[0]["lr"]
            lr_d = optimizer_d.param_groups[0]["lr"]
            print(
                f"[train] step={step+1:06d} "
                f"l_g={logs['l_g_total']:.4f} pix={logs['l_g_pix']:.4f} "
                f"percep={logs['l_g_percep']:.4f} gan={logs['l_g_gan']:.4f} "
                f"color={logs['l_g_color']:.4f} l_d={logs['l_d']:.4f} "
                f"real={logs['real_score']:.3f} fake={logs['fake_score']:.3f} "
                f"lr_g={lr_g:.2e} lr_d={lr_d:.2e}"
            )

        if (step + 1) % optim_cfg["eval_every"] == 0:
            metrics = evaluate(model_g, tensor_lab2rgb, rgb2lab_fn, valid_loader, device, optim_cfg["use_amp"])
            print(f"[valid] step={step+1:06d} val_l1_ab={metrics['val_l1_ab']}")

        if (step + 1) % optim_cfg["save_every"] == 0:
            save_checkpoint(output_dir, step + 1, model_g, model_d,
                            optimizer_g, optimizer_d, scheduler_g, scheduler_d, scaler, cfg)

    save_checkpoint(output_dir, optim_cfg["total_iters"], model_g, model_d,
                    optimizer_g, optimizer_d, scheduler_g, scheduler_d, scaler, cfg)


if __name__ == "__main__":
    main()