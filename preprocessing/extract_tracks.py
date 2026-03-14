import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF


IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_frame_files(frame_dir: Path) -> List[Path]:
    files = [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    return files


def load_video_from_frames(frame_dir: Path, image_size: int) -> Tuple[torch.Tensor, List[str]]:
    frame_files = list_frame_files(frame_dir)
    if not frame_files:
        raise RuntimeError(f"No image frames found in: {frame_dir}")

    frames, names = [], []
    for fp in frame_files:
        img = Image.open(fp).convert("RGB")
        img = TF.resize(img, image_size, antialias=True)
        frames.append(np.array(img))
        names.append(fp.name)

    video = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).float()
    return video, names


def sample_spacetime_queries(
    num_frames: int,
    height: int,
    width: int,
    num_queries: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Returns (N, 3) queries with columns [frame_idx, x, y]."""
    frame_ids = rng.integers(0, num_frames, size=(num_queries,))
    xs = rng.uniform(0.0, max(width - 1, 1), size=(num_queries,))
    ys = rng.uniform(0.0, max(height - 1, 1), size=(num_queries,))
    queries = np.stack([frame_ids, xs, ys], axis=-1).astype(np.float32)
    return torch.from_numpy(queries)


def load_cotracker3_offline(device: torch.device):
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    return model.to(device).eval()


@torch.no_grad()
def run_cotracker_offline(
    model,
    video_tchw: torch.Tensor,
    queries_n3: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_tracks, pred_visibility = model(
        video_tchw.unsqueeze(0).to(device),
        queries=queries_n3.unsqueeze(0).to(device),
        backward_tracking=True,
    )
    return pred_tracks[0].cpu(), pred_visibility[0].cpu()


def save_track_file(
    out_file: Path,
    tracks_tn2: torch.Tensor,
    visibility_tn: torch.Tensor,
    queries_n3: torch.Tensor,
    frame_names: List[str],
    image_hw: Tuple[int, int],
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "tracks": tracks_tn2,
        "visibility": visibility_tn,
        "queries": queries_n3,
        "frame_names": frame_names,
        "image_hw": image_hw,
    }, out_file)


def write_meta_json(
    out_file: Path,
    clip_name: str,
    num_frames: int,
    image_hw: Tuple[int, int],
    num_queries: int,
    seed: int,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "clip_name": clip_name,
        "num_frames": num_frames,
        "image_hw": list(image_hw),
        "num_queries": num_queries,
        "sampling": "uniform_spacetime",
        "seed": seed,
    }, indent=2))


def process_clip(
    clip_dir: Path,
    output_root: Path,
    model,
    device: torch.device,
    num_queries: int,
    image_size: int,
    seed: int,
    overwrite: bool,
) -> None:
    clip_name = clip_dir.name
    out_dir = output_root / clip_name
    out_pt = out_dir / "tracks.pt"

    if out_pt.exists() and not overwrite:
        print(f"[skip] {clip_name}")
        return

    video_tchw, frame_names = load_video_from_frames(clip_dir, image_size)
    T, C, H, W = video_tchw.shape

    clip_seed = seed + abs(hash(clip_name)) % (2**32)
    rng = np.random.default_rng(clip_seed)

    queries_n3 = sample_spacetime_queries(T, H, W, num_queries, rng)
    tracks_tn2, visibility_tn = run_cotracker_offline(model, video_tchw, queries_n3, device)

    save_track_file(out_pt, tracks_tn2, visibility_tn, queries_n3, frame_names, (H, W))
    write_meta_json(out_dir / "meta.json", clip_name, T, (H, W), num_queries, clip_seed)

    print(f"[done] {clip_name}: T={T}, H={H}, W={W}, queries={num_queries}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=576)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clip", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_cotracker3_offline(device)

    if args.clip is not None:
        clip_dir = input_root / args.clip
        if not clip_dir.exists():
            raise FileNotFoundError(f"Clip folder not found: {clip_dir}")
        clip_dirs = [clip_dir]
    else:
        clip_dirs = sorted(
            (p for p in input_root.iterdir() if p.is_dir()),
            key=lambda p: natural_key(p.name),
        )

    if not clip_dirs:
        raise RuntimeError(f"No clip folders found under: {input_root}")

    for clip_dir in clip_dirs:
        process_clip(
            clip_dir=clip_dir,
            output_root=output_root,
            model=model,
            device=device,
            num_queries=args.num_queries,
            image_size=args.image_size,
            seed=args.seed,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()