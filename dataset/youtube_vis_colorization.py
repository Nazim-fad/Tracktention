from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images_in_dir(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    return files


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luma."""
    r, g, b = rgb[0:1], rgb[1:2], rgb[2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


class YouTubeVISColorizationDataset(Dataset):
    """
    YouTube-VIS video colorization dataset.

    Directory layout:
        <frames_root>/<split>/JPEGImages/<video_id>/<frame>.jpg
        <tracks_root>/<split>/JPEGImages/<video_id>/tracks.pt

    Each sample returns:
        "gray"        (T, 1, H, W)
        "rgb"         (T, 3, H, W)
        "tracks"      (T, M, 2) or None
        "visibility"  (T, M)    or None
        "clip_name"   str
        "frame_names" list[str]
        "image_hw"    (H, W)
    """

    def __init__(
        self,
        frames_root: str,
        tracks_root: Optional[str],
        split: str = "train",
        clip_len: int = 6,
        image_size: int = 256,
        stride: int = 1,
        require_tracks: bool = True,
    ) -> None:
        super().__init__()

        self.frames_root = Path(frames_root) / split / "JPEGImages"
        self.tracks_root = None if tracks_root is None else Path(tracks_root) / split / "JPEGImages"
        self.split = split
        self.clip_len = clip_len
        self.image_size = image_size
        self.stride = stride
        self.require_tracks = require_tracks

        if not self.frames_root.exists():
            raise FileNotFoundError(f"Frames root does not exist: {self.frames_root}")

        video_dirs = sorted(
            (p for p in self.frames_root.iterdir() if p.is_dir()),
            key=lambda p: natural_key(p.name),
        )

        if not video_dirs:
            raise RuntimeError(f"No video folders found in {self.frames_root}")

        self.samples: List[Dict] = []

        for video_dir in video_dirs:
            frame_files = list_images_in_dir(video_dir)
            if len(frame_files) < clip_len:
                continue

            tracks_file = None
            if self.tracks_root is not None:
                tracks_file = self.tracks_root / video_dir.name / "tracks.pt"

            if self.require_tracks and (tracks_file is None or not tracks_file.exists()):
                continue

            for start in range(0, len(frame_files) - clip_len + 1, stride):
                self.samples.append({
                    "video_dir": video_dir,
                    "frame_files": frame_files,
                    "tracks_file": tracks_file,
                    "start": start,
                    "end": start + clip_len,
                })

        if not self.samples:
            raise RuntimeError(
                f"No valid samples found under {self.frames_root}. "
                f"Check clip_len={clip_len} and track files under {self.tracks_root}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_track_payload(self, tracks_file: Path) -> Dict:
        payload = torch.load(tracks_file, map_location="cpu")
        if "tracks" not in payload:
            raise KeyError(f"'tracks' key not found in {tracks_file}")
        return payload

    def _resize_tracks(
        self,
        tracks: torch.Tensor,
        src_hw: Tuple[int, int],
        dst_hw: Tuple[int, int],
    ) -> torch.Tensor:
        src_h, src_w = src_hw
        dst_h, dst_w = dst_hw
        x = tracks[..., 0] * (dst_w - 1) / max(src_w - 1, 1)
        y = tracks[..., 1] * (dst_h - 1) / max(src_h - 1, 1)
        return torch.stack([x, y], dim=-1)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        video_dir = sample["video_dir"]
        frame_files = sample["frame_files"][sample["start"]:sample["end"]]
        tracks_file = sample["tracks_file"]

        frame_names = [fp.name for fp in frame_files]
        resized_hw = (self.image_size, self.image_size)

        first_img = Image.open(frame_files[0]).convert("RGB")
        orig_hw = (first_img.size[1], first_img.size[0])

        rgb_frames, gray_frames = [], []
        for fp in frame_files:
            img = Image.open(fp).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
            rgb = pil_to_tensor(img)
            rgb_frames.append(rgb)
            gray_frames.append(rgb_to_gray(rgb))

        rgb = torch.stack(rgb_frames, dim=0)
        gray = torch.stack(gray_frames, dim=0)

        tracks = None
        visibility = None

        if tracks_file is not None and tracks_file.exists():
            payload = self._load_track_payload(tracks_file)

            full_tracks = payload["tracks"]
            full_visibility = payload.get("visibility")
            payload_hw = tuple(payload.get("image_hw", orig_hw))

            start, end = sample["start"], sample["end"]
            tracks = self._resize_tracks(full_tracks[start:end], src_hw=payload_hw, dst_hw=resized_hw)

            if full_visibility is not None:
                visibility = full_visibility[start:end]
                if visibility.ndim == 3 and visibility.shape[-1] == 1:
                    visibility = visibility.squeeze(-1)

        return {
            "gray": gray,
            "rgb": rgb,
            "tracks": tracks,
            "visibility": visibility,
            "clip_name": video_dir.name,
            "frame_names": frame_names,
            "image_hw": resized_hw,
        }