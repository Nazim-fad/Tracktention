import contextlib
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .tracktention import Tracktention


def get_submodule_by_name(model: nn.Module, name: str) -> nn.Module:
    module = model
    for attr in name.split("."):
        module = module[int(attr)] if attr.isdigit() else getattr(module, attr)
    return module


def list_named_modules(model: nn.Module, contains: Optional[str] = None) -> List[str]:
    return [name for name, _ in model.named_modules() if contains is None or contains in name]


class DDColorWithTracktention(nn.Module):
    """
    Wraps a DDColor model and injects Tracktention after selected backbone stages.

    Input:
        video_gray: (B, T, 1, H, W) or (B, T, 3, H, W)
        tracks:     (B, T, M, 2) in image pixel coordinates

    Output:
        video_rgb:  (B, T, C, H, W)
    """

    def __init__(
        self,
        ddcolor_model: nn.Module,
        stage_names: Sequence[str],
        feature_dims: Sequence[int],
        num_heads: Sequence[int],
        num_layers: int = 2,
        sigma: float = 0.5,
    ) -> None:
        super().__init__()

        if not stage_names:
            raise ValueError("stage_names must be non-empty")
        if not (len(stage_names) == len(feature_dims) == len(num_heads)):
            raise ValueError("stage_names, feature_dims, and num_heads must have the same length")

        self.model = ddcolor_model
        self.stage_names = list(stage_names)

        self.tracktentions = nn.ModuleList([
            Tracktention(feature_dim=dim, num_heads=heads, num_layers=num_layers, sigma=sigma)
            for dim, heads in zip(feature_dims, num_heads)
        ])

        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._hook_ctx = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        for idx, stage_name in enumerate(self.stage_names):
            module = get_submodule_by_name(self.model, stage_name)
            self._hook_handles.append(module.register_forward_hook(self._make_stage_hook(idx)))

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def _make_stage_hook(self, stage_idx: int):
        def hook(module: nn.Module, inputs, output):
            if self._hook_ctx is None:
                return output

            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Hooked module '{self.stage_names[stage_idx]}' returned "
                    f"{type(output).__name__}, expected a Tensor."
                )
            if output.ndim != 4:
                raise ValueError(
                    f"Expected 4D feature map, got shape {tuple(output.shape)} "
                    f"from '{self.stage_names[stage_idx]}'."
                )

            B = self._hook_ctx["B"]
            T = self._hook_ctx["T"]
            tracks = self._hook_ctx["tracks"]
            image_hw = self._hook_ctx["image_hw"]

            if output.shape[0] != B * T:
                raise ValueError(
                    f"Expected batch dimension B*T={B*T}, got {output.shape[0]} "
                    f"from '{self.stage_names[stage_idx]}'."
                )

            BT, C, H, W = output.shape

            feat = output.view(B, T, C, H, W).permute(0, 1, 3, 4, 2).contiguous()  # (B, T, H, W, C)
            feat = self.tracktentions[stage_idx](feature_map=feat, tracks=tracks, image_hw=image_hw)

            feat_btchw = feat.permute(0, 1, 4, 2, 3).contiguous().view(B * T, C, H, W)

            # Overwrite the stored encoder hook tensor so the decoder consumes
            # the Tracktention-enhanced features.
            if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "hooks"):
                if stage_idx < len(self.model.encoder.hooks):
                    self.model.encoder.hooks[stage_idx].feature = feat_btchw

            return feat_btchw

        return hook

    @contextlib.contextmanager
    def _set_hook_context(self, B: int, T: int, tracks: torch.Tensor, image_hw: Tuple[int, int]):
        prev = self._hook_ctx
        self._hook_ctx = {"B": B, "T": T, "tracks": tracks, "image_hw": image_hw}
        try:
            yield
        finally:
            self._hook_ctx = prev

    def forward(self, video_gray: torch.Tensor, tracks: torch.Tensor) -> torch.Tensor:
        if video_gray.ndim != 5:
            raise ValueError(f"video_gray must be (B,T,C,H,W), got {tuple(video_gray.shape)}")
        if tracks.ndim != 4 or tracks.size(-1) != 2:
            raise ValueError(f"tracks must be (B,T,M,2), got {tuple(tracks.shape)}")

        B, T, C, H, W = video_gray.shape

        if C == 1:
            video_in = video_gray.repeat(1, 1, 3, 1, 1)
        elif C == 3:
            video_in = video_gray
        else:
            raise ValueError(f"Expected 1 or 3 input channels, got {C}")

        flat_video = video_in.view(B * T, 3, H, W)

        with self._set_hook_context(B=B, T=T, tracks=tracks, image_hw=(H, W)):
            flat_out = self.model(flat_video)

        if not isinstance(flat_out, torch.Tensor) or flat_out.ndim != 4:
            raise TypeError(f"DDColor model must return a 4D Tensor, got {type(flat_out).__name__}")
        if flat_out.shape[0] != B * T:
            raise ValueError(f"Expected output batch dimension {B*T}, got {flat_out.shape[0]}")

        return flat_out.view(B, T, *flat_out.shape[1:])


def build_ddcolor_tracktention_wrapper(
    ddcolor_model: nn.Module,
    stage_names: Sequence[str],
    feature_dims: Sequence[int],
    num_heads: Sequence[int],
    num_layers: int = 2,
    sigma: float = 0.5,
) -> DDColorWithTracktention:
    return DDColorWithTracktention(
        ddcolor_model=ddcolor_model,
        stage_names=stage_names,
        feature_dims=feature_dims,
        num_heads=num_heads,
        num_layers=num_layers,
        sigma=sigma,
    )