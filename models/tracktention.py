import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PointEmbedding(nn.Module):
    """
    Fourier-style embedding for continuous 2D coordinates.
    Maps (..., 2) -> (..., dim_f).
    """
    def __init__(self, dim_f: int):
        super().__init__()
        assert dim_f % 4 == 0, "dim_f must be divisible by 4"
        self.dim_f = dim_f
        half_dim = dim_f // 2

        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[..., 0:1]
        y = coords[..., 1:2]

        x_freq = x * self.inv_freq
        y_freq = y * self.inv_freq

        x_emb = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)
        y_emb = torch.cat([torch.sin(y_freq), torch.cos(y_freq)], dim=-1)

        return torch.cat([x_emb, y_emb], dim=-1)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard 1D sinusoidal positional encoding.
    Expects input of shape (B, T, D).
    """
    def __init__(self, dim_f: int, max_len: int = 1000):
        super().__init__()
        assert dim_f % 2 == 0, "dim_f must be even"
        pe = torch.zeros(max_len, dim_f)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_f, 2, dtype=torch.float32) * (-math.log(10000.0) / dim_f))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(dtype=x.dtype, device=x.device)


class RoPE2D(nn.Module):
    """
    2D rotary position encoding.
    First half of channels encode x, second half encode y.
    """
    def __init__(self, dim_f: int, base: float = 100.0):
        super().__init__()
        assert dim_f % 4 == 0, "RoPE2D requires dim_f divisible by 4"
        self.dim_f = dim_f

        half_dim = dim_f // 2
        quarter_dim = dim_f // 4

        i = torch.arange(quarter_dim, dtype=torch.float32)
        theta = base ** (-2.0 * i / half_dim)
        self.register_buffer("theta", theta, persistent=False)

    def _apply_1d(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # features: (..., N, d/2)
        # positions: (..., N)
        half_dim = features.shape[-1]
        quarter_dim = half_dim // 2

        x = features.reshape(*features.shape[:-1], quarter_dim, 2)

        theta = self.theta.to(device=positions.device, dtype=positions.dtype)
        theta = theta.view(*([1] * positions.ndim), -1)

        angles = positions.unsqueeze(-1) * theta
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)

        x_even = x[..., 0]
        x_odd = x[..., 1]

        out_even = x_even * cos_vals - x_odd * sin_vals
        out_odd = x_even * sin_vals + x_odd * cos_vals

        out = torch.stack([out_even, out_odd], dim=-1)
        return out.reshape(*out.shape[:-2], half_dim)

    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # features: (..., N, D)
        # positions: (..., N, 2)
        half_dim = self.dim_f // 2

        fx = features[..., :half_dim]
        fy = features[..., half_dim:]

        px = positions[..., 0]
        py = positions[..., 1]

        rx = self._apply_1d(fx, px)
        ry = self._apply_1d(fy, py)

        return torch.cat([rx, ry], dim=-1)


class AttentionalSampling(nn.Module):
    """
    Stage 1: sample track features from the dense feature map.

    Q_t = T_t W_Q
    K_t = F_t W_K
    V_t = F_t

    S_t = A_t V_t
    """
    def __init__(self, feature_dim: int, num_heads: int, sigma: float = 0.5, eps: float = 1e-6):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.sigma = sigma
        self.eps = eps

        self.W_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, feature_dim, bias=False)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _spatial_bias(self, track_coords: torch.Tensor, feature_coords: torch.Tensor) -> torch.Tensor:
        # track_coords:   (B, T, M, 2)
        # feature_coords: (B, T, HW, 2)
        diff = track_coords.unsqueeze(3) - feature_coords.unsqueeze(2)   # (B, T, M, HW, 2)
        dist2 = (diff * diff).sum(dim=-1)                                # (B, T, M, HW)
        return -dist2 / (2.0 * self.sigma * self.sigma)

    def forward(
        self,
        feature_map: torch.Tensor,
        track_tokens: torch.Tensor,
        track_coords: torch.Tensor,
        feature_coords: torch.Tensor,
        rope2d: RoPE2D,
    ) -> torch.Tensor:
        # feature_map:    (B, T, HW, D)
        # track_tokens:   (B, T, M, D)
        # track_coords:   (B, T, M, 2)
        # feature_coords: (B, T, HW, 2)

        B, T, HW, D = feature_map.shape
        _, _, M, _ = track_tokens.shape

        Q = self.W_q(track_tokens)      # (B, T, M, D)
        K = self.W_k(feature_map)       # (B, T, HW, D)
        V = feature_map                 # (B, T, HW, D), unprojected

        # Relative spatial information.
        K = rope2d(K, feature_coords)
        V = rope2d(V, feature_coords)

        Q = Q.view(B, T, M, self.num_heads, self.head_dim).transpose(2, 3)   # (B, T, H, M, d)
        K = K.view(B, T, HW, self.num_heads, self.head_dim).transpose(2, 3)  # (B, T, H, HW, d)
        V = V.view(B, T, HW, self.num_heads, self.head_dim).transpose(2, 3)  # (B, T, H, HW, d)

        Q = F.normalize(Q, p=2, dim=-1, eps=self.eps)
        K = F.normalize(K, p=2, dim=-1, eps=self.eps)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale            # (B, T, H, M, HW)
        bias = self._spatial_bias(track_coords, feature_coords).unsqueeze(2)  # (B, T, 1, M, HW)
        scores = scores + bias

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)                                            # (B, T, H, M, d)
        out = out.transpose(2, 3).contiguous().view(B, T, M, D)               # (B, T, M, D)

        return out


class TrackTransformer(nn.Module):
    """
    Stage 2: temporal transformer applied independently per track.

    Input:  (B, T, M, D)
    Output: (B, T, M, D)
    """
    def __init__(self, feature_dim: int, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.pos_encoding = SinusoidalPositionalEncoding(feature_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=4 * feature_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, track_tokens: torch.Tensor) -> torch.Tensor:
        # (B, T, M, D) -> (B, M, T, D) -> (B*M, T, D)
        B, T, M, D = track_tokens.shape
        x = track_tokens.permute(0, 2, 1, 3).contiguous().view(B * M, T, D)

        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = self.norm(x)

        x = x.view(B, M, T, D).permute(0, 2, 1, 3).contiguous()
        return x


class AttentionalSplatting(nn.Module):
    """
    Stage 3: splat updated track tokens back onto the dense feature map.

    Q_t = G_t W_Q          where G_t are grid-coordinate tokens
    K_t = T'_t W_K
    V_t = T'_t

    U_t = A'_t V_t
    Tracktention(F)_t = W_out U_t
    """
    def __init__(self, feature_dim: int, num_heads: int, sigma: float = 0.5, eps: float = 1e-6):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.sigma = sigma
        self.eps = eps

        self.W_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_out = nn.Linear(feature_dim, feature_dim)

        nn.init.zeros_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _spatial_bias(self, track_coords: torch.Tensor, feature_coords: torch.Tensor) -> torch.Tensor:
        diff = track_coords.unsqueeze(3) - feature_coords.unsqueeze(2)   # (B, T, M, HW, 2)
        dist2 = (diff * diff).sum(dim=-1)                                # (B, T, M, HW)
        return -dist2 / (2.0 * self.sigma * self.sigma)

    def forward(
        self,
        updated_track_tokens: torch.Tensor,
        feature_coords: torch.Tensor,
        track_coords: torch.Tensor,
        point_embedder: PointEmbedding,
        rope2d: RoPE2D,
    ) -> torch.Tensor:
        # updated_track_tokens: (B, T, M, D)
        # feature_coords:       (B, T, HW, 2)
        # track_coords:         (B, T, M, 2)

        B, T, M, D = updated_track_tokens.shape
        _, _, HW, _ = feature_coords.shape

        grid_tokens = point_embedder(feature_coords)   # (B, T, HW, D)

        Q = self.W_q(grid_tokens)                      # (B, T, HW, D)
        K = self.W_k(updated_track_tokens)            # (B, T, M, D)
        V = updated_track_tokens                      # (B, T, M, D), unprojected

        Q = rope2d(Q, feature_coords)
        K = rope2d(K, track_coords)
        V = rope2d(V, track_coords)

        Q = Q.view(B, T, HW, self.num_heads, self.head_dim).transpose(2, 3)  # (B, T, H, HW, d)
        K = K.view(B, T, M, self.num_heads, self.head_dim).transpose(2, 3)   # (B, T, H, M, d)
        V = V.view(B, T, M, self.num_heads, self.head_dim).transpose(2, 3)   # (B, T, H, M, d)

        Q = F.normalize(Q, p=2, dim=-1, eps=self.eps)
        K = F.normalize(K, p=2, dim=-1, eps=self.eps)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale            # (B, T, H, HW, M)

        bias = self._spatial_bias(track_coords, feature_coords)                # (B, T, M, HW)
        bias = bias.transpose(2, 3).unsqueeze(2)                               # (B, T, 1, HW, M)
        scores = scores + bias

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)                                            # (B, T, H, HW, d)
        out = out.transpose(2, 3).contiguous().view(B, T, HW, D)              # (B, T, HW, D)

        return self.W_out(out)


class Tracktention(nn.Module):
    """
    Full Tracktention layer.

    Input:
        feature_map:
            (T, H, W, D),
            (B, T, H, W, D),
            (T, HW, D)    with feature_hw,
            (B, T, HW, D) with feature_hw

        tracks:
            (T, M, 2) or (B, T, M, 2)

    Output:
        same shape as feature_map
    """
    def __init__(self, feature_dim: int = 768, num_heads: int = 8, num_layers: int = 2, sigma: float = 0.5):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        assert feature_dim % 4 == 0, "feature_dim must be divisible by 4"

        self.feature_dim = feature_dim

        self.point_embedder = PointEmbedding(dim_f=feature_dim)
        self.rope2d = RoPE2D(dim_f=feature_dim)

        self.sampler = AttentionalSampling(
            feature_dim=feature_dim,
            num_heads=num_heads,
            sigma=sigma,
        )
        self.track_transformer = TrackTransformer(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.splatter = AttentionalSplatting(
            feature_dim=feature_dim,
            num_heads=num_heads,
            sigma=sigma,
        )

    @staticmethod
    def make_feature_coords(batch_size: int, T: int, feature_hw, device, dtype):
        H, W = feature_hw
        ys = torch.arange(H, device=device, dtype=dtype)
        xs = torch.arange(W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)      # (HW, 2)
        coords = coords.view(1, 1, H * W, 2).expand(batch_size, T, -1, -1)  # (B, T, HW, 2)
        return coords.contiguous()

    @staticmethod
    def scale_tracks_to_feature_grid(tracks: torch.Tensor, image_hw, feature_hw):
        img_h, img_w = image_hw
        feat_h, feat_w = feature_hw

        x = tracks[..., 0].clamp(0, img_w - 1)
        y = tracks[..., 1].clamp(0, img_h - 1)

        x = x * (feat_w - 1) / max(img_w - 1, 1)
        y = y * (feat_h - 1) / max(img_h - 1, 1)

        return torch.stack([x, y], dim=-1)

    def _flatten_feature_map(self, feature_map: torch.Tensor, feature_hw=None):
        """
        Returns:
            F_flat: (B, T, HW, D)
            feature_hw: (H, W)
            had_batch: bool
            was_spatial: bool  # True if input was (.., H, W, D)
        """
        if feature_map.ndim == 5:
            # (B, T, H, W, D)
            B, T, H, W, D = feature_map.shape
            assert D == self.feature_dim
            return feature_map.view(B, T, H * W, D), (H, W), True, True

        if feature_map.ndim == 4 and feature_hw is None:
            # (T, H, W, D)
            T, H, W, D = feature_map.shape
            assert D == self.feature_dim
            return feature_map.view(1, T, H * W, D), (H, W), False, True

        if feature_map.ndim == 4 and feature_hw is not None:
            # (B, T, HW, D)
            B, T, HW, D = feature_map.shape
            H, W = feature_hw
            assert D == self.feature_dim
            assert H * W == HW
            return feature_map, (H, W), True, False

        if feature_map.ndim == 3 and feature_hw is not None:
            # (T, HW, D)
            T, HW, D = feature_map.shape
            H, W = feature_hw
            assert D == self.feature_dim
            assert H * W == HW
            return feature_map.unsqueeze(0), (H, W), False, False

        raise ValueError("feature_map must be (T,H,W,D), (B,T,H,W,D), (T,HW,D) or (B,T,HW,D)")

    def _prepare_tracks(self, tracks: torch.Tensor, batch_size: int):
        if tracks.ndim == 4:
            return tracks
        if tracks.ndim == 3:
            return tracks.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        raise ValueError("tracks must have shape (T,M,2) or (B,T,M,2)")

    def forward(self, feature_map: torch.Tensor, tracks: torch.Tensor, feature_hw=None, image_hw=None):
        F_flat, feature_hw, had_batch, was_spatial = self._flatten_feature_map(feature_map, feature_hw)
        B, T, HW, D = F_flat.shape

        track_coords = self._prepare_tracks(tracks, B)
        if image_hw is not None:
            track_coords = self.scale_tracks_to_feature_grid(track_coords, image_hw, feature_hw)

        feature_coords = self.make_feature_coords(
            batch_size=B,
            T=T,
            feature_hw=feature_hw,
            device=F_flat.device,
            dtype=F_flat.dtype,
        )

        track_tokens = self.point_embedder(track_coords)  # (B, T, M, D)

        sampled_tokens = self.sampler(
            feature_map=F_flat,
            track_tokens=track_tokens,
            track_coords=track_coords,
            feature_coords=feature_coords,
            rope2d=self.rope2d,
        )

        updated_tokens = self.track_transformer(sampled_tokens)

        feature_updates = self.splatter(
            updated_track_tokens=updated_tokens,
            feature_coords=feature_coords,
            track_coords=track_coords,
            point_embedder=self.point_embedder,
            rope2d=self.rope2d,
        )

        out = F_flat + feature_updates

        H, W = feature_hw
        if was_spatial:
            out = out.view(B, T, H, W, D)
        if not had_batch:
            out = out.squeeze(0)

        return out