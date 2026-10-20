"""CT imaging branch: patch backbone, multiple-instance attention pooling and masked-patch prediction.

A CT examination is represented by ``N`` cubic patches of ``64^3`` voxels at 1 mm spacing. Every
patch is embedded by a 3D backbone,

* ``"swin_vit"`` - the Swin-ViT encoder of MONAI's Swin UNETR (``feature_size=48``), optionally
  initialised from the a caller-supplied local state, or
* ``"cnn3d"`` - a compact 3D residual network for fast experiments,

projected to the patch embedding ``u_ij`` and pooled into the scan embedding ``h_img`` by
attention-based multiple-instance pooling ``a_ij = softmax_j(w^T tanh(W u_ij))``. An optional
masked-patch prediction head predicts a low-resolution copy of patches whose voxel blocks
were partially hidden, providing a self-supervised objective.
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from molt.models.common import AttentionPool, make_mlp
from molt.models.pathway import masked_mse
from molt.utils.logging import get_logger

__all__ = [
    "Backbone",
    "CNN3DBackbone",
    "ImagingEncoder",
    "ResidualBlock3d",
    "SwinViTBackbone",
    "load_swinvit_weights",
    "read_swinvit_checkpoint",
    "remap_swinvit_key",
]

Backbone = Literal["swin_vit", "cnn3d"]
_LOG = get_logger(__name__)


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock3d(nn.Module):
    """Two 3x3x3 convolutions with GroupNorm and a projection shortcut when needed."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _group_norm(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = _group_norm(out_ch)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False), _group_norm(out_ch)
            )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block."""
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class CNN3DBackbone(nn.Module):
    """Compact 3D residual network mapping ``(M, C, 64, 64, 64)`` patches to ``(M, F)`` features.

    Args:
        in_channels: Input channels.
        widths: Channel widths; the stem uses ``widths[0]`` and each further width adds a
            stride-2 stage.
        blocks_per_stage: Residual blocks per stage.
    """

    def __init__(
        self,
        in_channels: int = 1,
        widths: Sequence[int] = (32, 64, 128, 256),
        blocks_per_stage: int = 1,
    ) -> None:
        super().__init__()
        widths = list(widths)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, widths[0], 3, stride=2, padding=1, bias=False),
            _group_norm(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        prev = widths[0]
        for width in widths[1:]:
            blocks = [ResidualBlock3d(prev, width, stride=2)]
            blocks += [ResidualBlock3d(width, width) for _ in range(blocks_per_stage - 1)]
            stages.append(nn.Sequential(*blocks))
            prev = width
        self.stages = nn.Sequential(*stages)
        self.out_dim = prev

    def forward(self, x: Tensor) -> Tensor:
        """Return globally average-pooled features ``(M, F)``."""
        return self.stages(self.stem(x)).mean(dim=(2, 3, 4))


_SWIN_KEY_RULES: tuple[tuple[str, str], ...] = (
    (r"^(module\.)+", ""),
    (r"^swinViT\.", ""),
    (r"^encoder\.", ""),
    (r"\.mlp\.fc1\.", ".mlp.linear1."),
    (r"\.mlp\.fc2\.", ".mlp.linear2."),
)


def remap_swinvit_key(key: str) -> str:
    """Map a key of a Swin-ViT state to MONAI's ``SwinTransformer`` naming.

    ``module.`` / ``swinViT.`` prefixes are removed and ``mlp.fc1``/``mlp.fc2`` become
    ``mlp.linear1``/``mlp.linear2``.
    """
    for pattern, repl in _SWIN_KEY_RULES:
        key = re.sub(pattern, repl, key)
    return key


def _checkpoint_key(path: str | os.PathLike[str]) -> tuple[str, int, int]:
    resolved = os.path.realpath(os.fspath(path))
    stat = os.stat(resolved)
    return resolved, int(stat.st_size), int(stat.st_mtime_ns)


@functools.lru_cache(maxsize=4)
def _read_checkpoint_cached(key: tuple[str, int, int]) -> dict[str, Tensor]:
    path = key[0]
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - checkpoints with optimiser state need full unpickling
        raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for name in ("state_dict", "model", "net"):
            if name in raw and isinstance(raw[name], dict):
                raw = raw[name]
                break
    if not isinstance(raw, dict):
        raise TypeError(f"{path} does not contain a state dict")
    _LOG.info("Read Swin-ViT checkpoint %s (%d entries)", path, len(raw))
    return {k: v for k, v in raw.items() if isinstance(v, Tensor)}


def read_swinvit_checkpoint(path: str | os.PathLike[str]) -> dict[str, Tensor]:
    """CPU state dict of a Swin-ViT checkpoint, read once per process.

    The dictionary is cached per resolved path, size and modification time, so the many
    model builds of a nested cross-validation run read the file once; a changed file is read
    again. The cached tensors are shared and must not be modified in place
    (:meth:`torch.nn.Module.load_state_dict` copies them into the module).
    """
    return _read_checkpoint_cached(_checkpoint_key(path))


def load_swinvit_weights(
    module: nn.Module, path: str | os.PathLike[str], *, skip_mismatched: bool = True
) -> dict[str, Any]:
    """Load Swin-ViT weights from a checkpoint into a MONAI ``SwinTransformer``.

    Accepts a raw state dict or a dict with a ``"state_dict"`` (or ``"model"``) entry. Keys are
    remapped with :func:`remap_swinvit_key`; keys absent from ``module`` (e.g. pretraining heads)
    are skipped, as are tensors whose shapes differ unless ``skip_mismatched`` is false.

    Args:
        module: Target ``SwinTransformer``.
        path: Checkpoint file.
        skip_mismatched: Skip tensors whose shape does not match instead of raising.

    Returns:
        A report with the numbers of loaded, skipped and missing tensors.
    """
    raw = read_swinvit_checkpoint(path)
    target = module.state_dict()
    loadable: dict[str, Tensor] = {}
    skipped: list[str] = []
    for key, value in raw.items():
        if not isinstance(value, Tensor):
            continue
        new_key = remap_swinvit_key(key)
        if new_key not in target:
            skipped.append(key)
            continue
        if target[new_key].shape != value.shape:
            if not skip_mismatched:
                raise ValueError(f"Shape mismatch for {new_key}: {tuple(value.shape)}")
            skipped.append(key)
            continue
        loadable[new_key] = value
    module.load_state_dict(loadable, strict=False)
    missing = sorted(set(target) - set(loadable))
    return {"loaded": len(loadable), "skipped": skipped, "missing": missing, "path": str(path)}


class SwinViTBackbone(nn.Module):
    """MONAI Swin-ViT encoder producing pooled patch features.

    Args:
        in_channels: Input channels.
        feature_size: Embedding size of the first stage (48 for the public weights).
        checkpoint: Optional path to a caller-supplied local state; if absent the encoder keeps its
            random initialisation and a warning is logged.
        multiscale: Concatenate globally pooled features of the last two stages
            (``8 + 16`` x ``feature_size``) instead of the last stage only.
        use_checkpoint: Enable MONAI's internal gradient checkpointing.
        drop_path_rate: Stochastic-depth rate.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_size: int = 48,
        checkpoint: str | os.PathLike[str] | None = None,
        multiscale: bool = True,
        use_checkpoint: bool = False,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        try:
            from monai.networks.nets.swin_unetr import SwinTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'swin_vit' backbone requires MONAI and einops (pip install 'molt[imaging]')"
            ) from exc
        self.swin = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
        )
        self.multiscale = bool(multiscale)
        self.out_dim = feature_size * (16 + 8 if multiscale else 16)
        self.load_report: dict[str, Any] | None = None
        if checkpoint is not None:
            if os.path.isfile(checkpoint):
                self.load_report = load_swinvit_weights(self.swin, checkpoint)
                _LOG.info(
                    "Loaded %d Swin-ViT tensors from %s (%d missing)",
                    self.load_report["loaded"],
                    checkpoint,
                    len(self.load_report["missing"]),
                )
            else:
                raise FileNotFoundError(f"Local Swin-ViT state does not exist: {checkpoint}")

    def forward(self, x: Tensor) -> Tensor:
        """Return pooled features ``(M, out_dim)`` for patches ``(M, C, D, H, W)``."""
        hidden = self.swin(x, normalize=True)
        feats = [hidden[-1].mean(dim=(2, 3, 4))]
        if self.multiscale:
            feats.insert(0, hidden[-2].mean(dim=(2, 3, 4)))
        return torch.cat(feats, dim=1)


class ImagingEncoder(nn.Module):
    """Encode a set of CT patches per scan into a scan embedding ``h_img``.

    Args:
        backbone: ``"swin_vit"`` or ``"cnn3d"``.
        embed_dim: Patch/scan embedding dimension.
        mil_hidden: Width of the attention-pooling scoring network.
        mil_gated: Use the sigmoid-gated attention variant.
        dropout: Dropout on patch embeddings.
        swin_checkpoint: Optional path to Swin-ViT weights.
        swin_feature_size: Swin-ViT ``feature_size``.
        cnn_widths: Channel widths of the ``cnn3d`` backbone.
        hu_window: If not ``None``, inputs are Hounsfield units and are clipped to this window
            and mapped linearly to ``[0, 1]``; if ``None`` inputs are used unchanged.
        chunk_size: Maximum number of patches passed through the backbone at once.
        grad_checkpoint: Recompute backbone activations chunk by chunk in the backward pass to
            bound memory (useful when fine-tuning the Swin-ViT on many patches).
        recon: Enable the masked-patch prediction head.
        recon_patches: Patches per scan used for the masked-patch objective.
        recon_mask_ratio: Fraction of ``8^3`` voxel blocks hidden in the masked-patch inputs.
        recon_grid: Resolution of the low-resolution prediction target.
    """

    def __init__(
        self,
        backbone: Backbone = "cnn3d",
        embed_dim: int = 128,
        mil_hidden: int = 128,
        mil_gated: bool = False,
        dropout: float = 0.1,
        swin_checkpoint: str | os.PathLike[str] | None = None,
        swin_feature_size: int = 48,
        cnn_widths: Sequence[int] = (32, 64, 128, 256),
        hu_window: tuple[float, float] | None = (-1024.0, 600.0),
        chunk_size: int = 64,
        grad_checkpoint: bool = False,
        recon: bool = False,
        recon_patches: int = 4,
        recon_mask_ratio: float = 0.5,
        recon_grid: int = 8,
    ) -> None:
        super().__init__()
        if backbone == "swin_vit":
            self.backbone: nn.Module = SwinViTBackbone(
                feature_size=swin_feature_size, checkpoint=swin_checkpoint
            )
        elif backbone == "cnn3d":
            self.backbone = CNN3DBackbone(widths=cnn_widths)
        else:
            raise ValueError(f"Unknown CT backbone {backbone!r}")
        self.backbone_name = backbone
        self.feature_dim = int(self.backbone.out_dim)
        self.embed_dim = int(embed_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = AttentionPool(embed_dim, hidden=mil_hidden, gated=mil_gated)
        self.hu_window = tuple(hu_window) if hu_window is not None else None
        self._range_checked = False
        self.chunk_size = max(1, int(chunk_size))
        self.grad_checkpoint = bool(grad_checkpoint)
        self.recon_grid = int(recon_grid)
        self.recon_patches = int(recon_patches)
        self.recon_mask_ratio = float(recon_mask_ratio)
        self.recon_head = (
            make_mlp(embed_dim, [512], self.recon_grid**3, dropout=0.0) if recon else None
        )

    @property
    def out_dim(self) -> int:
        """Dimension of the scan embedding."""
        return self.embed_dim

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze (``False``) or unfreeze (``True``) the patch backbone."""
        for param in self.backbone.parameters():
            param.requires_grad_(trainable)

    def normalize(self, x: Tensor) -> Tensor:
        """Map raw patch intensities to the backbone input range."""
        x = x.float()
        if self.hu_window is None:
            return x
        lo, hi = self.hu_window
        if not self._range_checked and x.numel():
            self._range_checked = True
            if float(x.max() - x.min()) < 50.0:
                _LOG.warning(
                    "CT patch intensities span less than 50 units although hu_window=%s expects "
                    "Hounsfield units; set hu_window=None for pre-normalised patches",
                    self.hu_window,
                )
        return (x.clamp(lo, hi) - lo) / (hi - lo)

    def _backbone_needs_grad(self) -> bool:
        return torch.is_grad_enabled() and any(p.requires_grad for p in self.backbone.parameters())

    def patch_features(self, patches: Tensor, normalized: bool = False) -> Tensor:
        """Backbone features for flat patches ``(M, C, D, H, W)`` -> ``(M, feature_dim)``.

        Patches are processed in chunks of ``chunk_size``; when the backbone is trained and
        ``grad_checkpoint`` is set, each chunk is recomputed during the backward pass.
        """
        x = patches if normalized else self.normalize(patches)
        if x.shape[0] == 0:
            return x.new_zeros(0, self.feature_dim)
        needs_grad = self._backbone_needs_grad()
        outputs = []
        for chunk in torch.split(x, self.chunk_size, dim=0):
            if needs_grad and self.grad_checkpoint and self.training:
                outputs.append(grad_checkpoint(self.backbone, chunk, use_reentrant=False))
            else:
                outputs.append(self.backbone(chunk))
        return torch.cat(outputs, dim=0).float()

    @torch.no_grad()
    def extract_features(self, patches: Tensor) -> Tensor:
        """Precompute backbone features ``(B, N, feature_dim)`` for patches ``(B, N, C, D, H, W)``.

        The result can be passed to :meth:`forward` as ``features`` to train the rest of the
        network without re-running a frozen backbone.
        """
        b, n = patches.shape[:2]
        flat = patches.reshape(b * n, *patches.shape[2:])
        return self.patch_features(flat).view(b, n, -1)

    def embed_features(self, features: Tensor) -> Tensor:
        """Project backbone features ``(..., feature_dim)`` to patch embeddings ``u``."""
        return self.proj(features)

    def forward(
        self,
        patches: Tensor | None,
        present: Tensor,
        patch_mask: Tensor | None = None,
        features: Tensor | None = None,
        compute_recon: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode the scans of a batch.

        Only rows with ``present`` true are passed through the backbone; other rows receive a
        zero embedding and zero attention.

        Args:
            patches: ``(B, N, C, 64, 64, 64)`` patches (ignored when ``features`` is given).
            present: ``(B,)`` boolean CT availability.
            patch_mask: Optional ``(B, N)`` boolean mask of valid (non-padding) patches.
            features: Optional precomputed backbone features ``(B, N, feature_dim)``.
            compute_recon: Also compute the masked-patch prediction loss.

        Returns:
            ``h_img`` ``(B, embed_dim)``, attention weights ``(B, N)`` and the masked-patch
            loss (scalar; zero if not computed).
        """
        source = features if features is not None else patches
        if source is None:
            raise ValueError("Either patches or features must be provided")
        device = self.pool.score.weight.device
        b, n = source.shape[:2]
        present = present.to(device=device, dtype=torch.bool)
        h = torch.zeros(b, self.embed_dim, device=device)
        attn = torch.zeros(b, n, device=device)
        recon = h.new_zeros(())
        idx = torch.nonzero(present, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return h, attn, recon + self.pool.score.weight.sum() * 0.0
        if features is not None:
            feats = features.to(device)[idx].float()
        else:
            sel = patches.to(device)[idx]
            flat = sel.reshape(idx.numel() * n, *sel.shape[2:])
            feats = self.patch_features(flat).view(idx.numel(), n, -1)
        u = self.embed_features(feats)
        mask = None if patch_mask is None else patch_mask.to(device=device, dtype=torch.bool)[idx]
        pooled, weights = self.pool(u, mask)
        h = h.index_copy(0, idx, pooled.to(h.dtype))
        attn = attn.index_copy(0, idx, weights.to(attn.dtype))
        if compute_recon and self.recon_head is not None and patches is not None:
            recon = self.masked_patch_loss(patches.to(device)[idx], mask)
        return h, attn, recon

    def masked_patch_loss(self, patches: Tensor, patch_mask: Tensor | None = None) -> Tensor:
        """Masked-patch prediction loss on a random subset of patches of each scan.

        Each selected patch is divided into ``recon_grid^3`` voxel blocks, a fraction
        ``recon_mask_ratio`` of the blocks is set to zero, and the head predicts the block means
        of the clean patch from the embedding of the masked patch; the loss is the MSE on the
        hidden blocks.

        Args:
            patches: ``(B, N, C, D, H, W)`` raw patches.
            patch_mask: Optional ``(B, N)`` validity mask.
        """
        if self.recon_head is None:
            raise RuntimeError("Masked-patch head is disabled (recon=False)")
        b, n = patches.shape[:2]
        k = min(self.recon_patches, n)
        scores = torch.rand(b, n, device=patches.device)
        if patch_mask is not None:
            scores = scores.masked_fill(~patch_mask, -1.0)
        pick = scores.topk(k, dim=1).indices
        chosen = patches[torch.arange(b, device=patches.device).unsqueeze(1), pick]
        chosen = self.normalize(chosen.reshape(b * k, *patches.shape[2:]))
        g = self.recon_grid
        target = F.adaptive_avg_pool3d(chosen.mean(dim=1, keepdim=True), g).flatten(1)
        hidden = torch.rand(b * k, 1, g, g, g, device=chosen.device) < self.recon_mask_ratio
        voxel_mask = F.interpolate(hidden.float(), size=chosen.shape[2:], mode="nearest")
        masked_input = chosen * (1.0 - voxel_mask)
        u = self.embed_features(self.patch_features(masked_input, normalized=True))
        pred = self.recon_head(u)
        return masked_mse(pred, target, hidden.flatten(1))
