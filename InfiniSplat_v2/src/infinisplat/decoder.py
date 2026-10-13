# Modified minimal inference subset. Component provenance: see THIRD_PARTY.md.
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

def depth_only(head, feats, H, W):
    """Released DualDPT main branch, with the unused ray branch omitted.

    Keep one view per chunk and the exact upstream FP32 operations. CameraDec
    uses camera tokens independently and discards the ray outputs in this mode.
    """
    from addict import Dict
    from depth_anything_3.model.utils.head_utils import custom_interpolate as resize
    B, S, N, C = feats[0][0].shape
    flat = [feat[0].reshape(B * S, N, C) for feat in feats]
    depths = []
    ph, pw = H // head.patch_size, W // head.patch_size
    for index in range(B * S):
        pyramid = []
        for stage, take in enumerate(head.intermediate_layer_idx):
            x = head.norm(flat[take][index:index+1])
            x = x.permute(0, 2, 1).reshape(1, C, ph, pw)
            x = head.projects[stage](x)
            if head.pos_embed:
                x = head._add_pos_embed(x, W, H)
            pyramid.append(head.resize_layers[stage](x))
        l1, l2, l3, l4 = [getattr(head.scratch, f'layer{i+1}_rn')(x)
                           for i, x in enumerate(pyramid)]
        out = head.scratch.refinenet4(l4, size=l3.shape[2:])
        out = head.scratch.refinenet3(out, l3, size=l2.shape[2:])
        out = head.scratch.refinenet2(out, l2, size=l1.shape[2:])
        out = head.scratch.refinenet1(out, l1)
        out = head.scratch.output_conv1(out)
        out = resize(out, (int(ph * head.patch_size / head.down_ratio),
                           int(pw * head.patch_size / head.down_ratio)),
                     mode='bilinear', align_corners=True)
        if head.pos_embed:
            out = head._add_pos_embed(out, W, H)
        logits = head.scratch.output_conv2(out).permute(0, 2, 3, 1)
        depths.append(head._apply_activation_single(logits[..., :-1], head.activation).squeeze(-1))
    depth = torch.cat(depths, 0)
    return Dict(depth=depth.reshape(B, S, *depth.shape[1:]))

def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float=100) -> torch.Tensor:
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    assert embed_dim % 2 == 0
    device = pos_grid.device
    pos_flat = pos_grid.reshape(-1, grid_dim)
    omega = torch.arange(embed_dim // 4, dtype=torch.float32 if device.type == 'mps' else torch.double, device=device)
    omega /= embed_dim / 4.0
    omega = 1.0 / omega_0 ** omega
    pos_x = pos_flat[:, 0].reshape(-1)
    pos_y = pos_flat[:, 1].reshape(-1)
    out_x = torch.einsum('m,d->md', pos_x, omega)
    out_y = torch.einsum('m,d->md', pos_y, omega)
    emb_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)
    emb_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    emb = torch.cat([emb_x, emb_y], dim=-1)
    return emb.float().view(H, W, embed_dim)

def create_uv_grid(width: int, height: int, aspect_ratio: float=None, dtype: torch.dtype=None, device: torch.device=None) -> torch.Tensor:
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag_factor = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing='xy')
    uv_grid = torch.stack((uu, vv), dim=-1)
    return uv_grid
def _make_fusion_block(features: int, size: int=None, has_residual: bool=True, groups: int=1) -> nn.Module:
    return FeatureFusionBlock(features, nn.ReLU(inplace=True), deconv=False, bn=False, expand=False, align_corners=True, size=size, has_residual=has_residual, groups=groups)

def _make_scratch(in_shape: List[int], out_shape: int, groups: int=1, expand: bool=False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch

class ResidualConvUnit(nn.Module):

    def __init__(self, features, activation, bn, groups=1):
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.norm1 = None
        self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        return self.skip_add.add(out, x)

class FeatureFusionBlock(nn.Module):

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None, has_residual=True, groups=1):
        super(FeatureFusionBlock, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups)
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)
        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        if size is None and self.size is None:
            modifier = {'scale_factor': 2}
        elif size is None:
            modifier = {'size': self.size}
        else:
            modifier = {'size': size}
        output = custom_interpolate(output, **modifier, mode='bilinear', align_corners=self.align_corners)
        output = self.out_conv(output)
        return output

def custom_interpolate(x: torch.Tensor, size: Tuple[int, int]=None, scale_factor: float=None, mode: str='bilinear', align_corners: bool=True) -> torch.Tensor:
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    INT_MAX = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=input_elements // INT_MAX + 1, dim=0)
        interpolated_chunks = [nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
class GSFeatHead(nn.Module):

    def __init__(self, dim_in: int, patch_size: int=14, features: int=256, out_channels: List[int]=[256, 512, 1024, 1024], pos_embed: bool=True, down_ratio: int=1) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels])
        self.resize_layers = nn.ModuleList([nn.ConvTranspose2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0), nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0), nn.Identity(), nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1)])
        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        head_features_1 = features
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        conv2_in_channels = head_features_1 // 2
        self.input_merger = nn.Sequential(nn.Conv2d(3, conv2_in_channels, 7, 1, 3), nn.ReLU())

    def forward(self, token_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frames_chunk_size: int=8, token_grid_hw=None) -> torch.Tensor:
        B, S, _, H, W = images.shape
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(token_list, images, patch_start_idx, token_grid_hw=token_grid_hw)
        assert frames_chunk_size > 0
        gs_chunks = []
        for frame_start in range(0, S, frames_chunk_size):
            frame_end = min(frame_start + frames_chunk_size, S)
            gs = self._forward_impl(token_list, images, patch_start_idx, frame_start, frame_end, token_grid_hw=token_grid_hw)
            gs_chunks.append(gs)
        return torch.cat(gs_chunks, dim=1)

    def forward_chunk(self, token_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frame_start: int, frame_end: int, token_grid_hw=None) -> torch.Tensor:
        return self._forward_impl(token_list, images, patch_start_idx, frame_start, frame_end, token_grid_hw=token_grid_hw)

    def _forward_impl(self, token_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frame_start: int=None, frame_end: int=None, token_grid_hw=None) -> torch.Tensor:
        if frame_start is not None and frame_end is not None:
            images = images[:, frame_start:frame_end].contiguous()
        B, S, _, H, W = images.shape
        token_h, token_w = (H, W) if token_grid_hw is None else tuple(token_grid_hw)
        if token_h <= 0 or token_w <= 0:
            raise ValueError(f'token_grid_hw must be positive, got {(token_h, token_w)}')
        ph = token_h // self.patch_size
        pw = token_w // self.patch_size
        expected_patch_tokens = ph * pw
        feats = []
        for proj, resize, tokens in zip(self.projects, self.resize_layers, token_list):
            patch_tokens = tokens[:, :, patch_start_idx:]
            if frame_start is not None and frame_end is not None:
                patch_tokens = patch_tokens[:, frame_start:frame_end]
            if patch_tokens.shape[-2] != expected_patch_tokens:
                raise ValueError(f'GSFeatHead token-grid mismatch: token_grid_hw={(token_h, token_w)} with patch_size={self.patch_size} expects {expected_patch_tokens} patch tokens, but received {patch_tokens.shape[-2]}')
            patch_tokens = patch_tokens.reshape(B * S, -1, patch_tokens.shape[-1])
            patch_tokens = self.norm(patch_tokens)
            feat = patch_tokens.permute(0, 2, 1).reshape(B * S, patch_tokens.shape[-1], ph, pw)
            feat = proj(feat)
            if self.pos_embed:
                feat = self._apply_pos_embed(feat, token_w, token_h)
            feat = resize(feat)
            feats.append(feat)
        fused = self.scratch_forward(feats)
        _interpolate_fn = lambda t: custom_interpolate(t, size=(H, W), mode='bilinear', align_corners=True)
        fused = _interpolate_fn(fused)
        if self.pos_embed:
            fused = self._apply_pos_embed(fused, W, H)
        img_flat = images.reshape(B * S, -1, H, W)
        img_feat = self.input_merger(img_flat)
        fused = fused + img_feat
        fused = fused.reshape(B, S, *fused.shape[1:])
        return fused

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float=0.1) -> torch.Tensor:
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2
        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1
        out = self.scratch.output_conv1(out)
        return out
