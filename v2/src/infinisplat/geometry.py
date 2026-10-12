# Modified minimal inference subset. Component provenance: see THIRD_PARTY.md.
from __future__ import annotations
from typing import Optional, Tuple
import torch

def depth_to_camera_coords(depthmap, camera_intrinsics):
    B, H, W = depthmap.shape
    device = depthmap.device
    dtype = depthmap.dtype
    camera_intrinsics = camera_intrinsics.float()
    fx = camera_intrinsics[:, 0, 0]
    fy = camera_intrinsics[:, 1, 1]
    cx = camera_intrinsics[:, 0, 2]
    cy = camera_intrinsics[:, 1, 2]
    v_grid, u_grid = torch.meshgrid(torch.arange(H, dtype=dtype, device=device), torch.arange(W, dtype=dtype, device=device), indexing='ij')
    u_grid = u_grid.unsqueeze(0)
    v_grid = v_grid.unsqueeze(0)
    z_cam = depthmap
    x_cam = (u_grid - cx.view(B, 1, 1)) * z_cam / fx.view(B, 1, 1)
    y_cam = (v_grid - cy.view(B, 1, 1)) * z_cam / fy.view(B, 1, 1)
    X_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    valid_mask = depthmap > 0.0
    return (X_cam, valid_mask)

def depth_to_world_coords_points(depth_map: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor, eps=1e-08) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if depth_map is None:
        return (None, None, None)
    point_mask = depth_map > eps
    camera_points, _ = depth_to_camera_coords(depth_map, intrinsic)
    R_cam_to_world = extrinsic[:, :3, :3]
    t_cam_to_world = extrinsic[:, :3, 3]
    world_coords_points = torch.einsum('bhwi,bji->bhwj', camera_points, R_cam_to_world) + t_cam_to_world[:, None, None, :]
    return (world_coords_points, camera_points, point_mask)
def normalize_all_camera_poses(camera_poses: torch.Tensor, context_nums: int, ref_c2w: Optional[torch.Tensor]=None, median_dist: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if camera_poses.ndim != 4 or camera_poses.shape[-2:] != (4, 4):
        raise ValueError(f'Expected camera_poses [B,V,4,4], got {tuple(camera_poses.shape)}')
    device_type = camera_poses.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        camera_poses_f = camera_poses.float()
        batch_size = camera_poses_f.shape[0]
        w2c_all = torch.linalg.inv(camera_poses_f)
        if ref_c2w is None:
            transform = torch.linalg.inv(w2c_all[:, :1])
        else:
            transform = ref_c2w.to(device=camera_poses_f.device, dtype=torch.float32)
            if transform.ndim == 2:
                transform = transform.view(1, 1, 4, 4)
            elif transform.ndim == 3:
                transform = transform[:, None]
            if transform.ndim != 4 or transform.shape[-2:] != (4, 4):
                raise ValueError(f'Expected ref_c2w broadcastable to [B,1,4,4], got {tuple(transform.shape)}')
            if transform.shape[0] == 1 and batch_size != 1:
                transform = transform.expand(batch_size, -1, -1, -1)
            if transform.shape[0] != batch_size:
                raise ValueError(f'ref_c2w batch mismatch: poses B={batch_size}, ref B={transform.shape[0]}')
            if transform.shape[1] != 1:
                raise ValueError(f'Expected ref_c2w view dimension 1, got {transform.shape[1]}')
        w2c_norm_all = (w2c_all @ transform).float()
        if median_dist is None:
            c2w_norm_ctx = torch.linalg.inv(w2c_norm_all[:, :context_nums])
            dists = c2w_norm_ctx[..., :3, 3].norm(dim=-1)
            median_dist = torch.median(dists).clamp_min(0.1)
        else:
            median_dist = torch.as_tensor(median_dist, device=camera_poses_f.device, dtype=torch.float32).clamp_min(0.1)
        w2c_norm_all = w2c_norm_all.clone()
        scale = median_dist
        if scale.numel() == 1:
            scale = scale.reshape(1, 1, 1)
        elif scale.ndim == 1 and scale.shape[0] == batch_size:
            scale = scale.view(batch_size, 1, 1)
        else:
            raise ValueError(f'Expected scalar median_dist or [B], got {tuple(median_dist.shape)}')
        w2c_norm_all[..., :3, 3] = w2c_norm_all[..., :3, 3] / scale
        c2w_norm_all = torch.linalg.inv(w2c_norm_all)
    return (w2c_norm_all, c2w_norm_all, median_dist)

def as_homogeneous_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f'Expected extrinsics [...,3,4] or [...,4,4], got {tuple(extrinsics.shape)}')
    pad = torch.zeros((*extrinsics.shape[:-2], 1, 4), device=extrinsics.device, dtype=extrinsics.dtype)
    pad[..., 0, 3] = 1.0
    return torch.cat([extrinsics, pad], dim=-2)

def fallback_pose_scale(ext_ref: torch.Tensor, ext_est: torch.Tensor) -> torch.Tensor:
    ref_c2w = torch.linalg.inv(ext_ref.float())
    est_c2w = torch.linalg.inv(ext_est.float())
    scales = []
    for ref_centers, est_centers in zip(ref_c2w[..., :3, 3], est_c2w[..., :3, 3]):
        n = ref_centers.shape[0]
        if n < 2:
            scales.append(torch.ones((), device=ext_ref.device, dtype=torch.float32))
            continue
        i, j = torch.triu_indices(n, n, offset=1, device=ext_ref.device)
        ref_dist = (ref_centers[i] - ref_centers[j]).norm(dim=-1)
        est_dist = (est_centers[i] - est_centers[j]).norm(dim=-1)
        valid = est_dist > 1e-06
        if valid.any():
            scales.append(torch.median(ref_dist[valid] / est_dist[valid]).clamp_min(1e-06))
        else:
            scales.append(torch.ones((), device=ext_ref.device, dtype=torch.float32))
    return torch.stack(scales)

def align_depth_to_input_extrinsics(depth: torch.Tensor, pred_extrinsics: Optional[torch.Tensor], input_extrinsics: Optional[torch.Tensor], mode: str='umeyama') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if pred_extrinsics is None or input_extrinsics is None:
        return (depth, None)
    pred_ext = as_homogeneous_extrinsics(pred_extrinsics.detach().float())
    input_ext = as_homogeneous_extrinsics(input_extrinsics.detach().float()).to(pred_ext.device)
    if mode == 'median_baseline':
        pred_c2w = torch.linalg.inv(pred_ext)
        pred_centers = pred_c2w[..., :3, 3]
        pred_distances = (pred_centers - pred_centers[:, :1]).norm(dim=-1)
        scales = pred_distances.median(dim=1).values.clamp_min(0.1)
    elif mode == 'umeyama':
        # Missing dependencies must not silently switch the camera scale algorithm.
        from depth_anything_3.utils.pose_align import batch_align_poses_umeyama
        try:
            _, _, scales = batch_align_poses_umeyama(pred_ext, input_ext)
        except Exception:
            scales = fallback_pose_scale(pred_ext, input_ext)
    else:
        raise ValueError(f"Unknown depth pose alignment mode {mode!r}; expected 'umeyama' or 'median_baseline'.")
    scales = scales.to(device=depth.device, dtype=torch.float32).clamp_min(1e-06)
    view_shape = [depth.shape[0]] + [1] * (depth.ndim - 1)
    return (depth / scales.view(*view_shape), scales)
@torch.no_grad()
def normalize_predicted_point_radius(depth, c2w, intrinsics, valid_mask=None):
    trailing_channel = depth.ndim == 5 and depth.shape[-1] == 1
    value = depth[..., 0] if trailing_channel else depth
    if value.ndim != 4:
        raise ValueError('Point normalization requires depth [B,S,H,W,(1)]')
    b, s, h, w = value.shape
    if c2w.shape != (b, s, 4, 4) or intrinsics.shape != (b, s, 3, 3):
        raise ValueError('Point normalization requires exactly the context camera bundle')
    if valid_mask is not None:
        if valid_mask.ndim == 5 and valid_mask.shape[-1] == 1:
            valid_mask = valid_mask[..., 0]
        if valid_mask.shape != value.shape:
            raise ValueError('Point normalization mask must match context depth')
    with torch.autocast(device_type=value.device.type, enabled=False):
        value = value.float()
        poses = c2w.float()
        if not torch.isfinite(poses).all() or not torch.isfinite(intrinsics).all():
            raise ValueError('Nonfinite point-normalization cameras')
        relative = torch.linalg.inv(poses[:, :1]) @ poses
        means, counts = ([], [])
        for bi in range(b):
            total = value.new_zeros(())
            count = torch.zeros((), device=value.device, dtype=torch.long)
            for vi in range(s):
                points, _, mask = depth_to_world_coords_points(value[bi, vi:vi + 1], relative[bi, vi:vi + 1], intrinsics[bi, vi:vi + 1].float())
                radius = points.norm(dim=-1)
                mask = mask & torch.isfinite(radius) & (radius > 0)
                if valid_mask is not None:
                    mask = mask & valid_mask[bi, vi:vi + 1].bool()
                total += radius.masked_fill(~mask, 0).sum()
                count += mask.sum()
            mean = total / count.clamp_min(1)
            if count == 0 or not torch.isfinite(mean) or mean <= 0:
                raise ValueError('No valid positive context point radius for normalization')
            means.append(mean)
            counts.append(count)
        mean = torch.stack(means)
        relative = relative.clone()
        relative[..., :3, 3] /= mean[:, None, None]
        normalized_depth = value / mean[:, None, None, None]
        if trailing_channel:
            normalized_depth = normalized_depth[..., None]
    return (normalized_depth, relative, mean, torch.stack(counts))

@torch.no_grad()
def align_rdnorm_gt_via_legacy_da3(geometry, radius):
    required = ('alignment_pred_extrinsics', 'alignment_gt_extrinsics', 'alignment_gt_c2w', 'alignment_gt_depth_scale')
    if any((key not in geometry for key in required)):
        raise ValueError('Legacy RDNorm alignment requires same-forward camera metadata')
    n = geometry['alignment_pred_extrinsics'].shape[1]
    dummy = radius.new_ones(radius.shape[0], n, 1, 1)
    _, raw_scale = align_depth_to_input_extrinsics(dummy, geometry['alignment_pred_extrinsics'], geometry['alignment_gt_extrinsics'][:, :n], mode='umeyama')
    if not torch.isfinite(raw_scale).all() or (raw_scale <= 0).any():
        raise ValueError('Invalid legacy DA3 pose scale')
    poses = geometry['alignment_gt_c2w'].float().clone()
    poses[..., :3, 3] *= (raw_scale / radius)[:, None, None]
    baseline = geometry['alignment_gt_depth_scale'].reshape(-1)
    return (poses, raw_scale / (radius * baseline))
