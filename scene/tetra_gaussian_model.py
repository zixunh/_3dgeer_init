import os
import json
import numpy as np
import torch
from torch import nn
from plyfile import PlyData, PlyElement

from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud


def _normalize_quat(q):
    return q / torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(1e-12)


def _rotmat_to_quat(rotmat):
    """Convert rotation matrices to [w, x, y, z] quaternions."""
    r = rotmat
    q_abs = torch.sqrt(torch.clamp(torch.stack([
        1.0 + r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2],
        1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2],
        1.0 - r[:, 0, 0] + r[:, 1, 1] - r[:, 2, 2],
        1.0 - r[:, 0, 0] - r[:, 1, 1] + r[:, 2, 2],
    ], dim=1), min=1e-12))

    quat_by_rijk = torch.stack([
        torch.stack([q_abs[:, 0] ** 2, r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]], dim=1),
        torch.stack([r[:, 2, 1] - r[:, 1, 2], q_abs[:, 1] ** 2, r[:, 1, 0] + r[:, 0, 1], r[:, 0, 2] + r[:, 2, 0]], dim=1),
        torch.stack([r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] + r[:, 0, 1], q_abs[:, 2] ** 2, r[:, 2, 1] + r[:, 1, 2]], dim=1),
        torch.stack([r[:, 1, 0] - r[:, 0, 1], r[:, 2, 0] + r[:, 0, 2], r[:, 2, 1] + r[:, 1, 2], q_abs[:, 3] ** 2], dim=1),
    ], dim=1)

    denom = (2.0 * q_abs[:, :, None]).clamp_min(1e-12)
    quat_candidates = quat_by_rijk / denom
    idx = q_abs.argmax(dim=1)
    quat = quat_candidates[torch.arange(rotmat.shape[0], device=rotmat.device), idx]
    return _normalize_quat(quat)


def tetra_to_gaussian_moment_matching(tetra_verts, eta=1.0, eps=1e-4):
    means = tetra_verts.mean(dim=1)
    centered = tetra_verts - means[:, None, :]
    cov = eta * torch.einsum("tki,tkj->tij", centered, centered)
    eye = torch.eye(3, dtype=tetra_verts.dtype, device=tetra_verts.device)[None]
    cov = cov + eps * eye

    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp_min(eps)

    # Make the eigenbasis right-handed before converting to quaternions.
    det = torch.linalg.det(eigvecs)
    eigvecs = eigvecs.clone()
    eigvecs[det < 0, :, 0] *= -1.0

    scales = torch.sqrt(eigvals)
    rotations = _rotmat_to_quat(eigvecs)
    return means, scales, rotations, cov


def tetra_signed_volumes(tetra_verts):
    a = tetra_verts[:, 1] - tetra_verts[:, 0]
    b = tetra_verts[:, 2] - tetra_verts[:, 0]
    c = tetra_verts[:, 3] - tetra_verts[:, 0]
    return torch.einsum("ti,ti->t", torch.cross(a, b, dim=1), c) / 6.0


def _voxel_downsample(points, colors, voxel_size):
    if voxel_size is None or voxel_size <= 0:
        return points, colors
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(coords, axis=0, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    points_ds = np.zeros((count.shape[0], 3), dtype=np.float64)
    colors_ds = np.zeros((count.shape[0], 3), dtype=np.float64)
    np.add.at(points_ds, inverse, points)
    np.add.at(colors_ds, inverse, colors)
    points_ds /= count[:, None]
    colors_ds /= count[:, None]
    return points_ds.astype(np.float32), colors_ds.astype(np.float32)


def _build_delaunay_tetrahedra(points, colors, extent, voxel_size):
    try:
        from scipy.spatial import Delaunay
    except ImportError as exc:
        raise ImportError("TetraGEER requires scipy for --tetra_init delaunay. Install scipy or use model_type=gaussian.") from exc

    points, colors = _voxel_downsample(points, colors, voxel_size)
    if points.shape[0] < 4:
        raise ValueError("Need at least four points to initialize tetrahedra.")

    delaunay = Delaunay(points)
    tets = delaunay.simplices.astype(np.int64)
    tetra_verts = points[tets]
    signed_vols = np.einsum(
        "ti,ti->t",
        np.cross(tetra_verts[:, 1] - tetra_verts[:, 0], tetra_verts[:, 2] - tetra_verts[:, 0]),
        tetra_verts[:, 3] - tetra_verts[:, 0],
    ) / 6.0
    vols = np.abs(signed_vols)

    edges = tetra_verts[:, :, None, :] - tetra_verts[:, None, :, :]
    max_edges = np.linalg.norm(edges, axis=-1).max(axis=(1, 2))
    positive = vols > max(1e-12, np.median(vols[vols > 0]) * 1e-4 if np.any(vols > 0) else 1e-12)
    compact = max_edges < max(float(extent) * 2.5, np.median(max_edges) * 5.0)
    keep = positive & compact
    if keep.sum() == 0:
        keep = positive
    tets = tets[keep]
    if tets.shape[0] == 0:
        raise ValueError("Delaunay tetrahedralization produced no valid tetrahedra after filtering.")
    tetra_verts = points[tets]
    signed_vols = np.einsum(
        "ti,ti->t",
        np.cross(tetra_verts[:, 1] - tetra_verts[:, 0], tetra_verts[:, 2] - tetra_verts[:, 0]),
        tetra_verts[:, 3] - tetra_verts[:, 0],
    ) / 6.0
    neg = signed_vols < 0
    tets[neg, 0], tets[neg, 1] = tets[neg, 1].copy(), tets[neg, 0].copy()
    return points, colors, tets


class TetraGaussianModel:
    def __init__(self, sh_degree):
        self.model_type = "tetra"
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.tetra_vertices = torch.empty(0)
        self.tetra_indices = torch.empty(0, dtype=torch.long)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.exposure_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.tetra_eta = 1.0
        self.tetra_eps = 1e-4
        self.initial_edge_lengths = None
        self.edge_index = None
        self.edge_to_tetra = None
        self.inverse_opacity_activation = inverse_sigmoid
        self.opacity_activation = torch.sigmoid

    @property
    def num_tetrahedra(self):
        return int(self.tetra_indices.shape[0])

    @property
    def tetra_verts(self):
        return self.tetra_vertices[self.tetra_indices]

    @property
    def get_xyz(self):
        means, _, _, _ = tetra_to_gaussian_moment_matching(self.tetra_verts, self.tetra_eta, self.tetra_eps)
        return means

    @property
    def get_scaling(self):
        _, scales, _, _ = tetra_to_gaussian_moment_matching(self.tetra_verts, self.tetra_eta, self.tetra_eps)
        return scales

    @property
    def get_rotation(self):
        _, _, rotations, _ = tetra_to_gaussian_moment_matching(self.tetra_verts, self.tetra_eta, self.tetra_eps)
        return rotations

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_h_scaled(self):
        scale_sq = torch.square(self.get_scaling)
        h_cov_scaling = torch.sqrt(scale_sq.prod(dim=1) / (scale_sq + 1e-7).prod(dim=1))
        return h_cov_scaling[..., None]

    @property
    def get_scaled_opacity(self):
        return self.get_opacity * self.get_h_scaled

    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        return self.pretrained_exposures[image_name]

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, cam_infos: int, spatial_lr_scale: float, args=None):
        self.spatial_lr_scale = spatial_lr_scale
        self.tetra_eta = float(getattr(args, "tetra_eta", 1.0))
        self.tetra_eps = float(getattr(args, "tetra_eps", 1e-4))
        voxel_size = float(getattr(args, "tetra_downsample_voxel", 0.0))
        tetra_init = getattr(args, "tetra_init", "delaunay")
        if tetra_init != "delaunay":
            raise ValueError(f"Unsupported tetra_init '{tetra_init}'. The current implementation supports 'delaunay'.")

        points_np = np.asarray(pcd.points, dtype=np.float32)
        colors_np = np.asarray(pcd.colors, dtype=np.float32)
        vertices_np, colors_ds, tets_np = _build_delaunay_tetrahedra(points_np, colors_np, spatial_lr_scale, voxel_size)
        tetra_colors = colors_ds[tets_np].mean(axis=1)

        vertices = torch.tensor(vertices_np, dtype=torch.float, device="cuda")
        tets = torch.tensor(tets_np, dtype=torch.long, device="cuda")
        fused_color = RGB2SH(torch.tensor(tetra_colors, dtype=torch.float, device="cuda"))
        features = torch.zeros((tets.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float, device="cuda")
        features[:, :3, 0] = fused_color

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((tets.shape[0], 1), dtype=torch.float, device="cuda"))

        self.tetra_vertices = nn.Parameter(vertices.requires_grad_(True))
        self.tetra_indices = tets
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.num_tetrahedra), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.num_tetrahedra, 1), device="cuda")
        self.denom = torch.zeros((self.num_tetrahedra, 1), device="cuda")
        self._cache_initial_edges()

        print("Number of tetra vertices at initialisation : ", self.tetra_vertices.shape[0])
        print("Number of tetrahedra at initialisation : ", self.num_tetrahedra)

        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def _cache_initial_edges(self):
        edge_pairs = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=torch.long, device=self.tetra_indices.device)
        edges = self.tetra_indices[:, edge_pairs].reshape(-1, 2)
        self.edge_index = edges
        self.edge_to_tetra = torch.arange(self.num_tetrahedra, device=self.tetra_indices.device).repeat_interleave(6)
        edge_vec = self.tetra_vertices.detach()[edges[:, 0]] - self.tetra_vertices.detach()[edges[:, 1]]
        self.initial_edge_lengths = torch.linalg.norm(edge_vec, dim=1).detach().clamp_min(1e-8)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        if self.xyz_gradient_accum.numel() == 0:
            self.xyz_gradient_accum = torch.zeros((self.num_tetrahedra, 1), device="cuda")
            self.denom = torch.zeros((self.num_tetrahedra, 1), device="cuda")

        params = [
            {'params': [self.tetra_vertices], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "tetra_vertices"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        ]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        if self.pretrained_exposures is None:
            self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)
        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init, training_args.exposure_lr_final,
            lr_delay_steps=training_args.exposure_lr_delay_steps,
            lr_delay_mult=training_args.exposure_lr_delay_mult,
            max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)
        lr = None
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "tetra_vertices":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
        return lr

    def capture(self):
        return (
            "tetra",
            self.active_sh_degree,
            self.tetra_vertices,
            self.tetra_indices,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.tetra_eta,
            self.tetra_eps,
            self._exposure,
            self.exposure_mapping,
        )

    def restore(self, model_args, training_args):
        if model_args[0] == "tetra":
            (_, self.active_sh_degree, self.tetra_vertices, self.tetra_indices,
             self._features_dc, self._features_rest, self._opacity, self.max_radii2D,
             self.xyz_gradient_accum, self.denom, opt_dict, self.spatial_lr_scale,
             self.tetra_eta, self.tetra_eps, self._exposure, self.exposure_mapping) = model_args
        else:
            raise ValueError("TetraGaussianModel can only restore tetra checkpoints.")
        self.pretrained_exposures = None
        self._cache_initial_edges()
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    def _prune_optimizer(self, mask):
        tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "tetra_vertices":
                tensors[group["name"]] = group["params"][0]
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
            tensors[group["name"]] = group["params"][0]
        return tensors

    def prune_tetrahedra(self, prune_mask):
        valid = ~prune_mask
        if valid.sum() == 0:
            return
        tensors = self._prune_optimizer(valid)
        self._features_dc = tensors["f_dc"]
        self._features_rest = tensors["f_rest"]
        self._opacity = tensors["opacity"]
        self.tetra_indices = self.tetra_indices[valid]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid]
        self.denom = self.denom[valid]
        self.max_radii2D = self.max_radii2D[valid]
        self._cache_initial_edges()

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        signed_vol = tetra_signed_volumes(self.tetra_verts)
        prune_mask = (self.get_scaled_opacity < min_opacity).squeeze()
        prune_mask = torch.logical_or(prune_mask, signed_vol.abs() < 1e-12)
        if max_screen_size:
            prune_mask = torch.logical_or(prune_mask, self.max_radii2D > max_screen_size)
        self.prune_tetrahedra(prune_mask)
        torch.cuda.empty_cache()

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) / self.get_h_scaled * 0.01))
        for group in self.optimizer.param_groups:
            if group["name"] == "opacity":
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(opacities_new)
                    stored_state["exp_avg_sq"] = torch.zeros_like(opacities_new)
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(opacities_new.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(opacities_new.requires_grad_(True))
                self._opacity = group["params"][0]

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        update_filter = update_filter.squeeze()
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :3], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_densification_stats_direct(self, grad, update_filter):
        update_filter = update_filter.squeeze()
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def regularization_loss(self, opt):
        loss = torch.zeros((), device="cuda")
        stats = {}
        tv = self.tetra_verts
        signed_vol = tetra_signed_volumes(tv)
        vols = signed_vol.abs()

        lap_w = float(getattr(opt, "tetra_lap_weight", 0.0))
        arap_w = float(getattr(opt, "tetra_arap_weight", 0.0))
        vol_w = float(getattr(opt, "tetra_vol_weight", 0.0))
        min_w = float(getattr(opt, "tetra_min_weight", 0.0))

        if lap_w > 0 and self.edge_index is not None and self.edge_index.numel() > 0:
            src, dst = self.edge_index[:, 0], self.edge_index[:, 1]
            diff = self.tetra_vertices[src] - self.tetra_vertices[dst]
            lap_loss = (diff.square().sum(dim=1)).mean()
            loss = loss + lap_w * lap_loss
            stats["tetra/lap_loss"] = lap_loss.detach()

        if arap_w > 0 and self.edge_index is not None and self.initial_edge_lengths is not None:
            src, dst = self.edge_index[:, 0], self.edge_index[:, 1]
            cur = torch.linalg.norm(self.tetra_vertices[src] - self.tetra_vertices[dst], dim=1).clamp_min(1e-8)
            arap_loss = ((cur - self.initial_edge_lengths) / self.initial_edge_lengths).square().mean()
            loss = loss + arap_w * arap_loss
            stats["tetra/arap_loss"] = arap_loss.detach()

        if vol_w > 0:
            vol_loss = torch.relu(1e-8 - signed_vol).mean()
            loss = loss + vol_w * vol_loss
            stats["tetra/vol_loss"] = vol_loss.detach()

        if min_w > 0:
            min_loss = (self.get_opacity.squeeze() * vols.detach()).mean()
            loss = loss + min_w * min_loss
            stats["tetra/min_loss"] = min_loss.detach()

        stats["tetra/count"] = torch.tensor(float(self.num_tetrahedra), device="cuda")
        stats["tetra/inversion_count"] = (signed_vol <= 0).sum().float().detach()
        stats["tetra/min_volume"] = vols.min().detach() if vols.numel() else torch.zeros((), device="cuda")
        stats["tetra/mean_volume"] = vols.mean().detach() if vols.numel() else torch.zeros((), device="cuda")
        stats["tetra/reg_loss"] = loss.detach()
        return loss, stats

    def construct_list_of_attributes(self):
        attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attrs.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attrs.append('f_rest_{}'.format(i))
        attrs.append('opacity')
        for i in range(3):
            attrs.append('scale_{}'.format(i))
        for i in range(4):
            attrs.append('rot_{}'.format(i))
        return attrs

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scales = torch.log(self.get_scaling.detach().clamp_min(1e-8)).cpu().numpy()
        rotations = self.get_rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, 'vertex')]).write(path)

    def save_tetra(self, path):
        mkdir_p(os.path.dirname(path))
        np.savez_compressed(
            path,
            tetra_vertices=self.tetra_vertices.detach().cpu().numpy(),
            tetra_indices=self.tetra_indices.detach().cpu().numpy(),
            features_dc=self._features_dc.detach().cpu().numpy(),
            features_rest=self._features_rest.detach().cpu().numpy(),
            opacity=self._opacity.detach().cpu().numpy(),
            active_sh_degree=np.array(self.active_sh_degree, dtype=np.int32),
            max_sh_degree=np.array(self.max_sh_degree, dtype=np.int32),
            tetra_eta=np.array(self.tetra_eta, dtype=np.float32),
            tetra_eps=np.array(self.tetra_eps, dtype=np.float32),
        )

    def load_tetra(self, path, use_train_test_exp=False):
        data = np.load(path)
        self.tetra_vertices = nn.Parameter(torch.tensor(data["tetra_vertices"], dtype=torch.float, device="cuda").requires_grad_(True))
        self.tetra_indices = torch.tensor(data["tetra_indices"], dtype=torch.long, device="cuda")
        self._features_dc = nn.Parameter(torch.tensor(data["features_dc"], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(data["features_rest"], dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(data["opacity"], dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = int(data["active_sh_degree"])
        self.tetra_eta = float(data["tetra_eta"])
        self.tetra_eps = float(data["tetra_eps"])
        self.max_radii2D = torch.zeros((self.num_tetrahedra), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.num_tetrahedra, 1), device="cuda")
        self.denom = torch.zeros((self.num_tetrahedra, 1), device="cuda")
        self._cache_initial_edges()

        exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
        if use_train_test_exp and os.path.exists(exposure_file):
            with open(exposure_file, "r") as f:
                exposures = json.load(f)
            self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
        else:
            self.pretrained_exposures = None

    def load_ply(self, path, use_train_test_exp=False):
        raise FileNotFoundError(
            "TetraGaussianModel requires native tetra_state.npz checkpoints. "
            f"Could not load tetra state next to compatibility PLY: {path}"
        )
