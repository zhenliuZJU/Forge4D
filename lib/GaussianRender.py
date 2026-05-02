import torch
import torch.nn as nn
from gaussian_renderer import render


def pts2render(predictions, xyz, rgb, masks, extrs, intrs, bg_color):
    """Render 3D gaussians from per-pixel predictions.

    Converts dense per-pixel gaussian attributes into a point cloud and renders
    via gsplat rasterization.

    Args:
        predictions: dict containing 'scale_maps', 'rot_maps', 'opacity_maps'.
        xyz: Gaussian positions, shape [B, V, 3, H, W].
        rgb: Gaussian colors, shape [B, V, 3, H, W].
        masks: Validity masks, shape [B, V, 1, H, W].
        extrs: Camera extrinsics, shape [B, V, 4, 4].
        intrs: Camera intrinsics, shape [B, V, 3, 3].
        bg_color: Background color (list/tuple of 3 values).

    Returns:
        Rendered images, shape [B, V, 3, H, W].
    """
    if len(extrs.shape) == 3:
        extrs = extrs.unsqueeze(0)
        intrs = intrs.unsqueeze(0)

    B, V, _, H, W = xyz.shape
    all_imgs = []

    for b in range(B):
        view_imgs = []
        for v in range(V):
            mask = masks[b, v, 0] > 0.5

            pts_xyz = xyz[b, v].permute(1, 2, 0)[mask]
            pts_rgb = rgb[b, v].permute(1, 2, 0)[mask]

            scales = predictions["scale_maps"][b, v].permute(1, 2, 0)[mask]
            rotations = predictions["rot_maps"][b, v].permute(1, 2, 0)[mask]
            opacity = predictions["opacity_maps"][b, v, 0][mask]

            if pts_xyz.shape[0] == 0:
                view_imgs.append(torch.zeros(3, H, W, device=xyz.device))
                continue

            img = render(
                data=xyz[:, v:v+1],
                extr=extrs[b:b+1, v],
                intr=intrs[b:b+1, v],
                pts_xyz=pts_xyz,
                pts_rgb=pts_rgb,
                rotations=rotations,
                scales=scales,
                opacity=opacity,
                bg_color=bg_color,
            )
            view_imgs.append(img[0])

        all_imgs.append(torch.stack(view_imgs, dim=0))

    return torch.stack(all_imgs, dim=0)
