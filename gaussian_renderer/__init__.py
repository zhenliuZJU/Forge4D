#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from gsplat.rendering import rasterization


def render(data, extr, intr, pts_xyz, pts_rgb, rotations, scales, opacity, bg_color):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if len(extr.shape) == 2:
        extr = extr.unsqueeze(0)
        intr = intr.unsqueeze(0)
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    b, _, _, h, w = data.shape
    v, _, _ = extr.shape
    bg_color = bg_color.unsqueeze(0).repeat(v, 1)
    
    rendered_image, rendered_alpha_all, info = rasterization(
                means=pts_xyz,
                quats=rotations,
                scales=scales,
                opacities=opacity.squeeze(-1),
                colors=pts_rgb,
                viewmats=extr,
                Ks=intr,
                width=w,
                height=h,
                near_plane=0.1,
                far_plane=100,
                packed=False,
                backgrounds=bg_color,
                render_mode="RGB",
            )
    
    return rendered_image.permute(0, 3, 1, 2)

# def render(data, extr, intr, pts_xyz, pts_rgb, rotations, scales, opacity, bg_color):
#     """
#     Render the scene. 
    
#     Background tensor (bg_color) must be on GPU!
#     """
#     if len(extr.shape) == 2:
#         extr = extr.unsqueeze(0)
#         intr = intr.unsqueeze(0)
#     bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
#     b, _, _, h, w = data.shape
#     v, _, _ = extr.shape
#     bg_color = bg_color.unsqueeze(0).repeat(v, 1)
    
#     rendered_image, rendered_alpha_all, info = rasterization(
#                 means=pts_xyz,
#                 quats=rotations,
#                 scales=scales,
#                 opacities=opacity.squeeze(-1),
#                 colors=pts_rgb,
#                 viewmats=extr,
#                 Ks=intr,
#                 width=1540,
#                 height=1540,
#                 near_plane=0.1,
#                 far_plane=100,
#                 packed=False,
#                 backgrounds=bg_color,
#                 render_mode="RGB",
#             )
    
#     return rendered_image.permute(0, 3, 1, 2)
