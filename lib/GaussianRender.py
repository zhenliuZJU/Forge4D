
import torch
from gaussian_renderer import render

from einops import rearrange
import time
import open3d as o3d

from forge4d.utils.pose_enc import pose_encoding_to_extri_intri

def render_gaussians(data, xyz, imgs, masks, extr, intr, bg_color, render_vis=False):
    bs, vs = imgs.shape[:2]
    # valid = (masks>0.66).view(bs, vs, -1)
    valid = (masks>0.33).view(bs, vs, -1)
    render_novel_batch = []
    for i in range(bs):
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []
        for view_id in range(vs):
            valid_i = valid[i, view_id, :]
            xyz_i = xyz[i, view_id, :, :, :].permute(1, 2, 0).view(-1, 3).float()
            rgb_i = imgs[i, view_id, :, :, :].permute(1, 2, 0).view(-1, 3).float()
            rot_i = data['rot_maps'][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 4).float()
            scale_i = data['scale_maps'][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 3).float()
            opacity_i = data['opacity_maps'][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 1).float()

            xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
            rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
            rot_i_valid.append(rot_i[valid_i].view(-1, 4))
            scale_i_valid.append(scale_i[valid_i].view(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))

        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        pts_rgb_i = torch.concat(rgb_i_valid, dim=0)
        # pts_rgb_i = pts_rgb_i * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)

        if render_vis:
            pts_xyz_i, scale_i, rot_i, pts_rgb_i, opacity_i = downsample_and_randomize(pts_xyz_i, scale_i, rot_i, pts_rgb_i, opacity_i)
        
        render_novel_i = render(xyz, extr[i], intr[i], pts_xyz_i, pts_rgb_i, rot_i, scale_i, opacity_i, bg_color=bg_color)
        render_novel_batch.append(render_novel_i)
            

    synthesized_images = torch.stack(render_novel_batch)
    return synthesized_images

def downsample_and_randomize(points, scales, rots, shs, opacities, downsample_ratio=0.1):
    """
    下采样并随机化颜色和形状
    """
    num_pts = points.shape[0]
    num_sample = int(num_pts * downsample_ratio)

    import random
    # 1. 随机下采样索引
    indices = random.sample(range(num_pts), num_sample)
    indices = torch.tensor(indices, device="cuda")

    # 2. 根据索引下采样所有属性
    points_sampled = points[indices]
    scales_sampled = scales[indices]
    rots_sampled = rots[indices]
    opacities_sampled = opacities[indices]
    opacities_sampled = torch.ones_like(opacities_sampled)
    shs_sampled = shs[indices]

    # 3. 生成随机颜色（替换球谐系数SH）
    # 假设shs的维度为 [N, 3, (max_sh_degree+1)**2]，我们只替换RGB部分（即第0阶球谐系数）
    random_colors = torch.rand((num_sample, 3), device="cuda") # 随机RGB，范围[0, 1]
    # 将随机颜色赋值给第0阶球谐系数（对应基函数为1，表示漫反射颜色）
    shs_sampled[:, :3] = random_colors

    # 4. 修改缩放以形成椭球
    # scales原本是各向同性的（Sx, Sy, Sz）可能相似，我们通过乘以一个随机向量来制造各向异性
    # 例如，让Z轴缩放是X/Y轴的2倍，形成拉长的椭球
    ellipse_factor = torch.tensor([1.0, 1.0, 2.0], device="cuda") # 缩放因子 [X, Y, Z]
    scales_sampled = scales_sampled * ellipse_factor

    return points_sampled, scales_sampled, rots_sampled, shs_sampled, opacities_sampled

