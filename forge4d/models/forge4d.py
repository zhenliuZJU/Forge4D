# Copyright (c) 2025 Forge4D Authors
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from forge4d.models.aggregator import RecurrentAggregator_CA
from forge4d.heads.camera_head import CameraHead
from forge4d.heads.dpt_head import DPTHead, DPTGSHead, DPTMotionHead
from forge4d.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
from forge4d.utils.rotation import quat_to_mat, mat_to_quat
from forge4d.layers.fusion import GaussianFusion
from lib.loss import l1_loss, l2_loss, ssim, psnr, PerceptualLoss

from lib.GaussianRender import pts2render
from lib.utils import print_gpu_memory, quaternion_multiply
from lib.flow import image_warp


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, motion_dimension=3, point_map=False, scaling_factor=1.0, num_scene_tokens=1024):
        super().__init__()
        depth = 24
        self.img_size = img_size

        self.point_map = point_map
        self.aggregator = RecurrentAggregator_CA(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, num_scene_tokens=num_scene_tokens, depth=depth, scaling_factor=scaling_factor)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")

        self.gs_head = DPTGSHead(dim_in=2 * embed_dim, output_dim=14)
        self.motion_head = DPTMotionHead(dim_in=2 * embed_dim, output_dim=motion_dimension, intermediate_layer_idx=[0, 1, 2, 3])
        self.fusion = GaussianFusion(in_channels=23, out_channels=11)

        self.scene_token = nn.Parameter(torch.zeros(depth, num_scene_tokens, embed_dim))

    def forward(
        self,
        data: dict,
        temp_images: torch.Tensor,
        temp_masks: torch.Tensor,
        bg_color=[1.0, 1.0, 1.0],
        query_points: torch.Tensor = None,
        is_val=False
    ):
        c, h, w = temp_images.shape[-3:]
        if len(temp_images.shape) == 5:
            temp_images = temp_images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        B, T, V, C, H, W = temp_images.shape
        predictions = {}

        predictions["pose_enc"] = []
        predictions["world_points"] = []
        predictions["images"] = []
        predictions["novel_img"] = []
        predictions['pose_gt'] = []

        scene_tokens = []
        scene_tokens.append(self.scene_token.unsqueeze(dim=1).repeat(1, B, 1, 1))

        for t in range(T):
            images = temp_images[:, t]
            masks = temp_masks[:, t]

            aggregated_tokens_list, aggregated_scene_tokens_list, patch_start_idx, updated_token = self.aggregator(images, scene_tokens[t])
            scene_tokens.append(updated_token)

            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"].append(pose_enc_list[-1])

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"].append(pts3d)

            if self.gs_head is not None:
                gs_attrs = self.gs_head(
                    aggregated_tokens_list, images=images, pc=predictions["world_points"][t].permute(0, 1, 4, 2, 3), patch_start_idx=patch_start_idx
                )

                predictions["offset"] = nn.functional.sigmoid(gs_attrs[:, :, :3]) - 0.5
                predictions["scale_maps"] = 0.004 * nn.functional.sigmoid(gs_attrs[:, :, 3:6])
                predictions["rot_maps"] = nn.functional.normalize(gs_attrs[:, :, 6:10], dim=2)
                predictions["opacity_maps"] = nn.functional.sigmoid(gs_attrs[:, :, 10:11])
                predictions["color_offset"] = nn.functional.sigmoid(gs_attrs[:, :, 11:14]) - 0.5

            predictions['xyz'] = predictions['world_points'][t].permute(0, 1, 4, 2, 3) + predictions["offset"] * 0.006

            predictions["rgb"] = images + 0.1 * predictions["color_offset"]
            masks_float = masks.float()
            pose_gt = extri_intri_to_pose_encoding(data['camera']['extr'][:, t], data['camera']['intr'][:, t], image_size_hw=(h, w))
            predictions['pose_gt'].append(pose_gt)
            norm_gt, norm_pred = torch.norm(pose_gt[:, 1:, :3], dim=-1, keepdim=True), torch.norm(predictions['pose_enc'][t][:, 1:, :3], dim=-1, keepdim=True)
            with torch.no_grad():
                scale = torch.mean(norm_pred / (norm_gt + 1e-8), dim=(1, 2), keepdim=True)

            if self.point_map:
                extrs_float = data['camera']['extr'][:, t].float()
                intrs_float = data['camera']['intr'][:, t].float()
            else:
                extrs_float = data['camera']['extr'][:, t].float()
                intrs_float = data['camera']['intr'][:, t].float()
                extrs_float_new = extrs_float.clone()
                extrs_float_new[:, :, :3, 3] = extrs_float[:, :, :3, 3] * scale
                extrs_float = extrs_float_new

            synthesized_images = pts2render(predictions, predictions['xyz'], predictions["rgb"], masks_float, extrs_float, intrs_float, bg_color=bg_color)
            predictions["images"].append(synthesized_images.to(images.dtype))

            extrs_float = data['novel_view']['extr'][:, t].float()
            intrs_float = data['novel_view']['intr'][:, t].float()

            extrs_float_new = extrs_float.clone()
            extrs_float_new[:, :, :3, 3] = extrs_float[:, :, :3, 3] * scale
            extrs_float = extrs_float_new
            novel_view_gt = {}
            novel_view_gt['novel_view'] = {}
            novel_view_gt['novel_view']['img'] = data['novel_view']['img'][:, t]
            novel_view_gt['novel_view']['mask'] = data['novel_view']['mask'][:, t]

            if is_val:
                extrs_float, intrs_float = self.test_step_align(novel_view_gt, predictions, masks_float, extrs_float, intrs_float, bg_color)
            synthesized_novel = pts2render(predictions, predictions['xyz'], predictions["rgb"], masks_float, extrs_float, intrs_float, bg_color=bg_color)
            predictions["novel_img"].append(synthesized_novel.to(images.dtype))
            del predictions["offset"], predictions["scale_maps"], predictions["rot_maps"], predictions["opacity_maps"], predictions["color_offset"], predictions["xyz"], aggregated_tokens_list, patch_start_idx, updated_token

        predictions["pose_enc"] = torch.stack(predictions["pose_enc"], dim=1)
        predictions["world_points"] = torch.stack(predictions["world_points"], dim=1)
        predictions["images"] = torch.stack(predictions["images"], dim=1)
        predictions["novel_img"] = torch.stack(predictions["novel_img"], dim=1)
        predictions['pose_gt'] = torch.stack(predictions['pose_gt'], dim=1)

        return predictions

    def test_step_align(self, data, predictions, mask, extrs, intrs, bg_color, new_position=None, gaussians=None):
        if new_position is None:
            new_position = predictions['xyz']
        b, v, _, _ = extrs.shape
        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(
                mat_to_quat(extrs[..., :3, :3]).clone().detach(),
                requires_grad=True
            ).to(extrs.device)

            cam_trans_delta = nn.Parameter(
                extrs[..., :3, 3].clone().detach(),
                requires_grad=True
            ).to(extrs.device)

            cam_scales_delta = nn.Parameter(
                torch.ones([b, 1, 1], device=extrs.device),
                requires_grad=True
            )

            opt_params = [
                {"params": [cam_rot_delta], "lr": 0.005},
                {"params": [cam_trans_delta], "lr": 0.005},
            ]

            pose_optimizer = torch.optim.Adam(opt_params)

            intrinsics = intrs.clone()

            mask_novel = (data['novel_view']['mask'] > 0.6).float()
            data['novel_view']['img'] = data['novel_view']['img'] * mask_novel + (1 - mask_novel) * (torch.tensor(bg_color).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-2)).to(extrs)
            gt_novel = data['novel_view']['img']

            for _ in range(200):
                pose_optimizer.zero_grad()

                R = quat_to_mat(cam_rot_delta)
                T = cam_trans_delta * cam_scales_delta
                last_row = torch.tensor([0, 0, 0, 1], device=extrs.device).reshape(1, 1, 1, 4).repeat(b, v, 1, 1)
                extrinsics = torch.cat([torch.cat([R, T[..., None]], dim=-1), last_row], dim=-2)

                if gaussians is not None:
                    render_novel = pts2render(gaussians, gaussians["xyz"], gaussians["rgb"], gaussians["masks"], extrinsics, intrinsics, bg_color=bg_color)
                elif new_position is not None:
                    render_novel = pts2render(predictions, new_position, predictions["rgb"], mask, extrinsics, intrinsics, bg_color=bg_color)
                else:
                    render_novel = pts2render(predictions, predictions["xyz"], predictions["rgb"], predictions["masks"], extrinsics, intrinsics, bg_color=bg_color)

                total_loss = 0.8 * l2_loss(render_novel, gt_novel)
                total_loss.backward()
                pose_optimizer.step()

        return extrinsics, intrinsics
