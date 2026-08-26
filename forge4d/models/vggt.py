# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from forge4d.models.aggregator import Aggregator
from forge4d.heads.camera_head import CameraHead
from forge4d.heads.dpt_head import DPTHead, GaussianHead
from forge4d.heads.track_head import TrackHead
from forge4d.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
from forge4d.utils.rotation import quat_to_mat, mat_to_quat
from lib.loss import l1_loss, l2_loss, ssim, psnr, PerceptualLoss

from lib.GaussianRender import render_gaussians

from einops import pack, rearrange, repeat
from torchvision.utils import save_image


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, point_map=False, scaling_factor=1.0):
        super().__init__()

        self.point_map = point_map
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.lpips = PerceptualLoss(device='cuda')

        self.gs_head = GaussianHead(dim_in=2 * embed_dim, output_dim=14)
        self.intrs = torch.tensor([[
            1280, 0.0, 500],
            [0.0, 1280, 500],
            [0.0, 0.0, 1.0]])
        self.intrs[:2] = self.intrs[:2] * img_size/1024

    def whole_inferece(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        if self.camera_head is not None:
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            predictions["pose_encoding"] = pose_enc_list[-1]  # pose encoding of the last iteration

        if self.depth_head is not None:
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        if self.point_head is not None:
            pts3d, pts3d_conf = self.point_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            predictions["world_points"] = pts3d.permute(0,1,4,2,3)
            predictions["world_points_conf"] = pts3d_conf
            
        if self.gs_head is not None:
            gs_attrs = self.gs_head(
                aggregated_tokens_list, images=images, pc=predictions["world_points"], patch_start_idx=patch_start_idx
            )
            predictions["xyz"] = gs_attrs[:, :, :3]
            predictions["scale"] = gs_attrs[:, :, 3:6]
            predictions["rot"] = nn.functional.normalize(gs_attrs[:, :, 6:10], dim=2)
            predictions["opacity"] = nn.functional.sigmoid(gs_attrs[:, :, 10:11])   

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images

        return predictions
    
    def forward(
        self,
        data: dict,
        images: torch.Tensor,
        masks: torch.Tensor,
        bg_color=[1.0,1.0,1.0],
        query_points: torch.Tensor = None,
        is_val=False,
        gauge=1,
    ):
        # If without batch dimension, add it
        c, h, w = images.shape[-3:]
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # with torch.no_grad():
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        # with torch.no_grad():
        if self.camera_head is not None:
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            predictions["pose_encoding"] = pose_enc_list[-1]  # pose encoding of the last iteration

        if self.depth_head is not None:
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            predictions["depth"] = depth.permute(0,1,4,2,3)
            predictions["depth_conf"] = depth_conf
            # with torch.no_grad():
        if self.point_head is not None:
            pts3d, pts3d_conf = self.point_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            predictions["world_points"] = pts3d.permute(0,1,4,2,3)
            predictions["world_points_conf"] = pts3d_conf
            
        if self.gs_head is not None:
            gs_attrs = self.gs_head(
                aggregated_tokens_list, images=images, pc=predictions["world_points"], patch_start_idx=patch_start_idx
            )
            predictions["offset"] = nn.functional.sigmoid(gs_attrs[:, :, :3])-0.5
            # predictions["scale_maps"] = 0.001*nn.functional.softplus(gs_attrs[:, :, 3:6])
            predictions["scale_maps"] = 0.004*nn.functional.sigmoid(gs_attrs[:, :, 3:6])
            # predictions["scale_maps"] = 0.005*nn.functional.sigmoid(gs_attrs[:, :, 3:6]) # larger scales for resolution 256, cause there are less pixels and thus less 3D Gaussians.
            predictions["rot_maps"] = nn.functional.normalize(gs_attrs[:, :, 6:10], dim=2)
            predictions["opacity_maps"] = nn.functional.sigmoid(gs_attrs[:, :, 10:11])
            predictions["color_offset"] = nn.functional.sigmoid(gs_attrs[:, :, 11:14])-0.5
                # predictions["opacity"] = torch.ones_like(predictions["opacity"]).cuda()
        
        extrs, intrs = pose_encoding_to_extri_intri(predictions["pose_encoding"], images.shape[-2:])
        predictions["extr"] = extrs
        predictions["intr"] = intrs
        predictions["rgb"] = images + 0.1*predictions["color_offset"]
        
        masks_float = masks.float()
        pose_gt = extri_intri_to_pose_encoding(data['camera']['extr'], data['camera']['intr'], image_size_hw = (h, w))
        predictions['pose_gt'] = pose_gt
        norm_gt, norm_pred = torch.norm(pose_gt[:,1:,:3], dim=-1, keepdim=True), torch.norm(predictions['pose_encoding'][:,1:,:3], dim=-1, keepdim=True) 
        with torch.no_grad():
            scale = torch.mean(norm_pred/norm_gt, dim=1, keepdim=True) if gauge>0 else 1

        # data['xyz'] = predictions['xyz']
        predictions['xyz'] = predictions['world_points']+ predictions["offset"]*0.006 #0.05*scale

        if self.point_map:
            extrs_float = data['camera']['extr'].float()#extrs.float()
            intrs_float = data['camera']['intr'].float()#intrs.float()
        else:
            extrs_float = data['camera']['extr'].float()#extrs.float()
            intrs_float = data['camera']['intr'].float()#intrs.float()
            extrs_float[:,:,:3,3] = extrs_float[:,:,:3,3]*scale

        # extrs_float = extrs.float()
        # if is_val:
        #     intrs_float = intrs.float()
        # intrs_float = self.intrs.unsqueeze(dim=0).unsqueeze(dim=0).repeat(data['novel_view']['intr'].shape[0],data['novel_view']['intr'].shape[1],1,1).float().to(extrs_float.device)
        predictions_float = {}
        for key in predictions.keys():
            predictions_float[key] = predictions[key].float()
            
        synthesized_images = render_gaussians(predictions_float, predictions_float["xyz"], predictions_float["rgb"], masks_float, extrs_float, intrs_float, bg_color=bg_color)
        predictions["images"] = synthesized_images.to(images.dtype)

        extrs_float = data['novel_view']['extr'].float()
        intrs_float = data['novel_view']['intr'].float()
        extrs_float[:,:,:3,3] = extrs_float[:,:,:3,3]*scale
        # if is_val:
        # intrs_float = intrs[:, :1].repeat(1,data['novel_view']['intr'].shape[1],1,1).float()
            # intrs_float = self.intrs.unsqueeze(dim=0).unsqueeze(dim=0).repeat(data['novel_view']['intr'].shape[0],data['novel_view']['intr'].shape[1],1,1).float().to(extrs_float.device)
        
        if is_val:
            # intrs_float = intrs[:, :1].repeat(1,data['novel_view']['intr'].shape[1],1,1).float()
            extrs_float, intrs_float = self.align_test_step(data, predictions_float, masks_float, extrs_float, intrs_float, bg_color)
        synthesized_novel = render_gaussians(predictions_float, predictions_float["xyz"], predictions_float["rgb"], masks_float, extrs_float, intrs_float, bg_color=bg_color)
        predictions["novel_img"] = synthesized_novel.to(images.dtype)
        # exit()

        return predictions

    def align_test_step(self, data, predictions, mask, extrs, intrs, bg_color):
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

            # cam_intrs_delta = nn.Parameter(
            #     torch.stack([intrs[..., 0,0], intrs[..., 0,2], intrs[..., 1,1], intrs[..., 1,2]], dim=-1).clone().detach(),
            #     requires_grad=True
            # ).to(extrs.device)

            cam_scales_delta = nn.Parameter(
                torch.ones([b, 1, 1], device=extrs.device),
                requires_grad=True
            )

            opt_params = []
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": 0.005,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": 0.005,
                }
            )

            pose_optimizer = torch.optim.Adam(opt_params)

            extrinsics = extrs.clone()
            intrinsics = intrs.clone()
            for i in range(200):
                pose_optimizer.zero_grad()

                R = quat_to_mat(cam_rot_delta)
                T = cam_trans_delta*cam_scales_delta
                extrinsics = torch.cat([torch.cat([R,T[...,None]], dim=-1), torch.tensor([0,0,0,1]).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).repeat(b,v,1,1).to(extrs.device)], dim=-2)

                mask_novel = (data['novel_view']['mask']>0.6).float()
                data['novel_view']['img'] = data['novel_view']['img'] *mask_novel + (1-mask_novel)*(torch.tensor(bg_color).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-2)).to(data['imgs'])
                        
                gt_novel = data['novel_view']['img']
                render_novel = render_gaussians(predictions, predictions["xyz"], predictions["rgb"], mask, extrinsics, intrinsics, bg_color=bg_color)
                # save_image(rearrange(render_novel, "b v c h w -> (b v) c h w"), f'visualize/{i}_novel.png', normalize=True)

                # Compute and log loss.
                total_loss = 0.8*l2_loss(render_novel, gt_novel) #+ 0.2*(1.0 - ssim(rearrange(render_novel, "b v c h w -> (b v) c h w"), rearrange(gt_novel, "b v c h w -> (b v) c h w"))) + 0.02* self.lpips(rearrange(render_novel, "b v c h w -> (b v) c h w"), rearrange(gt_novel, "b v c h w -> (b v) c h w")) 

                # print(total_loss)
                # print(total_loss)
                total_loss.backward()
                # print(i,total_loss,psnr(render_novel, gt_novel))
                pose_optimizer.step()
            # save_image(rearrange(render_novel, "b v c h w -> (b v) c h w"), 'visualize/opt_novel.png', normalize=True)
            # save_image(rearrange(gt_novel, "b v c h w -> (b v) c h w"), 'visualize/opt_gt.png', normalize=True)

        return extrinsics, intrinsics
