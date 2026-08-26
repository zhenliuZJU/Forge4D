# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from forge4d.models.aggregator import StateTokenAggregator
from forge4d.heads.camera_head import CameraHead
from forge4d.heads.dpt_head import DPTHead, GaussianHead, MotionHead
from forge4d.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
from forge4d.utils.rotation import quat_to_mat, mat_to_quat
from forge4d.layers.fusion import GaussianFusion
from lib.loss import l2_loss, ssim, psnr

from lib.GaussianRender import render_gaussians
from lib.utils import print_gpu_memory, quaternion_multiply


class Forge4D(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, motion_dimension=3, point_map=False, scaling_factor=1.0, num_state_tokens=1024):
        super().__init__()
        depth = 24
        self.img_size = img_size

        self.point_map = point_map
        self.aggregator = StateTokenAggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, num_state_tokens=num_state_tokens, depth=depth, scaling_factor=scaling_factor)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")

        self.gaussian_head = GaussianHead(dim_in=2 * embed_dim, output_dim=14)
        self.motion_head = MotionHead(dim_in=2 * embed_dim, output_dim=motion_dimension, intermediate_layer_idx=[0,1,2,3])
        self.fusion = GaussianFusion(in_channels=23, out_channels=11)
        
        self.state_token = nn.Parameter(torch.zeros(depth, num_state_tokens, embed_dim))

    def forward(
        self,
        data: dict,
        frames: torch.Tensor,
        masks: torch.Tensor,
        bg_color=[1.0,1.0,1.0],
        query_points: torch.Tensor = None,
        is_val=False
    ):
        c, h, w = frames.shape[-3:]
        # If without batch dimension, add it
        if len(frames.shape) == 5:
            frames = frames.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        B, T, V, C, H, W = frames.shape
        predictions = {}

        predictions["pose_encoding"] = []
        predictions["world_points"] = []
        predictions["images"] = []
        predictions["novel_img"] = []
        predictions['pose_gt'] = []

        state_tokens = []
        state_tokens.append(self.state_token.unsqueeze(dim=1).repeat(1, B, 1, 1))

        for t in range(T):
            images = frames[:, t]
            masks = masks[:, t]

            aggregated_tokens_list, patch_start_idx, updated_state_token = self.aggregator(images, state_tokens[t])
            state_tokens.append(updated_state_token)

            if self.camera_head is not None:
                pose_encoding_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_encoding"].append(pose_encoding_list[-1])

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"].append(pts3d)
                
            if self.gaussian_head is not None:
                gaussian_attrs = self.gaussian_head(
                    aggregated_tokens_list, images=images, pc=predictions["world_points"][t].permute(0,1,4,2,3), patch_start_idx=patch_start_idx
                )

                predictions["offset"] = nn.functional.sigmoid(gaussian_attrs[:, :, :3])-0.5
                predictions["scale_maps"] = 0.004*nn.functional.sigmoid(gaussian_attrs[:, :, 3:6])
                predictions["rot_maps"] = nn.functional.normalize(gaussian_attrs[:, :, 6:10], dim=2)
                predictions["opacity_maps"] = nn.functional.sigmoid(gaussian_attrs[:, :, 10:11])
                predictions["color_offset"] = nn.functional.sigmoid(gaussian_attrs[:, :, 11:14])-0.5
                    
            predictions['xyz'] = predictions['world_points'][t].permute(0,1,4,2,3) + predictions["offset"]*0.006
            
            predictions["rgb"] = images + 0.1*predictions["color_offset"]
            masks_float = masks.float()
            pose_gt = extri_intri_to_pose_encoding(data['camera']['extr'][:,t], data['camera']['intr'][:,t], image_size_hw = (h, w))
            predictions['pose_gt'].append(pose_gt)
            norm_gt, norm_pred = torch.norm(pose_gt[:,1:,:3], dim=-1, keepdim=True), torch.norm(predictions['pose_encoding'][t][:,1:,:3], dim=-1, keepdim=True) 
            with torch.no_grad():
                scale = torch.mean(norm_pred/(norm_gt+1e-8), dim=(1,2), keepdim=True)

            if self.point_map:
                extrs_float = data['camera']['extr'][:,t].float()#extrs.float()
                intrs_float = data['camera']['intr'][:,t].float()#intrs.float()
            else:
                extrs_float = data['camera']['extr'][:,t].float()#extrs.float()
                intrs_float = data['camera']['intr'][:,t].float()#intrs.float()
                extrs_float_new = extrs_float.clone()
                extrs_float_new[:,:,:3,3] = extrs_float[:,:,:3,3]*scale
                extrs_float = extrs_float_new
                
            synthesized_images = render_gaussians(predictions, predictions['xyz'], predictions["rgb"], masks_float, extrs_float, intrs_float, bg_color=bg_color)
            predictions["images"].append(synthesized_images.to(images.dtype))

            extrs_float = data['novel_view']['extr'][:,t].float()
            intrs_float = data['novel_view']['intr'][:,t].float()

            extrs_float_new = extrs_float.clone()
            extrs_float_new[:,:,:3,3] = extrs_float[:,:,:3,3]*scale
            extrs_float = extrs_float_new
            novel_view_gt = {}
            novel_view_gt['novel_view'] = {}
            novel_view_gt['novel_view']['img'] = data['novel_view']['img'][:,t]
            novel_view_gt['novel_view']['mask'] = data['novel_view']['mask'][:,t]

            if is_val:
                # intrs_float = intrs[:, :1].repeat(1,data['novel_view']['intr'].shape[1],1,1).float()
                extrs_float, intrs_float = self.align_test_step(novel_view_gt, predictions, masks_float, extrs_float, intrs_float, bg_color)
            synthesized_novel = render_gaussians(predictions, predictions['xyz'], predictions["rgb"], masks_float, extrs_float, intrs_float, bg_color=bg_color)
            predictions["novel_img"].append(synthesized_novel.to(images.dtype))
            del predictions["offset"], predictions["scale_maps"], predictions["rot_maps"], predictions["opacity_maps"], predictions["color_offset"], predictions["xyz"], aggregated_tokens_list, patch_start_idx, updated_state_token

        predictions["pose_encoding"] = torch.stack(predictions["pose_encoding"], dim=1)
        predictions["world_points"] = torch.stack(predictions["world_points"], dim=1)
        predictions["images"] = torch.stack(predictions["images"], dim=1)
        predictions["novel_img"] = torch.stack(predictions["novel_img"], dim=1)
        predictions['pose_gt'] = torch.stack(predictions['pose_gt'], dim=1)

        return predictions
    
    def forward_matching_fusion(
        self,
        data: dict,
        frames: torch.Tensor,
        masks: torch.Tensor,
        bg_color=[1.0,1.0,1.0],
        query_points: torch.Tensor = None,
        is_val=False
    ):
        c, h, w = frames.shape[-3:]
        # If without batch dimension, add it
        if len(frames.shape) == 5:
            frames = frames.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        B, T, V, C, H, W = frames.shape
        predictions = {}

        predictions["pose_encoding"] = []
        predictions["world_points"] = []
        predictions["images"] = []
        predictions["novel_img"] = []
        predictions['pose_gt'] = []
        predictions["retarget_images_fw"] = []
        predictions["retarget_images_bw"] = []
        predictions["novel_time_fw"] = []
        predictions["novel_time_bw"] = []
        predictions["novel_time"] = []
        predictions["novel_time_novel_view"] = []
        predictions['gaussians'] = []

        state_tokens = []
        state_tokens.append(self.state_token.unsqueeze(dim=1).repeat(1, B, 1, 1))

        aggregated_tokens_alltime_list = []

        for t in range(T):
            timestep = t
            images = frames[:, timestep]
            masks = masks[:, timestep]

            aggregated_tokens_list, patch_start_idx, updated_state_token = self.aggregator(images, state_tokens[t])
            aggregated_tokens_alltime_list.append(aggregated_tokens_list)
            state_tokens.append(updated_state_token)
            # state_tokens.append(torch.zeros_like(updated_state_token).to(updated_state_token.device))

            if self.camera_head is not None:
                pose_encoding_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_encoding"].append(pose_encoding_list[-1])

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"].append(pts3d)
                
            if self.gaussian_head is not None:
                gaussian_attrs = self.gaussian_head(
                    aggregated_tokens_list, images=images, pc=predictions["world_points"][t].permute(0,1,4,2,3), patch_start_idx=patch_start_idx
                )
                gaussian_keys = ["rot_maps", "scale_maps", "opacity_maps", "color_offset"]

                predictions["offset"] = nn.functional.sigmoid(gaussian_attrs[:, :, :3])-0.5
                predictions["scale_maps"] = 0.004*nn.functional.sigmoid(gaussian_attrs[:, :, 3:6])
                predictions["rot_maps"] = nn.functional.normalize(gaussian_attrs[:, :, 6:10], dim=2)
                predictions["opacity_maps"] = nn.functional.sigmoid(gaussian_attrs[:, :, 10:11])
                predictions["color_offset"] = nn.functional.sigmoid(gaussian_attrs[:, :, 11:14])-0.5
                    
            predictions['xyz'] = predictions['world_points'][t].permute(0,1,4,2,3) + predictions["offset"]*0.006
            predictions["rgb"] = images + 0.1*predictions["color_offset"]
            masks_float = masks.float()
            
            pose_gt = extri_intri_to_pose_encoding(data['camera']['extr'][:,timestep], data['camera']['intr'][:,timestep], image_size_hw = (h, w))
            predictions['pose_gt'].append(pose_gt)
            norm_gt, norm_pred = torch.norm(pose_gt[:,1:,:3], dim=-1, keepdim=True), torch.norm(predictions['pose_encoding'][t][:,1:,:3], dim=-1, keepdim=True) 
            with torch.no_grad():
                scale = torch.mean(norm_pred/(norm_gt+1e-8), dim=(1,2), keepdim=True)

            if self.point_map:
                extrs_float = data['camera']['extr'][:,timestep].float()#extrs.float()
                intrs_float = data['camera']['intr'][:,timestep].float()#intrs.float()
            else:
                extrs_float = data['camera']['extr'][:,timestep].float()#extrs.float()
                intrs_float = data['camera']['intr'][:,timestep].float()#intrs.float()
                extrs_float_new = extrs_float.clone()
                extrs_float_new[:,:,:3,3] = extrs_float[:,:,:3,3]*scale
                extrs_float = extrs_float_new

            predictions_gaussians = {"xyz": predictions['xyz'], "rgb": predictions["rgb"], "masks": masks_float, 
                                    "extrs": extrs_float, "intrs": intrs_float, 
                                    "rot_maps": predictions["rot_maps"], "opacity_maps": predictions["opacity_maps"], 
                                    "scale_maps": predictions["scale_maps"],
                                    "rgb": predictions["rgb"]}
            predictions["gaussians"].append(predictions_gaussians)
                
            synthesized_images = render_gaussians(predictions, predictions["xyz"], predictions["rgb"], masks_float, extrs_float, intrs_float, bg_color=bg_color)

            extrs_float_novel = data['novel_view']['extr'][:,timestep].float()
            intrs_float_novel = data['novel_view']['intr'][:,timestep].float()

            extrs, intrs = pose_encoding_to_extri_intri(predictions["pose_encoding"][t], images.shape[-2:])
            # print(intrs_float_novel, intrs)
            # exit()

            extrs_float_new = extrs_float_novel.clone()
            extrs_float_new[:,:,:3,3] = extrs_float_novel[:,:,:3,3]*scale
            extrs_float_novel = extrs_float_new

            novel_view_gt = {}
            novel_view_gt['novel_view'] = {}
            novel_view_gt['novel_view']['img'] = data['novel_view']['img'][:,t]
            novel_view_gt['novel_view']['mask'] = data['novel_view']['mask'][:,t]
            if is_val:
                # intrs_float_novel = intrs[:, :1].repeat(1,data['novel_view']['intr'][:,timestep].shape[1],1,1).float()
                extrs_float_novel, intrs_float_novel = self.align_test_step(novel_view_gt, predictions, masks_float, extrs_float_novel, intrs_float_novel, bg_color)
            synthesized_novel = render_gaussians(predictions, predictions["xyz"], predictions["rgb"], masks_float, extrs_float_novel, intrs_float_novel, bg_color=bg_color)                        

            ### Predict Novel Time Images ###
            if self.motion_head is not None and t>0:
                intermediate_layer_idx = [4,11,17,23]
                aggregated_motion_tokens_list_bw, aggregated_motion_tokens_list_fw = [], []
                for idx, (previous_token, current_token) in enumerate(zip([aggregated_tokens_alltime_list[t-1][i] for i in intermediate_layer_idx], [aggregated_tokens_list[i] for i in intermediate_layer_idx])):
                    B, V, C, P = current_token.shape
                    aggregate_token_attn_bw = self.aggregator.cross_blocks[idx](current_token.reshape(B*V,C,P).detach(), previous_token.reshape(B*V,C,P).detach(), pos1=self.aggregator.pos, pos2=self.aggregator.pos).reshape(B,V,C,P)
                    aggregate_token_attn_fw = self.aggregator.cross_blocks[idx](previous_token.reshape(B*V,C,P).detach(), current_token.reshape(B*V,C,P).detach(), pos1=self.aggregator.pos, pos2=self.aggregator.pos).reshape(B,V,C,P)
                    aggregated_motion_tokens_list_bw.append(aggregate_token_attn_bw)
                    aggregated_motion_tokens_list_fw.append(aggregate_token_attn_fw)

                # Backward
                concat_points = torch.cat([predictions["world_points"][t-1].permute(0,1,4,2,3).detach(), predictions["world_points"][t].permute(0,1,4,2,3).detach()], dim=2)
                concat_images = torch.cat([frames[:, timestep-1], frames[:, timestep]], dim=2)
                motions = self.motion_head(
                    aggregated_motion_tokens_list_bw, images=concat_images, pc=concat_points, patch_start_idx=patch_start_idx
                )

                motion_maps = motions[:, :, :3]
                points_local = torch.einsum('BVDC,BVCHW->BVDHW', predictions['gaussians'][t]["extrs"][:,:,:3,:3].detach(), predictions['xyz'].detach())+predictions['gaussians'][t]["extrs"][:,:,:3,3].unsqueeze(-1).unsqueeze(-1).detach()
                points_local_3d = torch.einsum('BVDC,BVCHW->BVDHW', predictions['gaussians'][t]["intrs"], points_local)
                points_local_2d = points_local_3d[:,:,:2]/(points_local_3d[:,:,2:]+1e-8)

                points_local_trans = torch.einsum('BVDC,BVCHW->BVDHW', predictions['gaussians'][t]["extrs"][:,:,:3,:3].detach(), predictions['xyz'].detach() + motion_maps)+predictions['gaussians'][t]["extrs"][:,:,:3,3].unsqueeze(-1).unsqueeze(-1).detach()
                points_local_3d_trans = torch.einsum('BVDC,BVCHW->BVDHW', predictions['gaussians'][t]["intrs"], points_local_trans)
                points_local_2d_trans = points_local_3d_trans[:,:,:2]/(points_local_3d_trans[:,:,2:]+1e-8)
                
                predictions["flow_bw"] = points_local_2d_trans - points_local_2d

                gs_position_novelt_bw = predictions['xyz'].detach() + motion_maps*0.5
                synthesized_novel_time = render_gaussians(predictions, gs_position_novelt_bw, predictions["rgb"], masks_float, predictions['gaussians'][t]["extrs"].detach(), predictions['gaussians'][t]["intrs"], bg_color=bg_color)

                gs_position_bw = predictions['xyz'].detach() + motion_maps
                synthesized_retargeted = render_gaussians(predictions, gs_position_bw, predictions["rgb"], masks_float, predictions['gaussians'][t]["extrs"].detach(), predictions['gaussians'][t]["intrs"], bg_color=bg_color)
            
                predictions["novel_time_bw"].append(synthesized_novel_time.to(images.dtype))
                predictions["retarget_images_bw"].append(synthesized_retargeted.to(images.dtype))

                # Forward
                concat_points = torch.cat([predictions["world_points"][t].permute(0,1,4,2,3).detach(), predictions["world_points"][t-1].permute(0,1,4,2,3).detach()], dim=2)
                concat_images = torch.cat([frames[:, timestep], frames[:, timestep-1]], dim=2)
                motions = self.motion_head(
                    aggregated_motion_tokens_list_fw, images=concat_images, pc=concat_points, patch_start_idx=patch_start_idx
                )
                motion_maps = motions[:, :, :3]
                points_local = torch.einsum('BVDC,BVCHW->BVDHW', extrs_float[:,:,:3,:3].detach(), predictions['gaussians'][t-1]['xyz'].detach())+extrs_float[:,:,:3,3].unsqueeze(-1).unsqueeze(-1).detach()
                points_local_3d = torch.einsum('BVDC,BVCHW->BVDHW', intrs_float, points_local)
                points_local_2d = points_local_3d[:,:,:2]/(points_local_3d[:,:,2:]+1e-8)

                points_local_trans = torch.einsum('BVDC,BVCHW->BVDHW', extrs_float[:,:,:3,:3].detach(), predictions['gaussians'][t-1]['xyz'].detach() + motion_maps)+extrs_float[:,:,:3,3].unsqueeze(-1).unsqueeze(-1).detach()
                points_local_3d_trans = torch.einsum('BVDC,BVCHW->BVDHW', intrs_float, points_local_trans)
                points_local_2d_trans = points_local_3d_trans[:,:,:2]/(points_local_3d_trans[:,:,2:]+1e-8)
                
                predictions["flow_fw"] = points_local_2d_trans - points_local_2d

                gs_position_novelt_fw = predictions['gaussians'][t-1]['xyz'].detach() + motion_maps*0.5
                synthesized_novel_time = render_gaussians(predictions["gaussians"][t-1], gs_position_novelt_fw, predictions['gaussians'][t-1]["rgb"], predictions['gaussians'][t-1]["masks"], extrs_float.detach(), intrs_float, bg_color=bg_color)

                gs_position_fw = predictions['gaussians'][t-1]['xyz'].detach() + motion_maps
                synthesized_retargeted = render_gaussians(predictions["gaussians"][t-1], gs_position_fw, predictions['gaussians'][t-1]["rgb"], predictions['gaussians'][t-1]["masks"], extrs_float.detach(), intrs_float, bg_color=bg_color)
            
                predictions["novel_time_fw"].append(synthesized_novel_time.to(images.dtype))
                predictions["retarget_images_fw"].append(synthesized_retargeted.to(images.dtype))

                origin_gaussian_0 = torch.cat([predictions["gaussians"][t-1]['rgb'], predictions["gaussians"][t-1]['rot_maps'], predictions["gaussians"][t-1]['opacity_maps'], predictions["gaussians"][t-1]['scale_maps']], dim=2)
                origin_gaussian_1 = torch.cat([predictions["gaussians"][t]['rgb'], predictions["gaussians"][t]['rot_maps'], predictions["gaussians"][t]['opacity_maps'], predictions["gaussians"][t]['scale_maps']], dim=2)

                # forward mask
                tenOne = (predictions["xyz"].detach()*predictions["gaussians"][t]['masks']).reshape(B*V,3,H,W).detach()
                tenTwo = (gs_position_fw*predictions["gaussians"][t-1]['masks']).reshape(B*V,3,H,W).detach()
                
                xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
                yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
                xx = xx.view(1, H, W, 1).repeat(B*V, 1, 1, 1)
                yy = yy.view(1, H, W, 1).repeat(B*V, 1, 1, 1)
                grid = torch.cat((xx, yy), 3).float()
                vgrid = grid.to(predictions["flow_fw"]) + predictions["flow_fw"].reshape(B*V, 2, H, W).permute(0, 2, 3, 1)
                # vgrid = grid.to(data["flow_fw"][:, t-1]) + data["flow_fw"][:, t-1].reshape(B*V, 2, H, W).permute(0, 2, 3, 1)
                # vgrid = points_local_2d_trans.reshape(B*V,2,H,W).permute(0, 2, 3, 1)
                vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
                vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
                tenSoftmax = F.grid_sample(tenOne, vgrid, align_corners=True)
                gaussian_sampled_fw = F.grid_sample(origin_gaussian_1.reshape(B*V,-1,H,W), vgrid, align_corners=True).reshape(B,V,-1,H,W)

                dist_fw = torch.sqrt((tenTwo - tenSoftmax).pow(2).mean(dim=1)).reshape(B, V, 1, H, W)/scale.unsqueeze(dim=-1).unsqueeze(dim=-1)
                flow_matching_mask_fw = (dist_fw > 0.05).float()
                # predictions["gaussians"][t-1]['masks'] = predictions["gaussians"][t-1]['masks'] * flow_matching_mask
                fused_fw = self.fusion(torch.cat([origin_gaussian_0, gaussian_sampled_fw, dist_fw], dim=2).reshape(B*V,-1,H,W)).reshape(B,V,-1,H,W)

                fused_gs_fw = {}
                fused_gs_fw["rgb"] = predictions["gaussians"][t-1]['rgb']+0.1*(nn.functional.sigmoid(fused_fw[:, :, :3])-0.5)
                fused_gs_fw["rot_maps"] = quaternion_multiply(predictions["gaussians"][t-1]['rot_maps'], nn.functional.normalize(fused_fw[:, :, 3:7] + 1e-8, dim=2))
                fused_gs_fw["opacity_maps"] = predictions["gaussians"][t-1]['opacity_maps']*nn.functional.sigmoid(fused_fw[:, :, 7:8])
                fused_gs_fw["scale_maps"] = predictions["gaussians"][t-1]['scale_maps']+fused_fw[:, :, 8:11]

                # backward mask
                tenOne = (predictions["gaussians"][t-1]["xyz"].detach()*predictions["gaussians"][t-1]['masks']).reshape(B*V,3,H,W).detach()
                tenTwo = (gs_position_bw*predictions["gaussians"][t]['masks']).reshape(B*V,3,H,W).detach()
                
                xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
                yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
                xx = xx.view(1, H, W, 1).repeat(B*V, 1, 1, 1)
                yy = yy.view(1, H, W, 1).repeat(B*V, 1, 1, 1)
                grid = torch.cat((xx, yy), 3).float()
                vgrid = grid.to(predictions["flow_bw"]) + predictions["flow_bw"].reshape(B*V, 2, H, W).permute(0, 2, 3, 1)
                # vgrid = grid.to(data["flow_fw"][:, t-1]) + data["flow_fw"][:, t-1].reshape(B*V, 2, H, W).permute(0, 2, 3, 1)
                # vgrid = points_local_2d_trans.reshape(B*V,2,H,W).permute(0, 2, 3, 1)
                vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
                vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
                tenSoftmax = F.grid_sample(tenOne, vgrid, align_corners=True)
                gaussian_sampled_bw = F.grid_sample(origin_gaussian_0.reshape(B*V,-1,H,W), vgrid, align_corners=True).reshape(B, V,-1,H,W)

                dist_bw = torch.sqrt((tenTwo - tenSoftmax).pow(2).mean(dim=1)).reshape(B, V, 1, H, W)/scale.unsqueeze(dim=-1).unsqueeze(dim=-1)
                flow_matching_mask_bw = (dist_bw > 0.05).float()
                # predictions["gaussians"][t]['masks'] = predictions["gaussians"][t]['masks'] * flow_matching_mask
                fused_bw = self.fusion(torch.cat([origin_gaussian_1, gaussian_sampled_bw, dist_bw], dim=2).reshape(B*V,-1,H,W)).reshape(B,V,-1,H,W)

                fused_gs_bw = {}
                fused_gs_bw["rgb"] = predictions["gaussians"][t]['rgb']+0.1*(nn.functional.sigmoid(fused_bw[:, :, :3])-0.5)
                fused_gs_bw["rot_maps"] = quaternion_multiply(predictions["gaussians"][t]['rot_maps'], nn.functional.normalize(fused_bw[:, :, 3:7] + 1e-8, dim=2))
                fused_gs_bw["opacity_maps"] = predictions["gaussians"][t]['opacity_maps']*nn.functional.sigmoid(fused_bw[:, :, 7:8])
                fused_gs_bw["scale_maps"] = predictions["gaussians"][t]['scale_maps']+fused_bw[:, :, 8:11]
                
                gaussians = {"xyz": torch.cat([gs_position_novelt_fw, gs_position_novelt_bw], dim=1), 
                            "rgb": torch.cat([fused_gs_fw['rgb'], fused_gs_bw['rgb']], dim=1), 
                            "masks": torch.cat([predictions["gaussians"][t-1]['masks'], predictions["gaussians"][t]['masks']], dim=1), 
                            "extrs": torch.cat([predictions["gaussians"][t-1]['extrs'], predictions["gaussians"][t]['extrs']], dim=1), 
                            "intrs": torch.cat([predictions["gaussians"][t-1]['intrs'], predictions["gaussians"][t]['intrs']], dim=1), 
                            "rot_maps": torch.cat([fused_gs_fw['rot_maps'], fused_gs_bw['rot_maps']], dim=1), 
                            "opacity_maps": torch.cat([fused_gs_fw['opacity_maps'], fused_gs_bw['opacity_maps']], dim=1), 
                            "scale_maps": torch.cat([fused_gs_fw['scale_maps'], fused_gs_bw['scale_maps']], dim=1)}
                
                novel_time_gt = {}
                novel_time_gt['novel_view'] = {}
                novel_time_gt['novel_view']['img'] = data['novel_time'][:,t-1]
                novel_time_gt['novel_view']['mask'] = data['novel_time_mask'][:,t-1]
                if is_val:
                    # intrs_float_novel = intrs[:, :1].repeat(b,data['novel_view']['intr'].shape[1],1,1).float()
                    extrs_float_novel, intrs_float_novel = self.align_test_step(novel_time_gt, predictions, masks_float, extrs_float, intrs_float, bg_color, gaussians=gaussians)
                synthesized_novel_time = render_gaussians(gaussians, gaussians["xyz"], gaussians["rgb"], gaussians["masks"], extrs_float_novel.detach(), intrs_float_novel, bg_color=bg_color)

                predictions["novel_time"].append(synthesized_novel_time.to(images.dtype))

                #######
                extrs_float_novel = data['novel_view']['extr'][:,timestep].float()
                intrs_float_novel = data['novel_view']['intr'][:,timestep].float()

                extrs_float_new = extrs_float_novel.clone()
                extrs_float_new[:,:,:3,3] = extrs_float_novel[:,:,:3,3]*scale
                extrs_float_novel = extrs_float_new

                if is_val:
                    novel_time_gt = {}
                    novel_time_gt['novel_view'] = {}
                    novel_time_gt['novel_view']['img'] = data['novel_view']['novel_time'][:,t-1]
                    novel_time_gt['novel_view']['mask'] = data['novel_view']['novel_time_mask'][:,t-1]
                    # intrs_float_novel = intrs[:, :1].repeat(1,data['novel_view']['intr'][:,timestep].shape[1],1,1).float()
                    extrs_float_novel, intrs_float_novel = self.align_test_step(novel_time_gt, predictions, masks_float, extrs_float_novel, intrs_float_novel, bg_color, gaussians=gaussians)
                synthesized_novel_time = render_gaussians(gaussians, gaussians["xyz"], gaussians["rgb"], gaussians["masks"], extrs_float_novel.detach(), intrs_float_novel, bg_color=bg_color)
                predictions["novel_time_novel_view"].append(synthesized_novel_time.to(images.dtype))
            
            predictions["images"].append(synthesized_images.to(images.dtype))
            predictions["novel_img"].append(synthesized_novel.to(images.dtype))

            del predictions["offset"], predictions["scale_maps"], predictions["rot_maps"], predictions["opacity_maps"], predictions["color_offset"], predictions["xyz"], aggregated_tokens_list, patch_start_idx, updated_state_token

        predictions["pose_encoding"] = torch.stack(predictions["pose_encoding"], dim=1)
        predictions["world_points"] = torch.stack(predictions["world_points"], dim=1)
        predictions["images"] = torch.stack(predictions["images"], dim=1)
        predictions["novel_img"] = torch.stack(predictions["novel_img"], dim=1)
        predictions['pose_gt'] = torch.stack(predictions['pose_gt'], dim=1)
        predictions["novel_time_fw"] = torch.stack(predictions["novel_time_fw"], dim=1)
        predictions["novel_time_bw"] = torch.stack(predictions["novel_time_bw"], dim=1)
        predictions["novel_time"] = torch.stack(predictions["novel_time"], dim=1)
        predictions["retarget_images_fw"] = torch.stack(predictions["retarget_images_fw"], dim=1)
        predictions["retarget_images_bw"] = torch.stack(predictions["retarget_images_bw"], dim=1)
        predictions["novel_time_novel_view"] = torch.stack(predictions["novel_time_novel_view"], dim=1)

        return predictions
 
    def align_test_step(self, data, predictions, mask, extrs, intrs, bg_color, new_position=None, gaussians=None):
        if new_position == None:
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
            # intrinsics[:, :, :2] = intrs[:, :, :2]*0.8
            
            mask_novel = (data['novel_view']['mask']>0.6).float()
            data['novel_view']['img'] = data['novel_view']['img'] *mask_novel + (1-mask_novel)*(torch.tensor(bg_color).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-2)).to(extrs)    
            gt_novel = data['novel_view']['img']

            for i in range(200):
                pose_optimizer.zero_grad()

                R = quat_to_mat(cam_rot_delta)
                T = cam_trans_delta*cam_scales_delta
                extrinsics = torch.cat([torch.cat([R,T[...,None]], dim=-1), torch.tensor([0,0,0,1]).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).repeat(b,v,1,1).to(extrs.device)], dim=-2)

                if gaussians is not None:
                    render_novel = render_gaussians(gaussians, gaussians["xyz"], gaussians["rgb"], gaussians["masks"], extrinsics, intrinsics, bg_color=bg_color)
                elif new_position is not None:
                    render_novel = render_gaussians(predictions, new_position, predictions["rgb"], mask, extrinsics, intrinsics, bg_color=bg_color)
                else:
                    render_novel = render_gaussians(predictions, predictions["xyz"], predictions["rgb"], predictions["masks"], extrinsics, intrinsics, bg_color=bg_color)
                # save_image(rearrange(render_novel, "b v c h w -> (b v) c h w"), f'visualize/{i}_novel.png', normalize=True)

                # Compute and log loss.
                total_loss = 0.8*l2_loss(render_novel, gt_novel) #+ 0.2*(1.0 - ssim(rearrange(render_novel, "b v c h w -> (b v) c h w"), rearrange(gt_novel, "b v c h w -> (b v) c h w"))) + 0.02* self.lpips(rearrange(render_novel, "b v c h w -> (b v) c h w"), rearrange(gt_novel, "b v c h w -> (b v) c h w")) 

                # print(total_loss)
                total_loss.backward()
                # print(i,total_loss,psnr(render_novel, gt_novel))
                pose_optimizer.step()
            # exit()
            # save_image(rearrange(render_novel, "b v c h w -> (b v) c h w"), 'visualize/opt_novel.png', normalize=True)
            # save_image(rearrange(gt_novel, "b v c h w -> (b v) c h w"), 'visualize/opt_gt.png', normalize=True)

        return extrinsics, intrinsics