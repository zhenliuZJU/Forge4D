#!/usr/bin/env python3
"""Forge4D inference demo.

Feed-forward 4D human reconstruction and novel-view / novel-time rendering
from uncalibrated sparse-view videos (Forge4D, ECCV 2026, arXiv:2509.24209).

The pipeline streams over the frames of one sparse-view (4-view) video clip:
state-token aggregation -> camera pose -> 3D point map -> per-view 3D Gaussians
-> bidirectional scene-flow estimation with occlusion-aware fusion, followed by
novel-view and novel-time rendering through a differentiable Gaussian splatter.

Usage:
    python runfiles/inference.py --weight /path/to/checkpoint.pth \
        --data_root /path/to/datasets \
        --cfg config/inference.yaml \
        --test_ids 0007_04 \
        --output_dir ./output

The dataset is expected to follow the DNA-Rendering layout under
``<data_root>/DNA-Rendering/main/<subject_id>`` (see README.md).
"""
import argparse
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.transformer_human_config import ConfigStereoHuman as config
from lib.dna_dataset import DNADataset
from lib.GaussianRender import render_gaussians
from lib.utils import quaternion_multiply
from tools.camera import generate_trajectory
from forge4d.models.forge4d import Forge4D
from forge4d.utils.pose_enc import extri_intri_to_pose_encoding


def parse_args():
    parser = argparse.ArgumentParser(
        description="Forge4D inference: 4D human reconstruction + novel view/time rendering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weight", type=str, required=True,
                        help="Path to a trained Forge4D checkpoint (state dict with a 'network' key).")
    parser.add_argument("--cfg", type=str, default="config/inference.yaml",
                        help="Inference config yaml (model / dataset hyper-parameters).")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root of the dataset tree; overrides cfg.dataset.data_root.")
    parser.add_argument("--test_ids", type=str, default="0007_04",
                        help="Comma-separated subject ids, matched as substrings against "
                             "<data_root>/DNA-Rendering/main/.")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory where rendered images and videos are written.")
    parser.add_argument("--frames", type=int, default=50,
                        help="Number of input frames to process per subject.")
    parser.add_argument("--sampling_rate", type=int, default=1,
                        help="Temporal sampling rate of the input frames.")
    parser.add_argument("--extend_frames", type=int, default=2,
                        help="Number of novel-time frames interpolated between consecutive inputs.")
    parser.add_argument("--motion_dimension", type=int, default=3,
                        help="Motion-head output channels (3: flow only, 4: flow + opacity).")
    parser.add_argument("--bg_color", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Background color used to composite the masked inputs.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- configuration ----
    cfg = config()
    cfg.load(args.cfg)
    cfg = cfg.get_cfg()
    if args.data_root is not None:
        cfg.dataset.data_root = args.data_root
    data_root = cfg.dataset.data_root

    # ---- model ----
    model = Forge4D(motion_dimension=args.motion_dimension)
    checkpoint = torch.load(args.weight, map_location="cpu")
    state_dict = OrderedDict()
    for key, value in checkpoint["network"].items():
        state_dict[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(state_dict, strict=True)
    model.cuda().eval()

    # ---- resolve subject ids under <data_root>/DNA-Rendering/main/ ----
    main_dir = os.path.join(data_root, "DNA-Rendering", "main")
    test_ids = [tid for tid in args.test_ids.split(",") if tid.strip()]
    subject_ids = sorted(hid for hid in os.listdir(main_dir)
                         if any(tid in hid for tid in test_ids))
    if not subject_ids:
        raise ValueError(f"No subjects matched {test_ids} under {main_dir}")
    print(f"Subjects: {subject_ids}")

    video_dir = os.path.join(args.output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    for test_id in subject_ids:
        print(f"\n=== Processing subject {test_id} ===")

        # ---- dataset ----
        dataset = DNADataset(cfg.dataset, phase="val", sampling_rate=args.sampling_rate,
                             max_frames=args.frames, test_ids=[test_id])
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        iterator = iter(loader)
        print(f"Dataset length: {len(dataset)}")
        if len(dataset) == 0:
            raise ValueError(f"Dataset is empty for subject {test_id}!")

        # ---- output dirs ----
        base_dir = os.path.join(args.output_dir, test_id)
        render_dir = os.path.join(base_dir, "render", str(args.extend_frames))
        gs_dir = os.path.join(base_dir, "3DGS", str(args.extend_frames))
        os.makedirs(render_dir, exist_ok=True)
        os.makedirs(gs_dir, exist_ok=True)

        bg_color = args.bg_color
        frames = args.frames
        sampling_rate = args.sampling_rate
        extend_frames = args.extend_frames

        all_synthesized_images = []
        predictions = {}
        predictions["pose_encoding"] = []
        predictions["world_points"] = []
        predictions["images"] = []
        predictions["novel_img"] = []
        predictions["pose_gt"] = []
        predictions["retarget_images"] = []
        predictions["novel_time"] = []
        predictions["novel_time&view"] = []
        predictions["gaussians"] = []
        predictions["gt_scene_flow"] = []
        predictions["scene_flow_mask"] = []
        predictions["motions"] = []

        frame_id = 0
        novel_extr = None

        with torch.no_grad():
            for t in range(frames // sampling_rate):

                try:
                    data = next(iterater)
                except:
                    iterater = iter(loader)
                    data = next(iterater)
                print("Processing time step:", t)

                attr_list = ["imgs", "masks", "novel_time", "novel_time_mask"]
                for item in attr_list:
                    if item in data.keys():
                        data[item] = data[item].cuda()

                for item in ["novel_view", "camera"]:
                    if item in data.keys():
                        for k in data[item].keys():
                            data[item][k] = data[item][k].cuda()
                mask = (data['masks']>0.66).float()#.expand_as(data['xyz'])
                data['imgs'] = data['imgs'] *mask + (1-mask)*(torch.tensor(bg_color).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-2)).to(data['imgs'])

                images = data['imgs']
                masks = data['masks']
                # If without batch dimension, add it
                if len(images.shape) == 4:
                    images = images.unsqueeze(0)
                B, V, C, H, W = images.shape

                state_token = model.state_token.unsqueeze(dim=1).repeat(1, B, 1, 1)

                timestep = t

                aggregated_tokens_list, patch_start_idx, state_token = model.aggregator(images, state_token) #

                if model.camera_head is not None:
                    pose_encoding_list = model.camera_head(aggregated_tokens_list)
                    predictions["pose_encoding"] = pose_encoding_list[-1]

                if model.point_head is not None:
                    pts3d, pts3d_conf = model.point_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    predictions["world_points"] = pts3d

                if model.gaussian_head is not None:
                    gaussian_attrs = model.gaussian_head(
                        aggregated_tokens_list, images=images, pc=predictions["world_points"].permute(0,1,4,2,3), patch_start_idx=patch_start_idx
                    )
                    gaussian_keys = ["rot_maps", "scale_maps", "opacity_maps", "color_offset"]

                    predictions["offset"] = nn.functional.sigmoid(gaussian_attrs[:, :, :3])-0.5
                    predictions["scale_maps"] = 0.004*nn.functional.sigmoid(gaussian_attrs[:, :, 3:6])
                    predictions["rot_maps"] = nn.functional.normalize(gaussian_attrs[:, :, 6:10], dim=2)
                    predictions["opacity_maps"] = nn.functional.sigmoid(gaussian_attrs[:, :, 10:11])
                    predictions["color_offset"] = nn.functional.sigmoid(gaussian_attrs[:, :, 11:14])-0.5

                predictions['xyz'] = predictions['world_points'].permute(0,1,4,2,3) + predictions["offset"]*0.006

                predictions["rgb"] = images + 0.1*predictions["color_offset"]
                masks_float = masks.float()
                pose_gt = extri_intri_to_pose_encoding(data['camera']['extr'], data['camera']['intr'], image_size_hw = (H, W))
                norm_gt, norm_pred = torch.norm(pose_gt[:,1:,:3], dim=-1, keepdim=True), torch.norm(predictions['pose_encoding'][:,1:,:3], dim=-1, keepdim=True) 
                with torch.no_grad():
                    scale = torch.mean(norm_pred/(norm_gt+1e-8), dim=(1,2), keepdim=True)

                if model.point_map:
                    extrs_float = data['camera']['extr'].float()#extrs.float()
                    intrs_float = data['camera']['intr'].float()#intrs.float()
                else:
                    extrs_float = data['camera']['extr'].float()#extrs.float()
                    intrs_float = data['camera']['intr'].float()#intrs.float()
                    extrs_float_new = extrs_float.clone()
                    extrs_float_new[:,:,:3,3] = extrs_float[:,:,:3,3]*scale
                    extrs_float = extrs_float_new

                    extr_input = extrs_float
                    intr_input = intrs_float

                extr_novel = data['novel_view']['extr'].float()
                intr_novel = data['novel_view']['intr'].float()
                extr_novel[:,:,:3,3] = extr_novel[:,:,:3,3]*scale
                if novel_extr is None:
                    novel_extr = generate_trajectory(extrs_float[:,0], extrs_float[:,1], num_frames=25*extend_frames//2)+generate_trajectory(extrs_float[:,1], extrs_float[:,2], num_frames=25*extend_frames//2)+generate_trajectory(extrs_float[:,2], extrs_float[:,3], num_frames=25*extend_frames//2)
                    novel_extr = novel_extr + novel_extr[::-1] +novel_extr +novel_extr[::-1] +novel_extr +novel_extr[::-1] +novel_extr + novel_extr[::-1] +novel_extr +novel_extr[::-1] +novel_extr +novel_extr[::-1] +novel_extr + novel_extr[::-1] +novel_extr +novel_extr[::-1] +novel_extr +novel_extr[::-1]
                    novel_extr = novel_extr + novel_extr[::-1]
                    novel_extr = novel_extr + novel_extr[::-1]
                    novel_extr = novel_extr + novel_extr[::-1]
                    novel_extr = novel_extr + novel_extr[::-1]

                predictions_gaussians = {"xyz": predictions['xyz'], "rgb": predictions["rgb"], "masks": masks_float, 
                                        "extrs": extrs_float, "intrs": intrs_float, 
                                        "rot_maps": predictions["rot_maps"], "opacity_maps": predictions["opacity_maps"], 
                                        "scale_maps": predictions["scale_maps"],
                                        "rgb": predictions["rgb"]}


                # ---- bidirectional motion estimation + occlusion-aware fusion ----
                if model.motion_head is not None and t > 0:
                    intermediate_layer_idx = [4,11,17,23]
                    aggregated_motion_tokens_list_bw, aggregated_motion_tokens_list_fw = [], []
                    for idx, (previous_token, current_token) in enumerate(zip([aggregated_tokens_last_list[i] for i in intermediate_layer_idx], [aggregated_tokens_list[i] for i in intermediate_layer_idx])):
                        B, V, C, P = current_token.shape
                        aggregate_token_attn_bw = model.aggregator.cross_blocks[idx](current_token.reshape(B*V,C,P).detach(), previous_token.reshape(B*V,C,P).detach(), pos1=model.aggregator.pos, pos2=model.aggregator.pos).reshape(B,V,C,P)
                        aggregate_token_attn_fw = model.aggregator.cross_blocks[idx](previous_token.reshape(B*V,C,P).detach(), current_token.reshape(B*V,C,P).detach(), pos1=model.aggregator.pos, pos2=model.aggregator.pos).reshape(B,V,C,P)
                        # aggregate_token_attn = self.aggregator.cross_blocks[idx](current_token.reshape(B*V,C,P), previous_token.reshape(B*V,C,P), pos1=self.aggregator.pos, pos2=self.aggregator.pos).reshape(B,V,C,P)
                        aggregated_motion_tokens_list_bw.append(aggregate_token_attn_bw)
                        aggregated_motion_tokens_list_fw.append(aggregate_token_attn_fw)

                    # Backward
                    concat_points = torch.cat([world_points_last.permute(0,1,4,2,3).detach(), predictions["world_points"].permute(0,1,4,2,3).detach()], dim=2)
                    concat_images = torch.cat([last_time_images, images], dim=2)
                    motions = model.motion_head(
                        aggregated_motion_tokens_list_bw, images=concat_images, pc=concat_points, patch_start_idx=patch_start_idx
                    )

                    motion_maps_bw = motions[:, :, :3]

                    points_local = torch.einsum('BVDC,BVCHW->BVDHW', predictions['gaussians'][1]["extrs"][:,:,:3,:3].detach(), predictions['xyz'].detach())+predictions['gaussians'][1]["extrs"][:,:,:3,3].unsqueeze(-1).unsqueeze(-1).detach()
                    points_local_3d = torch.einsum('BVDC,BVCHW->BVDHW', predictions['gaussians'][1]["intrs"], points_local)
                    points_local_2d = points_local_3d[:,:,:2]/(points_local_3d[:,:,2:]+1e-8)

                    points_local_trans = torch.einsum('BVDC,BVCHW->BVDHW', predictions['gaussians'][1]["extrs"][:,:,:3,:3].detach(), predictions['xyz'].detach() + motion_maps_bw)+predictions['gaussians'][1]["extrs"][:,:,:3,3].unsqueeze(-1).unsqueeze(-1).detach()
                    points_local_3d_trans = torch.einsum('BVDC,BVCHW->BVDHW', predictions['gaussians'][1]["intrs"], points_local_trans)
                    points_local_2d_trans = points_local_3d_trans[:,:,:2]/(points_local_3d_trans[:,:,2:]+1e-8)

                    predictions["flow_bw"] = points_local_2d_trans - points_local_2d

                    # Forward
                    concat_points = torch.cat([predictions["world_points"].permute(0,1,4,2,3).detach(), world_points_last.permute(0,1,4,2,3).detach()], dim=2)
                    concat_images = torch.cat([images, last_time_images], dim=2)
                    motions = model.motion_head(
                        aggregated_motion_tokens_list_fw, images=concat_images, pc=concat_points, patch_start_idx=patch_start_idx
                    )
                    motion_maps_fw = motions[:, :, :3]

                    points_local = torch.einsum('BVDC,BVCHW->BVDHW', extrs_float[:,:,:3,:3].detach(), predictions['gaussians'][0]['xyz'].detach())+extrs_float[:,:,:3,3].unsqueeze(-1).unsqueeze(-1).detach()
                    points_local_3d = torch.einsum('BVDC,BVCHW->BVDHW', intrs_float, points_local)
                    points_local_2d = points_local_3d[:,:,:2]/(points_local_3d[:,:,2:]+1e-8)

                    points_local_trans = torch.einsum('BVDC,BVCHW->BVDHW', extrs_float[:,:,:3,:3].detach(), predictions['gaussians'][0]['xyz'].detach() + motion_maps_fw)+extrs_float[:,:,:3,3].unsqueeze(-1).unsqueeze(-1).detach()
                    points_local_3d_trans = torch.einsum('BVDC,BVCHW->BVDHW', intrs_float, points_local_trans)
                    points_local_2d_trans = points_local_3d_trans[:,:,:2]/(points_local_3d_trans[:,:,2:]+1e-8)

                    predictions["flow_fw"] = points_local_2d_trans - points_local_2d

                    gs_position_fw = predictions['gaussians'][0]['xyz'].detach() + motion_maps_fw
                    gs_position_bw = predictions['xyz'].detach() + motion_maps_bw

                    origin_gaussian_0 = torch.cat([predictions["gaussians"][0]['rgb'], predictions["gaussians"][0]['rot_maps'], predictions["gaussians"][0]['opacity_maps'], predictions["gaussians"][0]['scale_maps']], dim=2)
                    origin_gaussian_1 = torch.cat([predictions["gaussians"][1]['rgb'], predictions["gaussians"][1]['rot_maps'], predictions["gaussians"][1]['opacity_maps'], predictions["gaussians"][1]['scale_maps']], dim=2)

                    # forward mask
                    tenOne = (predictions["xyz"].detach()*predictions["gaussians"][1]['masks']).reshape(B*V,3,H,W).detach()
                    tenTwo = (gs_position_fw*predictions["gaussians"][0]['masks']).reshape(B*V,3,H,W).detach()

                    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
                    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
                    xx = xx.view(1, H, W, 1).repeat(B*V, 1, 1, 1)
                    yy = yy.view(1, H, W, 1).repeat(B*V, 1, 1, 1)
                    grid = torch.cat((xx, yy), 3).float()
                    vgrid = grid.to(predictions["flow_fw"]) + predictions["flow_fw"].reshape(B*V, 2, H, W).permute(0, 2, 3, 1)
                    vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
                    vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
                    tenSoftmax = F.grid_sample(tenOne, vgrid, align_corners=True)*predictions["gaussians"][0]['masks'].reshape(B*V,1,H,W)
                    gaussian_sampled_fw = F.grid_sample(origin_gaussian_1.reshape(B*V,-1,H,W), vgrid, align_corners=True).reshape(B,V,-1,H,W)

                    # flow_matching_mask = (torch.sqrt((tenTwo - tenSoftmax).pow(2).mean(dim=1)).reshape(B, V, 1, H, W) > 0.05*scale.unsqueeze(dim=-1).unsqueeze(dim=-1)).float()
                    # predictions["gaussians"][0]['masks'] = predictions["gaussians"][0]['masks'] * flow_matching_mask

                    dist_fw = torch.sqrt((tenTwo - tenSoftmax).pow(2).mean(dim=1)).reshape(B, V, 1, H, W)/scale.unsqueeze(dim=-1).unsqueeze(dim=-1)
                    flow_matching_mask_fw = (dist_fw > 0.05).float()
                    predictions["gaussians"][0]['masks'] = predictions["gaussians"][0]['masks'] * flow_matching_mask_fw
                    # fused_fw = model.fusion(torch.cat([origin_gaussian_0, gaussian_sampled_fw, dist_fw], dim=2).reshape(B*V,-1,H,W)).reshape(B,V,-1,H,W)

                    # predictions["gaussians"][0]["rgb"] = predictions["gaussians"][0]['rgb']+0.1*(nn.functional.sigmoid(fused_fw[:, :, :3])-0.5)
                    # predictions["gaussians"][0]["rot_maps"] = quaternion_multiply(predictions["gaussians"][0]['rot_maps'], nn.functional.normalize(fused_fw[:, :, 3:7] + 1e-8, dim=2))
                    # predictions["gaussians"][0]["opacity_maps"] = predictions["gaussians"][0]['opacity_maps']*nn.functional.sigmoid(fused_fw[:, :, 7:8])
                    # predictions["gaussians"][0]["scale_maps"] = predictions["gaussians"][0]['scale_maps']+fused_fw[:, :, 8:11]

                    # backward mask
                    tenOne = (predictions["gaussians"][0]["xyz"].detach()*predictions["gaussians"][0]['masks']).reshape(B*V,3,H,W).detach()
                    tenTwo = (gs_position_bw*predictions["gaussians"][1]['masks']).reshape(B*V,3,H,W).detach()

                    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
                    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
                    xx = xx.view(1, H, W, 1).repeat(B*V, 1, 1, 1)
                    yy = yy.view(1, H, W, 1).repeat(B*V, 1, 1, 1)
                    grid = torch.cat((xx, yy), 3).float()
                    vgrid = grid.to(predictions["flow_bw"]) + predictions["flow_bw"].reshape(B*V, 2, H, W).permute(0, 2, 3, 1)
                    vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
                    vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
                    tenSoftmax = F.grid_sample(tenOne, vgrid, align_corners=True)
                    gaussian_sampled_bw = F.grid_sample(origin_gaussian_0.reshape(B*V,-1,H,W), vgrid, align_corners=True).reshape(B, V,-1,H,W)

                    dist_bw = torch.sqrt((tenTwo - tenSoftmax).pow(2).mean(dim=1)).reshape(B, V, 1, H, W)/scale.unsqueeze(dim=-1).unsqueeze(dim=-1)
                    flow_matching_mask_bw = (dist_bw > 0.05).float()
                    predictions["gaussians"][1]['masks'] = predictions["gaussians"][1]['masks'] * flow_matching_mask_bw
                    fused_bw = model.fusion(torch.cat([origin_gaussian_1, gaussian_sampled_bw, dist_bw], dim=2).reshape(B*V,-1,H,W)).reshape(B,V,-1,H,W)

                    # predictions["gaussians"][1]["rgb"] = predictions["gaussians"][1]['rgb']+0.1*(nn.functional.sigmoid(fused_bw[:, :, :3])-0.5)
                    # predictions["gaussians"][1]["rot_maps"] = quaternion_multiply(predictions["gaussians"][1]['rot_maps'], nn.functional.normalize(fused_bw[:, :, 3:7] + 1e-8, dim=2))
                    # predictions["gaussians"][1]["opacity_maps"] = predictions["gaussians"][1]['opacity_maps']*nn.functional.sigmoid(fused_bw[:, :, 7:8])
                    # predictions["gaussians"][1]["scale_maps"] = predictions["gaussians"][1]['scale_maps']+fused_bw[:, :, 8:11]

                    mask_indice = (data['masks']>0.66).reshape(-1,1).squeeze()
                    points_start = predictions['xyz'].permute(0,1,3,4,2).reshape(-1,3)[mask_indice]
                    points_end = (motion_maps_bw+predictions['xyz']).permute(0,1,3,4,2).reshape(-1,3)[mask_indice]

                    # savepath = f'vis/motion/{frame_id:05d}.pdf'


                    fused_gs_bw = {}
                    fused_gs_bw["rgb"] = predictions["gaussians"][1]['rgb']+0.1*(nn.functional.sigmoid(fused_bw[:, :, :3])-0.5)
                    fused_gs_bw["masks"] = predictions["gaussians"][1]['masks'] * (1-flow_matching_mask_bw)
                    fused_gs_bw["rot_maps"] = quaternion_multiply(predictions["gaussians"][1]['rot_maps'], nn.functional.normalize(fused_bw[:, :, 3:7] + 1e-8, dim=2))
                    fused_gs_bw["opacity_maps"] = predictions["gaussians"][1]['opacity_maps']*nn.functional.sigmoid(fused_bw[:, :, 7:8])
                    fused_gs_bw["scale_maps"] = predictions["gaussians"][1]['scale_maps']+fused_bw[:, :, 8:11]

                    for time_interpolation in range(1, extend_frames):
                        gs_position_novelt_bw = predictions['xyz'].detach() + motion_maps_bw*(1-time_interpolation/extend_frames)
                        gs_position_novelt_fw = predictions['gaussians'][0]['xyz'].detach() + motion_maps_fw*time_interpolation/extend_frames

                        gaussians = {"xyz": torch.cat([gs_position_novelt_fw, gs_position_novelt_bw, gs_position_novelt_bw], dim=1), 
                                    "rgb": torch.cat([predictions["gaussians"][0]['rgb'], predictions["gaussians"][1]['rgb'], fused_gs_bw['rgb']], dim=1), 
                                    "masks": torch.cat([predictions["gaussians"][0]['masks'], predictions["gaussians"][1]['masks'], fused_gs_bw['masks']], dim=1),
                                    "rot_maps": torch.cat([predictions["gaussians"][0]['rot_maps'], predictions["gaussians"][1]['rot_maps'],fused_gs_bw['rot_maps']], dim=1), 
                                    "opacity_maps": torch.cat([predictions["gaussians"][0]['opacity_maps'], predictions["gaussians"][1]['opacity_maps'], fused_gs_bw['opacity_maps']], dim=1), 
                                    "scale_maps": torch.cat([predictions["gaussians"][0]['scale_maps'], predictions["gaussians"][1]['scale_maps'], fused_gs_bw['scale_maps']], dim=1)}

                        extrs_float = novel_extr.pop().unsqueeze(dim=0).repeat(1,4,1,1)
                        synthesized_novel_time = render_gaussians(gaussians, gaussians["xyz"], gaussians["rgb"], gaussians["masks"], extrs_float.detach(), intrs_float, bg_color=bg_color)
                        all_synthesized_images.append(synthesized_novel_time[0,2].permute(1,2,0).cpu())

                        vis = render_gaussians(gaussians, gaussians["xyz"], gaussians["rgb"], gaussians["masks"], extr_novel.detach(), intr_novel, bg_color=bg_color, render_vis=True)
                        cv2.imwrite(os.path.join(gs_dir, f"{frame_id:05d}_v1.jpg"), 255*vis[0,0].permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
                        cv2.imwrite(os.path.join(gs_dir, f"{frame_id:05d}_v2.jpg"), 255*vis[0,1].permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
                        cv2.imwrite(os.path.join(gs_dir, f"{frame_id:05d}_v3.jpg"), 255*vis[0,2].permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())

                        vis = render_gaussians(gaussians, gaussians["xyz"], gaussians["rgb"], gaussians["masks"], extr_novel.detach(), intr_novel, bg_color=bg_color)
                        cv2.imwrite(os.path.join(render_dir, f"{frame_id:05d}_v1.jpg"), 255*vis[0,0].permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
                        cv2.imwrite(os.path.join(render_dir, f"{frame_id:05d}_v2.jpg"), 255*vis[0,1].permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
                        cv2.imwrite(os.path.join(render_dir, f"{frame_id:05d}_v3.jpg"), 255*vis[0,2].permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
                        frame_id += 1

                last_time_images = images
                aggregated_tokens_last_list = aggregated_tokens_list
                world_points_last = predictions["world_points"]
                if len(predictions["gaussians"])>1:
                    predictions["gaussians"] = predictions["gaussians"][1:]
        # ---- save the synthesized novel-view / novel-time video ----
        if len(all_synthesized_images) > 0:
            all_synthesized_images = torch.stack(all_synthesized_images, dim=0)
            frames_np = (all_synthesized_images.cpu().numpy() * 255).astype(np.uint8)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path = os.path.join(video_dir, f"{test_id}_{extend_frames}.mp4")
            video_writer = cv2.VideoWriter(
                video_path, fourcc, 15.0 * extend_frames / sampling_rate, (W, H))
            for frame in frames_np:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            video_writer.release()
            print(f"Saved video: {video_path}")

    print("\nInference finished.")


if __name__ == "__main__":
    main()
