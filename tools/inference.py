import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lib.GaussianRender import pts2render
import numpy as np
import os
from pathlib import Path
import argparse

from codes.Forge4D.Forge4D.forge4d.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding

import cv2
from codes.Forge4D.Forge4D.forge4d.models.forge4d import VGGT
from lib.dna_dataset import DNADataset
from lib.utils import print_gpu_memory, quaternion_multiply

import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def concatenate_videos(video_paths, output_path, layout='horizontal'):
    """Concatenate multiple videos into a single video."""
    caps = [cv2.VideoCapture(path) for path in video_paths]

    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: could not open video {video_paths[i]}")
            return

    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = caps[0].get(cv2.CAP_PROP_FPS)

    if layout == 'horizontal':
        total_width = width * len(caps)
        total_height = height
    else:
        total_width = width
        total_height = height * len(caps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, total_height))

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            frames.append(frame)

        if not any([cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in caps]):
            break

        if layout == 'horizontal':
            combined_frame = np.hstack(frames)
        else:
            combined_frame = np.vstack(frames)

        out.write(combined_frame)

    for cap in caps:
        cap.release()
    out.release()

    print(f"Video saved: {output_path}")


def plot_3d_trajectories(trajectory_data, savepath='3d_trajectories.png'):
    """Plot 3D trajectories with random colors per trajectory."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    num_sample = int(0.1 * len(trajectory_data))
    indices = random.sample(range(len(trajectory_data)), num_sample)
    indices = torch.tensor(indices, device="cuda")
    trajectory_data = trajectory_data[indices]

    trajectory_data = trajectory_data.cpu().numpy()

    for i in range(len(trajectory_data)):
        start_point = trajectory_data[i, 0, :]
        end_point = trajectory_data[i, 1, :]
        color = np.random.rand(3,)
        ax.plot([start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]],
                color=color, linewidth=3, alpha=0.8)

    ax.set_xlabel('X Axis', fontsize=12, labelpad=10)
    ax.set_ylabel('Y Axis', fontsize=12, labelpad=10)
    ax.set_zlabel('Z Axis', fontsize=12, labelpad=10)
    ax.set_title('3D Trajectories Visualization', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)

    max_range = np.array([
        trajectory_data[:, :, 0].max() - trajectory_data[:, :, 0].min(),
        trajectory_data[:, :, 1].max() - trajectory_data[:, :, 1].min(),
        trajectory_data[:, :, 2].max() - trajectory_data[:, :, 2].min()
    ]).max() / 2.0

    mid_x = (trajectory_data[:, :, 0].max() + trajectory_data[:, :, 0].min()) * 0.5
    mid_y = (trajectory_data[:, :, 1].max() + trajectory_data[:, :, 1].min()) * 0.5
    mid_z = (trajectory_data[:, :, 2].max() + trajectory_data[:, :, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description="Forge4D Inference")
    parser.add_argument("--weight", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--test_ids", type=str, nargs="+", default=["0007_04"], help="Test subject IDs")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to process")
    parser.add_argument("--sampling_rate", type=int, default=1, help="Frame sampling rate")
    parser.add_argument("--stage", type=str, default="dual-matching", choices=["dual-matching", "no-fusion", "single", "dual"])
    parser.add_argument("--resolution", type=int, default=518, help="Input resolution")
    parser.add_argument("--motion_dim", type=int, default=3, help="Motion dimension (3 for single view, 4 for dual)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VGGT(motion_dimension=args.motion_dim)
    checkpoint = torch.load(args.weight, map_location=device)
    state_dict = checkpoint.get('network', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device).eval()
    print(f"Loaded checkpoint from {args.weight}")

    dataset = DNADataset(
        type('Opt', (), {
            'data_root': args.data_root,
            'resolution': args.resolution,
            'train_view_num': 4,
            'use_processed_data': False,
        })(),
        phase='val',
        sampling_rate=args.sampling_rate,
        max_frames=args.frames,
        test_ids=args.test_ids,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    iterator = iter(loader)

    print(f"Dataset length: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError("Dataset is empty!")

    base_dir = args.output_dir
    os.makedirs(os.path.join(base_dir, 'render'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, '3DGS'), exist_ok=True)

    predictions = {
        "pose_enc": [], "world_points": [], "images": [], "novel_img": [],
        "pose_gt": [], "retarget_images": [], "novel_time": [],
        "novel_time&view": [], "gaussians": [], "gt_scene_flow": [],
        "scene_flow_mask": [], "motions": [],
    }

    for t in range(args.frames // args.sampling_rate):
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            data = next(iterator)

        print(f"Processing time step: {t}")

        attr_list = ["imgs", "masks", "novel_time", "novel_time_mask"]
        for item in attr_list:
            if item in data:
                data[item] = data[item].to(device)

        for item in ["novel_view", "camera"]:
            if item in data:
                for k in data[item].keys():
                    data[item][k] = data[item][k].to(device)

        bg_color = [0.0, 0.0, 0.0]
        mask = (data['masks'] > 0.66).float()
        data['imgs'] = data['imgs'] * mask + (1 - mask) * torch.tensor(bg_color).view(1, 1, 3, 1, 1).to(data['imgs'])

        images = data['imgs']
        masks = data['masks']
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, V, C, H, W = images.shape

        scene_token = model.scene_token.unsqueeze(dim=1).repeat(1, B, 1, 1)

        aggregated_tokens_list, aggregated_scene_tokens_list, patch_start_idx, scene_token = model.aggregator(images, scene_token)

        if model.camera_head is not None:
            pose_enc_list = model.camera_head(aggregated_tokens_list)
            predictions["pose_enc"] = pose_enc_list[-1]

        if model.point_head is not None:
            pts3d, pts3d_conf = model.point_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)
            predictions["world_points"] = pts3d

        print(f"Step {t} complete. GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")


if __name__ == "__main__":
    main()
