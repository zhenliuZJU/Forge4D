# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision import transforms as TF
from torch.utils.data import Dataset

import random
import numpy as np
import os
from PIL import Image, ImageFile
import cv2
import torch
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from pathlib import Path
import logging
import json
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def save_np_to_json(parm, save_name):
    for key in parm.keys():
        parm[key] = parm[key].tolist()
    with open(save_name, 'w') as file:
        json.dump(parm, file, indent=1)


def load_json_to_np(parm_name):
    with open(parm_name, 'r') as f:
        parm = json.load(f)
    for key in parm.keys():
        parm[key] = np.array(parm[key])
    return parm


def depth2pts(depth, extrinsic, intrinsic):
    # depth H W extrinsic 3x4 intrinsic 3x3 pts map H W 3
    pts_map = []
    for view in range(depth.shape[0]):
        rot = extrinsic[view, :3, :3]
        trans = extrinsic[view, :3, 3:]
        S, S = depth[view, 0].shape

        y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device),
                            torch.linspace(0.5, S-0.5, S, device=depth.device))
        pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # H W 3

        pts_2d[..., 2] = depth[view, 0]
        pts_2d[..., 0] -= intrinsic[view, 0, 2]
        pts_2d[..., 1] -= intrinsic[view, 1, 2]
        pts_2d_xy = pts_2d[..., :2] * pts_2d[..., 2:]
        pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

        pts_2d[..., 0] /= intrinsic[view, 0, 0]
        pts_2d[..., 1] /= intrinsic[view, 1, 1]
        pts_2d = pts_2d.reshape(-1, 3).T
        pts = rot.T @ pts_2d - rot.T @ trans
        # pts = pts_2d

        pts_map.append(pts.view(3, S, S))
    
    return torch.stack(pts_map, dim=0)


def pts2depth(ptsmap, extrinsic, intrinsic):
    S, S, _ = ptsmap.shape
    pts = ptsmap.view(-1, 3).T
    calib = intrinsic @ extrinsic
    pts = calib[:3, :3] @ pts
    pts = pts + calib[:3, 3:4]
    pts[:2, :] /= (pts[2:, :] + 1e-8)
    depth = 1.0 / (pts[2, :].view(S, S) + 1e-8)
    return depth


def stereo_pts2flow(pts0, pts1, rectify0, rectify1, Tf_x):
    new_extr0, new_intr0, rectify_mat0_x, rectify_mat0_y = rectify0
    new_extr1, new_intr1, rectify_mat1_x, rectify_mat1_y = rectify1
    new_depth0 = pts2depth(torch.FloatTensor(pts0), torch.FloatTensor(new_extr0), torch.FloatTensor(new_intr0))
    new_depth1 = pts2depth(torch.FloatTensor(pts1), torch.FloatTensor(new_extr1), torch.FloatTensor(new_intr1))
    new_depth0 = new_depth0.detach().numpy()
    new_depth1 = new_depth1.detach().numpy()
    new_depth0 = cv2.remap(new_depth0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
    new_depth1 = cv2.remap(new_depth1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)

    offset0 = new_intr1[0, 2] - new_intr0[0, 2]
    disparity0 = -new_depth0 * Tf_x
    flow0 = offset0 - disparity0

    offset1 = new_intr0[0, 2] - new_intr1[0, 2]
    disparity1 = -new_depth1 * (-Tf_x)
    flow1 = offset1 - disparity1

    flow0[new_depth0 < 0.05] = 0
    flow1[new_depth1 < 0.05] = 0

    return flow0, flow1


def read_img(name):
    img = np.array(Image.open(name))
    return img


def read_depth(name):
    return cv2.imread(name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 2.0 ** 15


class GenebodyDataset(Dataset):
    def __init__(self, opt, phase='train', test_ids=None, sampling_rate=5, timesteps=2):
        self.opt = opt
        self.train_novel_id = [28,40]
        self.val_novel_id = [28,40]
        self.use_processed_data = opt.use_processed_data
        self.phase = phase
        self.dataset_path = opt.data_root
        self.source_id = [25,31,37,43]
        self.timesteps = timesteps
        self.sampling_rate = sampling_rate
        # self.source_id = [1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46]

        id_path = os.path.join(self.dataset_path, 'Genebody/main')
        # # temporal compromise for the dataset
        # test_ids = ["304", "305"]
        if test_ids is None:
            # test_ids = ["kamal_ejaz"]
            test_ids = os.path.join(self.dataset_path, 'Genebody/main')

        self.sample_list = []
        if self.phase == 'train':
            human_ids = os.listdir(id_path)
            for human_id in human_ids:
                # # temporal compromise for the dataset
                # if not any(ids in human_id for ids in train_ids):
                    # continue
                    
                if any(ids in human_id for ids in test_ids):
                    continue
                img_path = os.path.join(self.dataset_path, 'Genebody/main', human_id, 'image', '00')
                files = [f for f in os.listdir(img_path) if f.endswith(".jpg")]

                min_timestep = min([int(f.split(".")[0]) for f in files], default=-1)
                max_timestep = max([int(f.split(".")[0]) for f in files], default=-1)
                # # temporal compromise for the dataset
                # max_timestep=min(300,max_timestep)
                if max_timestep != -1:
                    self.sample_list += [(human_id, t) for t in range(min_timestep, max_timestep-sampling_rate*(timesteps-1), sampling_rate)]
        else:
            human_ids = os.listdir(id_path)
            for human_id in human_ids:
                if not any(ids in human_id for ids in test_ids):
                    continue
                img_path = os.path.join(self.dataset_path, 'Genebody/main', human_id, 'image', '00')
                files = [f for f in os.listdir(img_path) if f.endswith(".jpg")]

                min_timestep = min([int(f.split(".")[0]) for f in files], default=-1)
                max_timestep = max([int(f.split(".")[0]) for f in files], default=-1)
                # # temporal compromise for the dataset
                # max_timestep=min(300,max_timestep)
                sampling_rate = timesteps
                if max_timestep != -1:
                    self.sample_list += [(human_id, t) for t in range(min_timestep, max_timestep-2*(timesteps-1), sampling_rate)] #range(-1,2)

    def load_novel_view(self, sample_name, view_ids):
        human_id, frame_id = sample_name
        intrs, extrs, imgs, masks = [], [], [], []
        novel_time, novel_time_mask = [], []
        for step in range(self.timesteps):
            timestep = frame_id + step * self.sampling_rate

            img_names = [os.path.join(self.dataset_path, 'Genebody/main', human_id, f'image/{source_id:02d}', f'{timestep:05d}.png' if human_id in ['rabbi', 'zhangziyu'] else f'{timestep:04d}.jpg') for source_id in view_ids]
            mask_names = [os.path.join(self.dataset_path, 'Genebody/main', human_id, f'mask/{source_id:02d}', f'mask{timestep:05d}.png' if human_id in ['rabbi', 'zhuna', 'zhangziyu'] else f'mask{timestep:04d}.png') for source_id in view_ids]
            cam_names = os.path.join(self.dataset_path, 'Genebody/main', human_id, 'annots.npy')
            img, intr = load_and_preprocess_images_intrs(img_names, cam_names, [source_id  for source_id in view_ids], self.opt.resolution)
            mask = load_and_preprocess_masks(mask_names, self.opt.resolution).squeeze(dim=0)

            extr, _ = load_extrs(cam_names, [source_id for source_id in view_ids], normalize=False) 

            intrs.append(intr)
            extrs.append(extr)
            imgs.append(img)
            masks.append(mask)

            if step != self.timesteps-1:
                timestep = frame_id + step * self.sampling_rate + self.sampling_rate//2

                img_names = [os.path.join(self.dataset_path, 'Genebody/main', human_id, f'image/{source_id:02d}', f'{timestep:05d}.png' if human_id in ['rabbi', 'zhangziyu'] else f'{timestep:04d}.jpg') for source_id in view_ids]
                mask_names = [os.path.join(self.dataset_path, 'Genebody/main', human_id, f'mask/{source_id:02d}', f'mask{timestep:05d}.png' if human_id in ['rabbi', 'zhuna', 'zhangziyu'] else f'mask{timestep:04d}.png') for source_id in view_ids]
                img, intr = load_and_preprocess_images_intrs(img_names, cam_names, [source_id  for source_id in view_ids], self.opt.resolution)
                mask = load_and_preprocess_masks(mask_names, self.opt.resolution).squeeze(dim=0)

                novel_time.append(img)
                novel_time_mask.append(mask)

        return torch.stack(imgs, dim=0), torch.stack(masks, dim=0), torch.stack(intrs, dim=0), torch.stack(extrs, dim=0), torch.stack(novel_time, dim=0), torch.stack(novel_time_mask, dim=0)
        
    def get_novel_view_tensor(self, sample_name, view_id):
        img, mask, intr, w2c, novel_time, novel_time_mask = self.load_novel_view(sample_name, view_id)
        width, height = img.shape[2], img.shape[3]
        w2c = torch.einsum('tvij, tjk->tvik', [w2c, self.c_one_2_w])
        w2c[:, :, :3, 3] = w2c[:, :, :3, 3] / self.scale

        novel_view_data = {
            'view_id': torch.IntTensor([view_id]),
            'img': img,
            'mask': mask,
            'extr': torch.FloatTensor(w2c),
            'intr': torch.FloatTensor(intr),
            'novel_time': novel_time,
            'novel_time_mask': novel_time_mask
        }

        return novel_view_data

    def get_origin_data(self, sample_name, require_mask=True):
        human_id, frame_id = sample_name
        intrs, extrs, imgs, masks, c_one_2_w_s = [], [], [], [], []
        novel_time, novel_time_mask = [], []
        for step in range(self.timesteps):
            timestep = frame_id + step * self.sampling_rate

            img_names = [os.path.join(self.dataset_path, 'Genebody/main', human_id, f'image/{self.input_cams[i]:02d}', f'{timestep:05d}.jpg' if human_id in ['rabbi', 'zhangziyu'] else f'{timestep:04d}.jpg') for i in range(self.opt.train_view_num)] 
            mask_names = [os.path.join(self.dataset_path, 'Genebody/main', human_id, f'mask/{self.input_cams[i]:02d}', f'mask{timestep:05d}.png' if human_id in ['rabbi', 'zhuna', 'zhangziyu'] else f'mask{timestep:04d}.png')  for i in range(self.opt.train_view_num)] 
            cam_names = os.path.join(self.dataset_path, 'Genebody/main', human_id, 'annots.npy')
            # depth_names = [os.path.join(self.dataset_path, 'depth', human_id, f'{timestep:05d}_{self.input_cams[i]:02d}.png') for i in range(self.opt.train_view_num)] 
            # depth_names = [os.path.join(self.dataset_path, human_id, f'{timestep:05d}_{self.input_cams[i]:02d}.png') for i in range(self.opt.train_view_num)] 
            img, intr = load_and_preprocess_images_intrs(img_names, cam_names, [input_cams for input_cams in self.input_cams], self.opt.resolution)
            mask = load_and_preprocess_masks(mask_names, self.opt.resolution)

            # depth = load_and_preprocess_depth(depth_names)
            extr, c_one_2_w = load_extrs(cam_names, [input_cams for input_cams in self.input_cams]) 

            intrs.append(intr)
            extrs.append(extr)
            imgs.append(img)
            masks.append(mask)
            c_one_2_w_s.append(c_one_2_w)
            
            if step != self.timesteps-1:
                timestep = frame_id + step * self.sampling_rate + self.sampling_rate//2

                img_names = [os.path.join(self.dataset_path, 'Genebody/main', human_id, f'image/{self.input_cams[i]:02d}', f'{timestep:05d}.jpg' if human_id in ['rabbi', 'zhangziyu'] else f'{timestep:04d}.jpg') for i in range(self.opt.train_view_num)] 
                mask_names = [os.path.join(self.dataset_path, 'Genebody/main', human_id, f'mask/{self.input_cams[i]:02d}', f'mask{timestep:05d}.png' if human_id in ['rabbi', 'zhuna', 'zhangziyu'] else f'mask{timestep:04d}.png')  for i in range(self.opt.train_view_num)] 
                img, intr = load_and_preprocess_images_intrs(img_names, cam_names, self.input_cams, self.opt.resolution)
                mask = load_and_preprocess_masks(mask_names, self.opt.resolution)

                novel_time.append(img)
                novel_time_mask.append(mask)

        self.scale = 1
        self.c_one_2_w = torch.stack(c_one_2_w_s, dim=0)

        # scaling the data
        # extrs[:, :3, 3] = extrs[:, :3, 3] / self.scale
        # depth = depth / self.scale
        # pointmap = pointmap / self.scale

        depth, pointmap = torch.tensor(0), torch.tensor(0)

        camera = {
            'intr': torch.stack(intrs, dim=0),
            'extr': torch.stack(extrs, dim=0)
        }

        data = {
            'scene': human_id,
            'frame': frame_id,
            'imgs': torch.stack(imgs, dim=0), # T, V, C, H, W
            'masks': torch.stack(masks, dim=0),
            'depth': depth,
            'pointmap': pointmap,
            'camera': camera,
            'novel_time': torch.stack(novel_time, dim=0),
            'novel_time_mask': torch.stack(novel_time_mask, dim=0)
        }

        return data

    def get_item(self, index, novel_id=None):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]
        # rand_shift = 3*random.randint(0, 16)
        # rand_shift = 3*random.randint(-1, 1)
        # rand_shift = 0
        if len(self.source_id) == self.opt.train_view_num:
            self.input_cams = self.source_id
            # for i in range(len(self.input_cams)):
            #     self.input_cams[i] = (self.input_cams[i]+rand_shift)%48 
        else:
            self.input_cams = random.sample(self.source_id, self.opt.train_view_num)
            # for i in range(len(self.input_cams)):
            #     self.input_cams[i] = (self.input_cams[i]+rand_shift)%48
        dict_tensor = self.get_origin_data(sample_name, require_mask=True)

        if not novel_id:
            novel_id = self.val_cams
        
        # novel_id = np.random.choice(novel_id) #(np.random.choice(novel_id)+rand_shift)%48
        dict_tensor.update({
            'novel_view': self.get_novel_view_tensor(sample_name, novel_id)
        })
        # print("rand_shift:", rand_shift)
        # print(self.input_cams)
        # print('novel_id:', novel_id)

        return dict_tensor

    def get_test_item(self, index, source_id):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]

        if self.use_processed_data:
            logging.error('test data loader not support processed data')

        view0_data = self.load_single_view(sample_name, source_id[0], hr_img=False, require_mask=True, require_pts=False)
        view1_data = self.load_single_view(sample_name, source_id[1], hr_img=False, require_mask=True, require_pts=False)
        lmain_intr_ori, lmain_extr_ori = view0_data[2], view0_data[3]
        rmain_intr_ori, rmain_extr_ori = view1_data[2], view1_data[3]
        stereo_np = self.get_rectified_stereo_data(main_view_data=view0_data, ref_view_data=view1_data)
        dict_tensor = self.stereo_to_dict_tensor(stereo_np, sample_name)

        dict_tensor['lmain']['intr_ori'] = torch.FloatTensor(lmain_intr_ori)
        dict_tensor['rmain']['intr_ori'] = torch.FloatTensor(rmain_intr_ori)
        dict_tensor['lmain']['extr_ori'] = torch.FloatTensor(lmain_extr_ori)
        dict_tensor['rmain']['extr_ori'] = torch.FloatTensor(rmain_extr_ori)

        img_len = 2048 if self.opt.use_hr_img else 1024
        novel_dict = {
            'height': torch.IntTensor([img_len]),
            'width': torch.IntTensor([img_len])
        }

        dict_tensor.update({
            'novel_view': novel_dict
        })

        return dict_tensor

    def __getitem__(self, index):
        if self.phase == 'train':
            return self.get_item(index, novel_id=self.train_novel_id)
        elif self.phase == 'val':
            return self.get_item(index, novel_id=self.val_novel_id)

    def __len__(self):
        self.train_boost = 50
        self.val_boost = 200
        return len(self.sample_list)


def load_and_preprocess_images(image_path_list, res=518, mode="crop", crop_rate=0.2):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = res# 518

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size
        bbox = (int(crop_rate*height)+200, int((crop_rate)*height), width-int(crop_rate*height)-200, height-int((crop_rate)*height))
        img = img.crop(bbox)

        width, height = img.size
        
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_height = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_width = round(width * (new_height / height) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_width > target_size:
            start_x = (new_width - target_size) // 2
            img = img[:, :, start_x : start_x + target_size]
        
        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images

def load_and_preprocess_masks(image_path_list, res=518, mode="crop", crop_rate=0.2):
        # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = res #518

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Open image
        img = Image.open(image_path)

        # Now convert to "L" (this step assigns white for transparent areas)
        img = img.convert("L")

        width, height = img.size
        bbox = (int(crop_rate*height)+200, int((crop_rate)*height), width-int(crop_rate*height)-200, height-int((crop_rate)*height))
        img = img.crop(bbox)

        width, height = img.size
        
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_height = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_width = round(width * (new_height / height) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.NEAREST)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_width > target_size:
            start_x = (new_width - target_size) // 2
            img = img[:, :, start_x : start_x + target_size]
        
        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[0]
            w_padding = target_size - img.shape[1]
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[0]
            w_padding = max_width - img.shape[1]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images

def load_and_preprocess_depth(image_path_list, res=518, mode="crop", crop_rate=0.2):
        # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    target_size = res # 518

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Open image
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 10000

        width, height = img.size
        img = img[int(0.5*(crop_rate)*height):height-int(1.5*(crop_rate)*height), int(crop_rate*width):int(width-crop_rate*width)]

        height, width = img.shape[:2]
        
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_height = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_width = round(width * (new_height / height) / 14) * 14

        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        img = torch.FloatTensor(img).unsqueeze(dim=-1)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_width > target_size:
            start_x = (new_width - target_size) // 2
            img = img[:, :, start_x : start_x + target_size]
        
        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[0]
            w_padding = target_size - img.shape[1]
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img.permute(2,0,1))

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[0]
            w_padding = max_width - img.shape[1]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images

def load_and_preprocess_images_intrs(image_path_list, camera_files, cam_ids, res=518, mode="crop", crop_rate=0.2):
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    intrs = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = res#518
    
    cam = np.load(camera_files, allow_pickle=True).item()

    # First process all images and collect their shapes
    for i, image_path in enumerate(image_path_list):
        intr = np.array(cam["cams"][f"{cam_ids[i]:02d}"]["K"]).reshape(3, 3)
            
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size
        bbox = (int(crop_rate*height)+200, int((crop_rate)*height), width-int(crop_rate*height)-200, height-int((crop_rate)*height))
        img = img.crop(bbox)

        intr[0, 2] -= int(crop_rate*width)+200
        intr[1, 2] -= int(crop_rate*height)

        width, height = img.size
        
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_height = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_width = round(width * (new_height / height) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)
        scale_h = new_height / height
        scale_w = new_width / width
        intr[0, 0] *= scale_w
        intr[0, 2] *= scale_w
        intr[1, 1] *= scale_h
        intr[1, 2] *= scale_h

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_width > target_size:
            start_x = (new_width - target_size) // 2
            img = img[:, :, start_x : start_x + target_size]
            intr[0, 2] -= start_x

        
        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            raise ValueError("Pad mode is not implemented for intrs")
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        intrs.append(torch.from_numpy(intr).float())

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)
    
    intrs = torch.stack(intrs)
    if len(cam_ids) == 1:
        if intrs.dim() == 2:
            intrs = intrs.unsqueeze(0)

    return images, intrs

def load_extrs(camera_files, cam_ids, normalize=True):
    cam = np.load(camera_files, allow_pickle=True).item()

    extrs = []
    c_one_2_w = None
    for cam_id in cam_ids:
        c2w = np.array(cam["cams"][f"{cam_id:02d}"]["c2w"])
        w2c = torch.inverse(torch.FloatTensor(c2w))

        if c_one_2_w==None:
            c_one_2_w = torch.FloatTensor(c2w)

        if normalize:
            w2c = w2c@c_one_2_w
    
        extrs.append(w2c)

    extrs = torch.stack(extrs)
    extrs[:, :3, 3] = extrs[:, :3, 3]
    if len(cam_ids) == 1:
        if extrs.dim() == 2:
            extrs = extrs.unsqueeze(0)

    return extrs, c_one_2_w

def load_intrs(intr_list):
    intrs = []
    for intr_path in intr_list:
        intr = np.load(intr_path, allow_pickle=True)
        intrs.append(torch.from_numpy(intr)).float()

    intrs = torch.stack(intrs)
    if len(intr_list) == 1:
        if intrs.dim() == 2:
            intrs = intrs.unsqueeze(0)

    return intrs
