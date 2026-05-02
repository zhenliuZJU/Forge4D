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
from PIL import Image
import cv2
import torch
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
import json


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
    pts_map = []
    for view in range(depth.shape[0]):
        rot = extrinsic[view, :3, :3]
        trans = extrinsic[view, :3, 3:]
        S, S = depth[view, 0].shape

        y, x = torch.meshgrid(
            torch.linspace(0.5, S-0.5, S, device=depth.device),
            torch.linspace(0.5, S-0.5, S, device=depth.device)
        )
        pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1)

        pts_2d[..., 2] = depth[view, 0]
        pts_2d[..., 0] -= intrinsic[view, 0, 2]
        pts_2d[..., 1] -= intrinsic[view, 1, 2]
        pts_2d_xy = pts_2d[..., :2] * pts_2d[..., 2:]
        pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

        pts_2d[..., 0] /= intrinsic[view, 0, 0]
        pts_2d[..., 1] /= intrinsic[view, 1, 1]
        pts_2d = pts_2d.reshape(-1, 3).T
        pts = rot.T @ pts_2d - rot.T @ trans

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


class DNADataset(Dataset):
    """DNA-Rendering Dataset for Forge4D.

    Loads multi-view video frames with calibrations from the DNA-Rendering dataset.
    Supports training and evaluation modes with configurable input views and novel views.
    """

    def __init__(self, opt, phase='train', sampling_rate=5, max_frames=1e6, test_ids=None, test_timestamp=None):
        self.opt = opt
        self.train_novel_id = [19, 25, 31]
        self.val_novel_id = [16, 19, 22, 25, 28, 31, 34]
        self.phase = phase
        self.dataset_path = opt.data_root
        self.input_cams = [16, 22, 28, 34]
        self.source_id = [16, 22, 28, 34]

        id_path = os.path.join(self.dataset_path, 'DNA-Rendering/main')

        if test_ids is None:
            test_ids = ["0007", "0010", "0030", "0053", "0077", "0089", "0096", "0113", "0124", "0165"]

        self.sample_list = []
        if self.phase == 'train':
            human_ids = os.listdir(id_path)
            for human_id in human_ids:
                if any(ids in human_id for ids in test_ids):
                    continue
                img_path = os.path.join(self.dataset_path, 'DNA-Rendering/main', human_id, 'Cam00', 'mask')
                files = [f for f in os.listdir(img_path) if f.endswith(".png")]
                max_timestep = min(max([int(f.split(".")[0]) for f in files], default=-1), max_frames)
                if max_timestep != -1:
                    if test_timestamp is None:
                        self.sample_list += [(human_id, 0, t) for t in range(0, max_timestep, sampling_rate)]
                    else:
                        self.sample_list += [(human_id, 0, t) for t in test_timestamp]
        else:
            human_ids = os.listdir(id_path)
            for human_id in human_ids:
                if not any(ids in human_id for ids in test_ids):
                    continue
                img_path = os.path.join(self.dataset_path, 'DNA-Rendering/main', human_id, 'Cam00', 'mask')
                files = [f for f in os.listdir(img_path) if f.endswith(".png")]
                max_timestep = min(max([int(f.split(".")[0]) for f in files], default=-1), max_frames)
                if max_timestep != -1:
                    if test_timestamp is None:
                        self.sample_list += [(human_id, 0, t) for t in range(0, max_timestep, sampling_rate)]
                    else:
                        self.sample_list += [(human_id, 0, t) for t in test_timestamp]

    def load_novel_view(self, sample_name, view_ids):
        human_id, cam_shift, timestep = sample_name

        img_names = [
            os.path.join(self.dataset_path, 'DNA-Rendering/main', human_id, f'Cam{(source_id + cam_shift) % 48:02d}', f'{timestep:04d}.jpg')
            for source_id in view_ids
        ]
        mask_names = [
            os.path.join(self.dataset_path, 'DNA-Rendering/main', human_id, f'Cam{(source_id + cam_shift) % 48:02d}/mask', f'{timestep:04d}.png')
            for source_id in view_ids
        ]
        cam_names = os.path.join(self.dataset_path, 'DNA-Rendering/main', human_id, 'calibration_full.json')
        img, intr = load_and_preprocess_images_intrs(
            img_names, cam_names, [(source_id + cam_shift) % 48 for source_id in view_ids], self.opt.resolution
        )
        mask = load_and_preprocess_masks(mask_names, self.opt.resolution).squeeze(dim=0)
        extrs, _ = load_extrs(cam_names, [(source_id + cam_shift) % 48 for source_id in view_ids], normalize=False)

        return img, mask, intr, extrs

    def get_novel_view_tensor(self, sample_name, view_id):
        img, mask, intr, w2c = self.load_novel_view(sample_name, view_id)
        width, height = img.shape[2], img.shape[3]
        w2c = w2c @ self.c_one_2_w
        w2c[:, :3, 3] = w2c[:, :3, 3] / self.scale

        novel_view_data = {
            'view_id': torch.IntTensor([view_id]),
            'img': img,
            'mask': mask,
            'extr': torch.FloatTensor(w2c),
            'intr': torch.FloatTensor(intr),
        }
        return novel_view_data

    def get_origin_data(self, sample_name, require_mask=True):
        human_id, cam_shift, timestep = sample_name
        img_names = [
            os.path.join(self.dataset_path, 'DNA-Rendering/main', human_id, f'Cam{(self.input_cams[i] + cam_shift) % 48:02d}', f'{timestep:04d}.jpg')
            for i in range(self.opt.train_view_num)
        ]
        mask_names = [
            os.path.join(self.dataset_path, 'DNA-Rendering/main', human_id, f'Cam{(self.input_cams[i] + cam_shift) % 48:02d}/mask', f'{timestep:04d}.png')
            for i in range(self.opt.train_view_num)
        ]
        cam_names = os.path.join(self.dataset_path, 'DNA-Rendering/main', human_id, 'calibration_full.json')

        imgs, intrs = load_and_preprocess_images_intrs(
            img_names, cam_names, [(cam + cam_shift) % 48 for cam in self.input_cams], self.opt.resolution
        )
        masks = load_and_preprocess_masks(mask_names, self.opt.resolution)
        extrs, c_one_2_w = load_extrs(cam_names, [(cam + cam_shift) % 48 for cam in self.input_cams])

        self.scale = 1
        self.c_one_2_w = c_one_2_w

        depth, pointmap = torch.tensor(0), torch.tensor(0)

        camera = {
            'intr': intrs,
            'extr': extrs,
            'c2w': c_one_2_w
        }

        data = {
            'imgs': imgs,
            'masks': masks,
            'depth': depth,
            'pointmap': pointmap,
            'camera': camera,
        }
        return data

    def get_item(self, index, novel_id=None):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]

        if len(self.source_id) == self.opt.train_view_num:
            self.input_cams = self.source_id
        else:
            self.input_cams = random.sample(self.source_id, self.opt.train_view_num)

        dict_tensor = self.get_origin_data(sample_name, require_mask=True)

        if not novel_id:
            novel_id = self.val_cams

        dict_tensor.update({
            'novel_view': self.get_novel_view_tensor(sample_name, novel_id)
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
        if self.phase == 'train':
            return len(self.sample_list) * self.train_boost
        elif self.phase == 'val':
            return len(self.sample_list) * self.val_boost
        else:
            return len(self.sample_list)


def load_and_preprocess_images(image_path_list, res=518, mode="crop", crop_rate=0.1):
    """Load and preprocess images for model input.

    Args:
        image_path_list: List of paths to image files.
        mode: "crop" sets width to 518px and center crops height; "pad" preserves
              all pixels by padding to a square.
        crop_rate: Fraction to crop from borders.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W).
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = res

    for image_path in image_path_list:
        img = Image.open(image_path)

        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        img = img.convert("RGB")

        width, height = img.size
        bbox = (
            int(crop_rate * width),
            int((0.5 * crop_rate) * height),
            width - int(crop_rate * width),
            height - int((1.5 * crop_rate) * height)
        )
        img = img.crop(bbox)

        width, height = img.size

        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)

        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y: start_y + target_size, :]

        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)

    if len(image_path_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)

    return images


def load_and_preprocess_masks(image_path_list, res=518, mode="crop", crop_rate=0.1):
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = res

    for image_path in image_path_list:
        img = Image.open(image_path)
        img = img.convert("L")

        width, height = img.size
        bbox = (
            int(crop_rate * width),
            int((0.5 * crop_rate) * height),
            width - int(crop_rate * width),
            height - int((1.5 * crop_rate) * height)
        )
        img = img.crop(bbox)

        width, height = img.size

        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), Image.NEAREST)
        img = to_tensor(img)

        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y: start_y + target_size, :]

        if mode == "pad":
            h_padding = target_size - img.shape[0]
            w_padding = target_size - img.shape[1]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[0]
            w_padding = max_width - img.shape[1]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)

    if len(image_path_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)

    return images


def load_and_preprocess_depth(image_path_list, res=518, mode="crop", crop_rate=0.1):
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    target_size = res

    for image_path in image_path_list:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 10000

        height, width = img.shape[:2]
        img = img[int((0.5 * crop_rate) * height):height - int((1.5 * crop_rate) * height),
                  int(crop_rate * width):int(width - crop_rate * width)]

        height, width = img.shape[:2]

        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        img = torch.FloatTensor(img).unsqueeze(dim=-1)

        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[start_y: start_y + target_size, :, :]

        if mode == "pad":
            h_padding = target_size - img.shape[0]
            w_padding = target_size - img.shape[1]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img.permute(2, 0, 1))

    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[0]
            w_padding = max_width - img.shape[1]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)

    if len(image_path_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)

    return images


def load_and_preprocess_images_intrs(image_path_list, camera_files, cam_ids, res=518, mode="crop", crop_rate=0.1):
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    intrs = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = res

    with open(camera_files, 'r', encoding='utf-8') as f:
        cam = json.load(f)

    for i, image_path in enumerate(image_path_list):
        intr = np.array(cam[f"Cam{cam_ids[i]:02d}"]["K"]).reshape(3, 3)

        img = Image.open(image_path)

        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        img = img.convert("RGB")

        width, height = img.size
        bbox = (
            int(crop_rate * width),
            int((0.5 * crop_rate) * height),
            width - int(crop_rate * width),
            height - int((1.5 * crop_rate) * height)
        )
        img = img.crop(bbox)

        intr[0, 2] -= int(crop_rate * width)
        intr[1, 2] -= int((0.5 * crop_rate) * height)

        width, height = img.size

        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)
        scale_h = new_height / height
        scale_w = new_width / width
        intr[0, 0] *= scale_w
        intr[0, 2] *= scale_w
        intr[1, 1] *= scale_h
        intr[1, 2] *= scale_h

        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y: start_y + target_size, :]
            intr[1, 2] -= start_y

        if mode == "pad":
            raise ValueError("Pad mode is not implemented for intrs")

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        intrs.append(torch.from_numpy(intr).float())

    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)

    if len(image_path_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)

    intrs = torch.stack(intrs)
    if len(cam_ids) == 1 and intrs.dim() == 2:
        intrs = intrs.unsqueeze(0)

    return images, intrs


def load_extrs(camera_files, cam_ids, normalize=True):
    with open(camera_files, 'r', encoding='utf-8') as f:
        cam = json.load(f)

    extrs = []
    c_one_2_w = None
    for cam_id in cam_ids:
        w2c = torch.eye(4, 4)
        R = np.array(cam[f"Cam{cam_id:02d}"]["R"]).reshape(3, 3)
        T = np.array(cam[f"Cam{cam_id:02d}"]["T"])

        w2c[:3, :3] = torch.FloatTensor(R)
        w2c[:3, 3] = torch.FloatTensor(T)

        if c_one_2_w is None:
            c_one_2_w = torch.eye(4, 4)
            c_one_2_w[:3, :3] = w2c[:3, :3].T
            c_one_2_w[:3, 3] = -w2c[:3, 3] @ w2c[:3, :3]

        if normalize:
            w2c = w2c @ c_one_2_w

        extrs.append(w2c)

    extrs = torch.stack(extrs)
    extrs[:, :3, 3] = extrs[:, :3, 3]

    if len(cam_ids) == 1 and extrs.dim() == 2:
        extrs = extrs.unsqueeze(0)

    return extrs, c_one_2_w
