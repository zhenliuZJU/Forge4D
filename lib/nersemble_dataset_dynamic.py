from torch.utils.data import Dataset

import random
import numpy as np
import os
from PIL import Image
import cv2
import torch
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from pathlib import Path
import logging
import json
from tqdm import tqdm
from forge4d.utils.load_fn import load_and_preprocess_images, load_and_preprocess_images_intrs, load_and_preprocess_masks, load_and_preprocess_depth, load_extrs, load_intrs


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
    for view in depth.shape[0]:
        rot = extrinsic[view, :3, :3]
        trans = extrinsic[view, :3, 3:]
        S, S = depth.shape

        y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device),
                            torch.linspace(0.5, S-0.5, S, device=depth.device))
        pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # H W 3

        pts_2d[..., 2] = depth
        pts_2d[..., 0] -= intrinsic[0, 2]
        pts_2d[..., 1] -= intrinsic[1, 2]
        pts_2d_xy = pts_2d[..., :2] * pts_2d[..., 2:]
        pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

        pts_2d[..., 0] /= intrinsic[0, 0]
        pts_2d[..., 1] /= intrinsic[1, 1]
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


def read_img(name):
    img = np.array(Image.open(name))
    return img


def read_depth(name):
    return cv2.imread(name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1e4


class NersembleDataset(Dataset):
    def __init__(self, opt, phase='train', dtype=torch.float32, test_ids=None, sampling_rate=5, timesteps=2):
        self.opt = opt
        self.val_novel_id = opt.val_novel_id
        self.use_processed_data = opt.use_processed_data
        self.dtype = dtype
        self.phase = phase
        self.dataset_path = opt.data_root
        self.input_cams = [0, 4, 8, 12]

        self.sampling_rate = sampling_rate
        self.timesteps = timesteps

        id_path = os.path.join(self.dataset_path, 'parms')

        # temporal compromise for the dataset
        train_ids = [f"{i}" for i in range(305)]
        test_ids = ["305", "306"]
        # test_ids = ["372", "373"]

        if test_ids is None:
            test_ids = ["372", "160", "304", "258", "298", "309", "305", "314", "322", "326"]

        self.sample_list = []
        if self.phase == 'train':
            human_ids = os.listdir(id_path)
            for human_id in human_ids:
                # temporal compromise for the dataset
                if not any(ids in human_id for ids in train_ids):
                    continue
                    
                if any(ids in human_id for ids in test_ids):
                    continue
                parm_path = os.path.join(self.dataset_path, 'parms', human_id, 'parms')
                files = [f for f in os.listdir(parm_path) if f.endswith(".npy")]
                max_timestep = max([int(f.split("_")[0]) for f in files], default=-1)
                if max_timestep != -1:
                    self.sample_list += [(human_id, t) for t in range(0, max_timestep-sampling_rate*(timesteps-1), sampling_rate)]
        else:
            human_ids = os.listdir(id_path)
            for human_id in human_ids:
                if not any(ids in human_id for ids in test_ids):
                    continue
                parm_path = os.path.join(self.dataset_path, 'parms', human_id, 'parms')
                files = [f for f in os.listdir(parm_path) if f.endswith(".npy")]
                max_timestep = max([int(f.split("_")[0]) for f in files], default=-1)
                if max_timestep != -1:
                    self.sample_list += [(human_id, t) for t in range(0, max_timestep-sampling_rate*(timesteps-1), sampling_rate)]

    def get_origin_data(self, sample_name, require_mask=True):
        human_id, frame_id = sample_name
        intrs, extrs, imgs, masks, depths, pointmap = [], [], [], [], [], []
        for step in range(self.timesteps):
            timestep = frame_id + step * self.sampling_rate
            
            img_names = [os.path.join(self.dataset_path, 'export', human_id, 'images', f'{timestep:05d}_{self.input_cams[i]:02d}.png') for i in range(self.opt.train_view_num)] 
            mask_names = [os.path.join(self.dataset_path, 'export', human_id, 'fg_masks', f'{timestep:05d}_{self.input_cams[i]:02d}.png') for i in range(self.opt.train_view_num)] 
            # depth_names = [os.path.join(self.dataset_path, 'depth', human_id, f'{timestep:05d}_{self.input_cams[i]:02d}.png') for i in range(self.opt.train_view_num)] 
            depth_names = [os.path.join(self.dataset_path, 'depth', 'depth', human_id, f'{timestep:05d}_{self.input_cams[i]:02d}.png') for i in range(self.opt.train_view_num)] 
            extr_names = [os.path.join(self.dataset_path, 'parms', human_id, 'parms', f'{timestep:05d}_{self.input_cams[i]:02d}_extr.npy') for i in range(self.opt.train_view_num)] 
            intr_names = [os.path.join(self.dataset_path, 'parms', human_id, 'parms', f'{timestep:05d}_{self.input_cams[i]:02d}_intr.npy') for i in range(self.opt.train_view_num)]
            img, intr = load_and_preprocess_images_intrs(img_names, intr_names)
            mask = load_and_preprocess_masks(mask_names)
            depth = load_and_preprocess_depth(depth_names)
            extr, c_one_2_w, scale = load_extrs(extr_names) 
            self.c_one_2_w = c_one_2_w
            self.scale = scale

            intrs.append(intr)
            extrs.append(extr)
            imgs.append(img)
            masks.append(mask)
            depths.append(depth)
            pointmap.append(depth2pts(depth, extr, intr))

        camera = {
            'intr': torch.stack(intrs, dim=0).to(self.dtype),
            'extr': torch.stack(extrs, dim=0).to(self.dtype),
        }

        data = {
            'imgs': torch.stack(imgs, dim=0).to(self.dtype), # T, V, C, H, W
            'masks': torch.stack(masks, dim=0).to(self.dtype),
            'depth': torch.stack(depths, dim=0).to(self.dtype),
            'pointmap': torch.stack(pointmap, dim=0).to(self.dtype),
            'camera': camera,
        }

        return data

    def get_item(self, index, novel_id=None):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]
        if len(self.opt.train_novel_id) == self.opt.train_view_num:
            self.input_cams = self.opt.train_novel_id
        else:
            self.input_cams = random.sample(self.opt.train_novel_id, self.opt.train_view_num)
        dict_tensor = self.get_origin_data(sample_name, require_mask=True)

        if not novel_id:
            novel_id = self.val_cams
        
        novel_id = np.random.choice(novel_id)
        dict_tensor.update({
            'novel_view': self.get_novel_view_tensor(sample_name, novel_id)
        })

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
            return self.get_item(index, novel_id=self.opt.train_novel_id)
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
