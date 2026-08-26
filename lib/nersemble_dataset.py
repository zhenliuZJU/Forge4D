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


class NersembleDataset(Dataset):
    def __init__(self, opt, phase='train', test_ids=None, sampling_rate=5):
        self.opt = opt
        self.val_novel_id = opt.val_novel_id
        self.use_processed_data = opt.use_processed_data
        self.phase = phase
        self.dataset_path = opt.data_root
        self.input_cams = [0, 4, 8, 12]

        id_path = os.path.join(self.dataset_path, 'parms')
        # temporal compromise for the dataset
        train_ids = [f"{i:03d}" for i in range(305)]
        test_ids = ["304", "305"]
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
                # temporal compromise for the dataset
                max_timestep=min(300,max_timestep)
                if max_timestep != -1:
                    self.sample_list += [(human_id, t) for t in range(0, max_timestep, sampling_rate)]
        else:
            human_ids = os.listdir(id_path)
            for human_id in human_ids:
                if not any(ids in human_id for ids in test_ids):
                    continue
                parm_path = os.path.join(self.dataset_path, 'parms', human_id, 'parms')
                files = [f for f in os.listdir(parm_path) if f.endswith(".npy")]
                max_timestep = max([int(f.split("_")[0]) for f in files], default=-1)
                # temporal compromise for the dataset
                max_timestep=min(300,max_timestep)
                if max_timestep != -1:
                    self.sample_list += [(human_id, t) for t in range(0, max_timestep, sampling_rate)]

    def load_single_view(self, sample_name, source_id):
        human_id, timestep = sample_name

        img_names = [os.path.join(self.dataset_path, 'export', human_id, 'images', f'{timestep:05d}_{source_id:02d}.png')]
        mask_names = [os.path.join(self.dataset_path, 'export', human_id, 'fg_masks', f'{timestep:05d}_{source_id:02d}.png') ]
        intr_name = [os.path.join(self.dataset_path, 'parms', human_id, 'parms', f'{timestep:05d}_{source_id:02d}_intr.npy')]
        img, intr = load_and_preprocess_images_intrs(img_names, intr_name)
        mask = load_and_preprocess_masks(mask_names).squeeze(dim=0)

        extr_name = os.path.join(self.dataset_path, 'parms', human_id, 'parms', f'{timestep:05d}_{source_id:02d}_extr.npy') 

        extr = np.load(extr_name, allow_pickle=True)
        extr = torch.from_numpy(extr).float()
        w2c = torch.eye(4).float()
        w2c[:3, :3] = extr[:3, :3].T
        w2c[:3, 3:] = -w2c[:3, :3] @ extr[:3, 3:]

        return img.squeeze(dim=0), mask, intr.squeeze(dim=0), w2c

    def get_novel_view_tensor(self, sample_name, view_id):
        img, mask, intr, w2c = self.load_single_view(sample_name, view_id)
        width, height = img.shape[1], img.shape[2]
        w2c = self.c_one_2_w@w2c
        w2c[:3, 3] = w2c[:3, 3] / self.scale

        R = np.array(w2c[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(w2c[:3, 3], np.float32)

        FovX = focal2fov(intr[0, 0], width)
        FovY = focal2fov(intr[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=self.opt.znear, zfar=self.opt.zfar, K=intr, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.opt.trans), self.opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        novel_view_data = {
            'view_id': torch.IntTensor([view_id]),
            'img': img,
            'mask': mask,
            'extr': torch.FloatTensor(w2c),
            'intr': torch.FloatTensor(intr),
            'FovX': FovX,
            'FovY': FovY,
            'width': width,
            'height': height,
            'world_view_transform': world_view_transform,
            'full_proj_transform': full_proj_transform,
            'camera_center': camera_center
        }

        return novel_view_data

    def get_origin_data(self, sample_name, require_mask=True):
        human_id, timestep = sample_name
        img_names = [os.path.join(self.dataset_path, 'export', human_id, 'images', f'{timestep:05d}_{self.input_cams[i]:02d}.png') for i in range(self.opt.train_view_num)] 
        mask_names = [os.path.join(self.dataset_path, 'export', human_id, 'fg_masks', f'{timestep:05d}_{self.input_cams[i]:02d}.png') for i in range(self.opt.train_view_num)] 
        extr_names = [os.path.join(self.dataset_path, 'parms', human_id, 'parms', f'{timestep:05d}_{self.input_cams[i]:02d}_extr.npy') for i in range(self.opt.train_view_num)] 
        intr_names = [os.path.join(self.dataset_path, 'parms', human_id, 'parms', f'{timestep:05d}_{self.input_cams[i]:02d}_intr.npy') for i in range(self.opt.train_view_num)]
        # depth_names = [os.path.join(self.dataset_path, 'depth', human_id, f'{timestep:05d}_{self.input_cams[i]:02d}.png') for i in range(self.opt.train_view_num)] 
        depth_names = [os.path.join(self.dataset_path, 'depth', human_id, f'{timestep:05d}_{self.input_cams[i]:02d}.png') for i in range(self.opt.train_view_num)] 
        imgs, intrs = load_and_preprocess_images_intrs(img_names, intr_names)
        masks = load_and_preprocess_masks(mask_names)
        depth = load_and_preprocess_depth(depth_names)
        extrs, c_one_2_w, _ = load_extrs(extr_names) 

        # compromise, we multiply depth with mask when preprocess the dataset
        for i in range(self.opt.train_view_num):
            depth[i] = depth[i]/(masks[i]+1e-8)
        pointmap = depth2pts(depth, extrs, intrs)

        mask_pm = (masks > 0.99)
        scale = torch.mean(torch.norm(pointmap.permute(1,0,2,3)[:, mask_pm.squeeze()], p=2, dim=0))
        self.scale = scale

        # self.c_one_2_w = torch.inverse(extrs[0])
        self.c_one_2_w = c_one_2_w

        # scaling the data
        extrs[:, :3, 3] = extrs[:, :3, 3] / self.scale
        depth = depth / self.scale
        pointmap = pointmap / self.scale

        camera = {
            'intr': intrs,
            'extr': extrs,
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


