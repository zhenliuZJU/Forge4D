from __future__ import print_function, division

import logging

import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import argparse

from lib.nersemble_dataset import NersembleDataset
from config.transformer_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.loss import l1_loss, l2_loss, ssim, psnr
from lib.utils import compute_norm, compute_scale, save_3dgs_ply
from lib.gaussian_model import GaussianModel

from forge4d.models.vggt import VGGT

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import torchvision
from lib.GaussianRender import render_gaussians
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file

        self.model = VGGT()
        self.train_set = NersembleDataset(self.cfg.dataset, phase='train')
        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=self.cfg.batch_size*2, pin_memory=True)
        self.train_iterator = iter(self.train_loader)
        self.val_set = NersembleDataset(self.cfg.dataset, phase='val', test_ids=['372_FREE_v16_DS4_whiteBg_staticOffset_maskBelowLine'], sampling_rate=1)
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)
        self.val_novel_id = self.cfg.dataset.val_novel_id
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, self.cfg.num_steps + 5000,
                                                       pct_start=0.0, cycle_momentum=False, anneal_strategy='linear')

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0

        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt, load_optimizer=False)
            if self.cfg.DataParallel:
                self.model = torch.nn.DataParallel(self.model)
        elif self.cfg.pretrain_weight:
            logging.info(f"Using pretrained model weight from {self.cfg.pretrain_weight}")
            self.model.load_state_dict(torch.load(self.cfg.pretrain_weight), strict=False)
            if self.cfg.DataParallel:
                self.model = torch.nn.DataParallel(self.model)
        elif self.cfg.stage1_ckpt:
            if self.cfg.DataParallel:
                self.model = torch.nn.DataParallel(self.model)
            logging.info(f"Using model weight from stage 1")
            self.load_ckpt(self.cfg.stage1_ckpt, load_optimizer=False, strict=False)
        
        self.model.train()
        self.scaler = GradScaler(enabled=True)

    @torch.no_grad()
    def generate_video(self):
        os.makedirs('./visualize/vis_angle/front', exist_ok=True)
        self.model.eval()
        for step in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()
            data = self.fetch_data(phase='val')
            bs, vs = data['imgs'].shape[:2]

            predictions  = self.model(data['imgs'], data['masks'], bg_color=self.cfg.dataset.bg_color)
            
            valid = (data['masks']!=0).view(bs, vs, -1)
            i=0
            xyz_i_valid = []
            rgb_i_valid = []
            rot_i_valid = []
            scale_i_valid = []
            opacity_i_valid = []
            for view_id in range(vs):
                valid_i = valid[i, view_id, :]
                xyz_i = predictions['xyz'][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 3)
                rgb_i = predictions["images"][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 3)
                rot_i = predictions['rot_maps'][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 4)
                scale_i = predictions['scale_maps'][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 3)
                opacity_i = predictions['opacity_maps'][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 1)

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
            opacity_mask = (opacity_i.squeeze() > 0.1)
            gaussians = GaussianModel(xyz=pts_xyz_i[opacity_mask].view(-1, 3), opacity=opacity_i[opacity_mask].view(-1, 1), rotation=rot_i[opacity_mask].view(-1, 4), scaling=scale_i[opacity_mask].view(-1, 3), shs=pts_rgb_i[opacity_mask].view(-1, 3))
            gaussians.save_ply('./visualize/vis_pc.ply', rgb2sh=True)
            
            self.novel_extr = self.generate_trajectory(predictions["extr"][:,0], predictions["extr"][:,3])
            for i, extr in enumerate(self.novel_extr):
                synthesized_images = render_gaussians(predictions, predictions["xyz"], predictions["images"], data['masks'], extr.unsqueeze(dim=1), predictions["intr"][:, :1], bg_color=self.cfg.dataset.bg_color)

                tmp_novel = synthesized_images[0][0].detach()
                tmp_novel *= 255
                tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                tmp_img_name = f'./visualize/vis_angle/static/{step}_{i:03d}.jpg'#self.total_steps)
                cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))

            print("Generation finished.")
            break
    
    @torch.no_grad()
    def generate_dynamic_video(self):
        os.makedirs('./visualize/vis_angle/dynamic', exist_ok=True)
        self.model.eval()
        for step in range(self.len_val):
            data = self.fetch_data(phase='val')
            bs, vs = data['imgs'].shape[:2]

            predictions  = self.model(data['imgs'], data['masks'], bg_color=self.cfg.dataset.bg_color)
            self.novel_extr = self.generate_trajectory(predictions["extr"][:,0], predictions["extr"][:,-1], num_frames=self.len_val)
            
            valid = (data['masks']!=0).view(bs, vs, -1)
            i=0
            xyz_i_valid = []
            rgb_i_valid = []
            rot_i_valid = []
            scale_i_valid = []
            opacity_i_valid = []
            for view_id in range(vs):
                valid_i = valid[i, view_id, :]
                xyz_i = predictions['xyz'][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 3)
                rgb_i = predictions["images"][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 3)
                rot_i = predictions['rot_maps'][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 4)
                scale_i = predictions['scale_maps'][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 3)
                opacity_i = predictions['opacity_maps'][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 1)

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
            opacity_mask = (opacity_i.squeeze() > 0.1)
            gaussians = GaussianModel(xyz=pts_xyz_i[opacity_mask].view(-1, 3), opacity=opacity_i[opacity_mask].view(-1, 1), rotation=rot_i[opacity_mask].view(-1, 4), scaling=scale_i[opacity_mask].view(-1, 3), shs=pts_rgb_i[opacity_mask].view(-1, 3))
            # gaussians.save_ply('./visualize/vis_pc.ply', rgb2sh=True)
            
            extr = self.novel_extr[step]
            # extr = predictions["extr"][:,0]
            synthesized_images = render_gaussians(predictions, predictions["xyz"], predictions["images"], data['masks'], extr.unsqueeze(dim=1), predictions["intr"][:, :1], bg_color=self.cfg.dataset.bg_color)

            tmp_novel = synthesized_images[0][0].detach()
            tmp_novel *= 255
            tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
            tmp_img_name = f'./visualize/vis_angle/dynamic/{step:04d}.jpg'#self.total_steps)
            cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))

        print("Generation finished.")
        exit()
    
    @torch.no_grad()
    def generate_static_video(self):
        os.makedirs('./visualize/vis_angle/static', exist_ok=True)
        self.model.eval()
        for step in range(self.len_val):
            data = self.fetch_data(phase='val')
            bs, vs = data['imgs'].shape[:2]

            predictions  = self.model(data['imgs'], data['masks'], bg_color=self.cfg.dataset.bg_color)
            self.novel_extr = self.generate_trajectory(predictions["extr"][:,0], predictions["extr"][:,-1], num_frames=150)
            
            valid = (data['masks']!=0).view(bs, vs, -1)
            i=0
            xyz_i_valid = []
            rgb_i_valid = []
            rot_i_valid = []
            scale_i_valid = []
            opacity_i_valid = []
            for view_id in range(vs):
                valid_i = valid[i, view_id, :]
                xyz_i = predictions['xyz'][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 3)
                rgb_i = predictions["images"][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 3)
                rot_i = predictions['rot_maps'][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 4)
                scale_i = predictions['scale_maps'][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 3)
                opacity_i = predictions['opacity_maps'][i, view_id, :, :, :].permute(1, 2, 0).view(-1, 1)

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
            opacity_mask = (opacity_i.squeeze() > 0.1)
            gaussians = GaussianModel(xyz=pts_xyz_i[opacity_mask].view(-1, 3), opacity=opacity_i[opacity_mask].view(-1, 1), rotation=rot_i[opacity_mask].view(-1, 4), scaling=scale_i[opacity_mask].view(-1, 3), shs=pts_rgb_i[opacity_mask].view(-1, 3))
            # gaussians.save_ply('./visualize/vis_pc.ply', rgb2sh=True)
            
            # extr = predictions["extr"][:,0]
            for i, extr in tqdm(enumerate(self.novel_extr)):
                synthesized_images = render_gaussians(predictions, predictions["xyz"], predictions["images"], data['masks'], extr.unsqueeze(dim=1), predictions["intr"][:, :1], bg_color=self.cfg.dataset.bg_color)

                tmp_novel = synthesized_images[0][0].detach()
                tmp_novel *= 255
                tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                tmp_img_name = f'./visualize/vis_angle/static/{step:04d}_{i:03d}.jpg'#self.total_steps)
                cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))

            print("Generation finished.")
            exit()
         
    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
                
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)

        for item in ["imgs", "masks"]:
            data[item] = data[item].cuda()

        for item in ["novel_view", "camera"]:
            for k in data[item].keys():
                data[item][k] = data[item][k].cuda().contiguous()
                
        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        
        new_state_dict = {}
        for k, v in ckpt['network'].items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        self.model.load_state_dict(new_state_dict, strict=strict)
        logging.info(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            logging.info(f"Optimizer loading done")

    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)
        
    def generate_trajectory(self, P1: torch.Tensor, P2: torch.Tensor, num_frames: int=150) -> torch.Tensor:
        """
        生成从 P1 到 P2 的平滑相机轨迹
        Args:
            P1, P2: (4, 4) 或 (B, 4, 4) 的外参矩阵
            num_frames: 插值帧数
        Returns:
            trajectory: (num_frames, 4, 4) 的轨迹张量
        """
        alphas = torch.linspace(0, 1, num_frames, device=P1.device)
        return torch.stack([self.interpolate_poses_torch(P1, P2, a.item()) for a in alphas], dim=0)

    def interpolate_poses_torch(self, P1: torch.Tensor, P2: torch.Tensor, alpha: float) -> torch.Tensor:
        P1.shape == P2.shape, "Input shapes must match!"
        # assert P1.shape[-2:] == (4, 4), "Input must be 4x4 matrices"
        
        # 提取旋转和平移
        R1, t1 = P1[..., :3, :3], P1[..., :3, 3]
        R2, t2 = P2[..., :3, :3], P2[..., :3, 3]
        
        # 旋转矩阵 → 四元数 (使用 scipy 的 Slerp)
        q1 = R.from_matrix(R1.cpu().numpy()).as_quat()  # (..., 4)
        q2 = R.from_matrix(R2.cpu().numpy()).as_quat()

        q1 = torch.from_numpy(q1).to(R1.device)  # (B, 4)
        q2 = torch.from_numpy(q2).to(R1.device)  # (B, 4)
        
        # Slerp 插值
        # print(q1.shape, q2.shape)
        # slerp = Slerp([0, 1], R.from_quat([q1, q2]))
        # R_interp = torch.tensor(
        #     slerp(alpha).as_matrix(),
        #     device=P1.device, dtype=P1.dtype
        # )  # (..., 3, 3)

        R_interp = self.batch_slerp(q1, q2, alpha)  # (..., 3, 3)
        
        # Lerp 插值平移
        t_interp = (1 - alpha) * t1 + alpha * t2  # (..., 3)
        
        # 构建插值后的外参矩阵
        P_interp = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], device=P1.device, dtype=P1.dtype).expand_as(P1).clone()
        P_interp[..., :3, :3] = R_interp
        P_interp[..., :3, 3] = t_interp
        
        return P_interp

    def batch_slerp(self, q1: torch.Tensor, q2: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        批量四元数球面线性插值 (Slerp)
        Args:
            q1: (B, 4), 起始四元数 (需已归一化)
            q2: (B, 4), 终止四元数 (需已归一化)
            alpha: 插值系数 [0, 1]
        Returns:
            (B, 4), 插值后的四元数
        """
        # 确保输入是torch.Tensor且归一化
        q1 = F.normalize(q1, p=2, dim=-1)
        q2 = F.normalize(q2, p=2, dim=-1)
        
        # 计算点积 (夹角余弦)
        dot = (q1 * q2).sum(dim=-1, keepdim=True)  # (B, 1)
        
        # 处理反向四元数 (保证走最短路径)
        mask = dot < 0
        q2 = torch.where(mask, -q2, q2)
        dot = torch.where(mask, -dot, dot)
        
        # 限制dot范围避免数值误差
        dot = torch.clamp(dot, -1.0, 1.0)
        theta = torch.acos(dot)  # (B, 1)
        
        # 插值计算
        sin_theta = torch.sin(theta)
        w1 = torch.sin((1 - alpha) * theta) / (sin_theta + 1e-8)
        w2 = torch.sin(alpha * theta) / (sin_theta + 1e-8)
        q_new = w1 * q1 + w2 * q2  # (B, 4)

        rot_mat_np = R.from_quat(q_new.cpu().numpy()).as_matrix()  # (B, 3, 3)
        rot_mat = torch.from_numpy(rot_mat_np).to(q1.device)

        return rot_mat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default="config/generate.yaml")
    parser.add_argument('--log_path', type=str, default="./runs")
    parser.add_argument('--depth', type=bool, default=True)

    args = parser.parse_args()
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load(args.cfg_path)
    cfg = cfg.get_cfg()

    torch.manual_seed(1314)
    np.random.seed(1314)

    trainer = Trainer(cfg)
    # trainer.generate_video()
    trainer.generate_static_video()
    trainer.generate_dynamic_video()