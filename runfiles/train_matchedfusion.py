from __future__ import print_function, division

import logging

import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import argparse
import random

from lib.nersemble_dataset_dynamic import NersembleDataset
from lib.dna_dataset_dynamic import DNADataset, DNADatasetPart3, DNADatasetPart4, DNADatasetPart5, DNADatasetPart6, DNADataset4K4D
from lib.humman_dataset_dynamic import HuMManDataset
from lib.genebody_dataset_dynamic import GenebodyDataset
from lib.selfcap_dataset_dynamic import SelfCapDataset
from lib.vvc_dataset_dynamic import VVCDataset
from config.transformer_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.loss import l1_loss, l2_loss, ssim, psnr, opacity_regular, scale_regular, PerceptualLoss, flow_loss
from lib.utils import compute_norm, compute_scale, get_lr_func, gradient
from lib.flow import flow_to_color

from forge4d.models.forge4d import Forge4D

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs
from collections import OrderedDict
from torchvision.utils import save_image

import torchvision
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

Dataset = {'Nersemble': NersembleDataset, 'DNA': DNADataset, 'DNAP3': DNADatasetPart3, 'DNAP4': DNADatasetPart4, 'DNAP5': DNADatasetPart5, 'DNAP6': DNADatasetPart6, 'HuMMan': HuMManDataset, 'DNA4K4D': DNADataset4K4D, 'Genebody':GenebodyDataset, 'selfcap': SelfCapDataset, 'vvc': VVCDataset}
ValDataset = {'DNA': DNADataset, 'DNA4K4D': DNADataset4K4D, 'HuMMan': HuMManDataset, 'Genebody':GenebodyDataset, 'selfcap': SelfCapDataset, 'vvc': VVCDataset}

class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file
        
        self.model = Forge4D(point_map=self.cfg.pointmap, motion_dimension=3)
        self.lpips = PerceptualLoss(device='cuda').cuda()

        if self.cfg.stage == 'static':
            for name, module in self.model.aggregator.named_children():
                if name in ["global_blocks", "frame_blocks", "cross_blocks"]: 
                    for param in module.parameters():
                        param.requires_grad = False
            for name, module in self.model.motion_head.named_children():
                for param in module.parameters():
                    param.requires_grad = False
            l = [
                {'params': self.model.gaussian_head.parameters(), 'lr': 0.000001, "name": "gaussian head"},
                # {'params': self.model.motion_head.parameters(), 'lr': self.cfg.lr, "name": "motionhead"},
                {'params': [p for p in self.model.aggregator.parameters() 
                    if p not in set(self.model.aggregator.scene_blocks.parameters())
                    and p not in set(self.model.aggregator.cross_blocks.parameters()) 
                    and p not in set(self.model.aggregator.lambda_t)], 'lr': 0.000001, "name": "aggregator"},
                # {'params': self.model.aggregator.cross_blocks.parameters(), 'lr': self.cfg.lr, "name": "cross blocks"},
                {'params': self.model.aggregator.scene_blocks.parameters(), 'lr': self.cfg.lr, "name": "scene blocks"},
                {'params': self.model.aggregator.lambda_t, 'lr': self.cfg.lr, "name": "state lambda"},
                {'params': [self.model.state_token], 'lr': self.cfg.lr, "name": "state token"},
                {'params': self.model.camera_head.parameters(), 'lr': 0.000001, "name": "camhead"},
                {'params': self.model.point_head.parameters(), 'lr': 0.000001, "name": "ptshead"},
            ]
            self.optimizer = optim.AdamW(l, lr=0.0, weight_decay=self.cfg.wdecay, eps=1e-8)
            self.scheduler = get_lr_func(lr_init=self.cfg.lr, lr_final=1e-7, max_steps=self.cfg.num_steps, warmup_steps=5000)
            self.scheduler_ft = get_lr_func(lr_init=0.000001, lr_final=1e-7, max_steps=self.cfg.num_steps, warmup_steps=5000)
        elif self.cfg.stage == "motion":
            for name, module in self.model.aggregator.named_children():
                # if name not in ["cross_blocks"]:  # 排除 blocks 子模块
                    for param in module.parameters():
                        param.requires_grad = False
            for name, module in self.model.point_head.named_children():
                for param in module.parameters():
                    param.requires_grad = False
            for name, module in self.model.camera_head.named_children():
                for param in module.parameters():
                    param.requires_grad = False
            for name, module in self.model.gaussian_head.named_children():
                for param in module.parameters():
                    param.requires_grad = False
            self.model.state_token.requires_grad = False
            l = [
                # The paper's stage-3 trains the motion head together with the
                # Gaussian fusion head. The original code instead relied on a
                # pre-trained motion head from a separate "together" run; keep
                # the motion head enabled here so this stage is self-contained.
                {'params': self.model.motion_head.parameters(), 'lr': self.cfg.lr, "name": "motionhead"},
                # {'params': self.model.aggregator.cross_blocks.parameters(), 'lr': self.cfg.lr, "name": "cross blocks"},
                {'params': self.model.fusion.parameters(), 'lr': self.cfg.lr, "name": "fusion"},
            ]
            self.optimizer = optim.AdamW(l, lr=0.0, weight_decay=self.cfg.wdecay, eps=1e-8)
            self.scheduler = get_lr_func(lr_init=self.cfg.lr, lr_final=1e-7, max_steps=self.cfg.num_steps, warmup_steps=5000)
            self.scheduler_ft = get_lr_func(lr_init=0.000001, lr_final=1e-7, max_steps=self.cfg.num_steps, warmup_steps=5000)
        elif self.cfg.stage == "together":
            l = [
                {'params': self.model.gaussian_head.parameters(), 'lr': 0.000001, "name": "gaussian head"},
                {'params': self.model.motion_head.parameters(), 'lr': self.cfg.lr, "name": "motionhead"},
                {'params': [p for p in self.model.aggregator.parameters() 
                    if p not in set(self.model.aggregator.scene_blocks.parameters())
                    and p not in set(self.model.aggregator.cross_blocks.parameters()) 
                    and p not in set(self.model.aggregator.lambda_t)], 'lr': 0.000001, "name": "aggregator"},
                {'params': self.model.aggregator.cross_blocks.parameters(), 'lr': self.cfg.lr, "name": "cross blocks"},
                {'params': self.model.aggregator.scene_blocks.parameters(), 'lr': self.cfg.lr, "name": "scene blocks"},
                {'params': self.model.aggregator.lambda_t, 'lr': self.cfg.lr, "name": "state lambda"},
                {'params': [self.model.state_token], 'lr': self.cfg.lr, "name": "state token"},
                {'params': self.model.camera_head.parameters(), 'lr': 0.000001, "name": "camhead"},
                {'params': self.model.point_head.parameters(), 'lr': 0.000001, "name": "ptshead"},
                {'params': self.model.fusion.parameters(), 'lr': self.cfg.lr, "name": "fusion"},
            ]
            self.optimizer = optim.AdamW(l, lr=0.0, weight_decay=self.cfg.wdecay, eps=1e-8)
            self.scheduler = get_lr_func(lr_init=self.cfg.lr, lr_final=1e-7, max_steps=self.cfg.num_steps, warmup_steps=5000)
            self.scheduler_ft = get_lr_func(lr_init=0.000001, lr_final=1e-7, max_steps=self.cfg.num_steps, warmup_steps=5000)
        else:
            raise NotImplementedError("Stage Not Implemented")

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0
        
        self.accelerator = Accelerator(gradient_accumulation_steps=48, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

        #### Data loading
        if len(self.cfg.dataname)>1:
            print("Training on Multiple Datasets:", self.cfg.dataname)
            self.multi_dataset = True
        else:
            self.multi_dataset = False

        self.train_set, self.train_loader, self.val_set, self.val_loader = [], [], [], []
        for i,dataset_name in enumerate(self.cfg.dataname):
            self.train_set.append(Dataset[dataset_name](self.cfg.dataset, phase='train', sampling_rate=self.cfg.sampling_rate, timesteps=2))
            sampler = DistributedSampler(
                    self.train_set[i],
                    num_replicas=self.accelerator.num_processes,  # 总进程数
                    rank=self.accelerator.process_index,          # 当前进程rank
                    shuffle=True
                )
            self.train_loader.append(DataLoader(self.train_set[i], batch_size=self.cfg.batch_size, sampler=sampler,
                                        num_workers=min(8, os.cpu_count()), pin_memory=True, persistent_workers=True))
        for i,dataset_name in enumerate(self.cfg.val_dataname):
            test_ids = ['corgi-release'] if self.cfg.dataset == 'selfcap' else None
            self.val_set.append(ValDataset[dataset_name](self.cfg.dataset, phase='val', test_ids=test_ids, sampling_rate=self.cfg.sampling_rate, timesteps=16))
            self.val_loader.append(DataLoader(self.val_set[i], batch_size=2, shuffle=False, num_workers=4, pin_memory=True))

        self.val_novel_id = self.cfg.dataset.val_novel_id

        ### Prepare model
        self.model.to(self.accelerator.device)
        
        self.model.train()
        # self.scaler = GradScaler(enabled=True)
        
        if self.cfg.pretrain_weight:
            logging.info(f"Using pretrained model weight from {self.cfg.pretrain_weight}")
            self.model.load_state_dict(torch.load(self.cfg.pretrain_weight), strict=False)
        # self.model = self.model.to(DTYPE)
            
        self.model, self.lpips, self.optimizer, self.train_loader, self.val_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.lpips, self.optimizer, self.train_loader, self.val_loader, self.scheduler
        )
        
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        elif self.cfg.stage1_ckpt:
            logging.info(f"Using model weight from stage 1")
            self.load_ckpt(self.cfg.stage1_ckpt, load_optimizer=False, strict=False)
        
        self.train_iterator, self.val_iterator = [], []
        for train_loader in self.train_loader:
            self.train_iterator.append(iter(train_loader))

        for val_loader in self.val_loader:
            self.val_iterator.append(iter(val_loader))

    # @torch.amp.autocast(device_type='cuda', dtype=DTYPE)
    def train(self):
        # if self.accelerator.local_process_index == 0:
        #     try:
        #         self.model.eval()
        #         self.run_eval()
        #     except Exception as e:
        #         print("Error while evaluating the model", e)

        # self.accelerator.wait_for_everyone()
        # self.model.train()
        # exit()
        self.model.eval()
        self.run_eval()
        self.model.train()
        exit()
        for step in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()

            data = self.fetch_data(phase='train')

            metrics = {}
            bg_color = random.choice([[0.,0.,0.,],[0.5,0.5,0.5],[1.,1.,1.]])
             
            mask = (data['masks']>0.66).float()#.expand_as(data['xyz'])
            data['imgs'] = data['imgs'] *mask + (1-mask)*(torch.tensor(bg_color).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-2)).to(data['imgs'])
            novel_time_mask = (data['novel_time_mask']>0.66).float()
            data['novel_time'] = data['novel_time'] *novel_time_mask + (1-novel_time_mask)*(torch.tensor(bg_color).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-2)).to(data['novel_time'])
            predictions  = self.model.module.forward_matching_fusion(data, data['imgs'], data['masks'], bg_color=bg_color)
            
            render_in = predictions["images"]
            gt_in = data['imgs'] #data['novel_view']['img'].cuda()

            c, h, w = render_in.shape[-3:]

            L2 = l2_loss(render_in, gt_in)
            
#             with torch.no_grad():
#                 z_gs = compute_norm(predictions['xyz'], mask=mask)
#                 z_pt = compute_norm(predictions['world_points'], mask=mask)

#             L_pt = l2_loss(predictions['xyz']*mask/z_gs, predictions['world_points']*mask/z_pt)
            # L_intr = l2_loss(data["camera"]['intr'], predictions['intr'])
            Lssim = 1.0 - ssim(render_in.reshape(-1, c, h, w), (gt_in).reshape(-1, c, h, w))
            lpips = self.lpips(render_in.reshape(-1, c, h, w), (gt_in).reshape(-1, c, h, w)) #if step>10000 else torch.tensor(0.)
            psnr_value = psnr(render_in, gt_in).mean().double()

            mask = mask.reshape(-1, 1, h, w)
            if self.cfg.pointmap:
                predictions['depth'], data['depth'] = predictions['depth'].reshape(-1, 1, h, w), data['depth'].reshape(-1, 1, h, w)
                predictions['world_points'], data['pointmap'] = predictions['world_points'].reshape(-1, 3, h, w), data['pointmap'].reshape(-1, 3, h, w)

                l_depth = l2_loss(mask*predictions['depth'], mask*data['depth']) #+ l2_loss(mask*gradient(predictions['depth']), mask*gradient(data['depth']))
                l_pc = l2_loss(mask*predictions['world_points'], mask*data['pointmap']) #+ l2_loss(mask*gradient(predictions['world_points']), mask*gradient(data['pointmap']))
                # l_cam = l2_loss(predictions['extr'], data['camera']['extr']) + l2_loss(predictions['intr'], data['camera']['intr'])/h
                l_cam = l2_loss(predictions['pose_gt'], predictions['pose_encoding'])
            else:
                l_depth, l_pc = torch.tensor(0), torch.tensor(0)

                pose_gt = predictions['pose_gt']
                norm_gt, norm_pred = torch.norm(pose_gt[:,:,1:,:3], dim=-1, keepdim=True), torch.norm(predictions['pose_encoding'][:,:,1:,:3], dim=-1, keepdim=True) # exclude view 1

                l_t = l2_loss(pose_gt[:,:,1:,:3]/(norm_gt+1e-8), predictions['pose_encoding'][:,:,1:,:3]/(norm_pred+1e-8)) + ((norm_gt/(norm_pred+1e-8)-torch.mean(norm_gt/(norm_pred+1e-8), dim=(1, 2), keepdim=True))**2).mean()#
                l_cam = l2_loss(pose_gt[:,:,:,3:], predictions['pose_encoding'][:,:,:,3:])+l_t

            # lambda_lpips = 0.2 if step > 20000 else 0.0
            lambda_lpips = self.cfg.loss.lpips
            l_input = 0.8 * L2 + 0.2 * Lssim + lambda_lpips* lpips #if step > 20000 else 0

            l_flow_bw, _, flow_mask_bw = flow_loss(predictions["flow_bw"], data["flow_bw"][:,1], data["flow_fw"][:,0], (data['masks']>0.66).float()[:,1])
            l_flow_fw, _, flow_mask_fw = flow_loss(predictions["flow_fw"], data["flow_fw"][:,0], data["flow_bw"][:,1], (data['masks']>0.66).float()[:,1])
            l_flow = l_flow_bw + l_flow_fw
             
            retarget_novel_t_gt = data['novel_time']
            retarget_novel_t_images_bw = predictions['novel_time_bw']
            retarget_novel_t_images_fw = predictions['novel_time_fw']
            psnr_retarget_novel_time_bw = psnr(retarget_novel_t_gt, retarget_novel_t_images_bw).mean().double()
            psnr_retarget_novel_time_fw = psnr(retarget_novel_t_gt, retarget_novel_t_images_fw).mean().double()
            l_novel_t_bw = 0.8*l2_loss(retarget_novel_t_images_bw, retarget_novel_t_gt) + 0.2*(1.0 - ssim(retarget_novel_t_images_bw.reshape(-1, c, h, w), (retarget_novel_t_gt).reshape(-1, c, h, w))) + lambda_lpips* self.lpips(retarget_novel_t_images_bw.reshape(-1, c, h, w), (retarget_novel_t_gt).reshape(-1, c, h, w)) 
            l_novel_t_fw = 0.8*l2_loss(retarget_novel_t_images_fw, retarget_novel_t_gt) + 0.2*(1.0 - ssim(retarget_novel_t_images_fw.reshape(-1, c, h, w), (retarget_novel_t_gt).reshape(-1, c, h, w))) + lambda_lpips* self.lpips(retarget_novel_t_images_fw.reshape(-1, c, h, w), (retarget_novel_t_gt).reshape(-1, c, h, w)) 
            
            retarget_novel_t_images = predictions['novel_time']
            psnr_retarget_novel_time = psnr(retarget_novel_t_gt, retarget_novel_t_images).mean().double()
            l_novel_t_concate = 0.8*l2_loss(retarget_novel_t_images, retarget_novel_t_gt) + 0.2*(1.0 - ssim(retarget_novel_t_images.reshape(-1, c, h, w), (retarget_novel_t_gt).reshape(-1, c, h, w))) + lambda_lpips* self.lpips(retarget_novel_t_images.reshape(-1, c, h, w), (retarget_novel_t_gt).reshape(-1, c, h, w)) 

            retarget_gt = data['imgs'][:, :-1]
            retarget_images_bw = predictions['retarget_images_bw']
            psnr_retarget_bw = psnr(retarget_gt, retarget_images_bw).mean().double()
            l_retarget_bw = 0.8*l2_loss(retarget_images_bw, retarget_gt) + 0.2*(1.0 - ssim(retarget_images_bw.reshape(-1, c, h, w), (retarget_gt).reshape(-1, c, h, w))) + lambda_lpips* self.lpips(retarget_images_bw.reshape(-1, c, h, w), (retarget_gt).reshape(-1, c, h, w)) 
            retarget_gt = data['imgs'][:, 1:]
            retarget_images_fw = predictions['retarget_images_fw']
            psnr_retarget_fw = psnr(retarget_gt, retarget_images_fw).mean().double()
            l_retarget_fw = 0.8*l2_loss(retarget_images_fw, retarget_gt) + 0.2*(1.0 - ssim(retarget_images_fw.reshape(-1, c, h, w), (retarget_gt).reshape(-1, c, h, w))) + lambda_lpips* self.lpips(retarget_images_fw.reshape(-1, c, h, w), (retarget_gt).reshape(-1, c, h, w)) 
            
            l_novel_t = l_novel_t_bw + l_novel_t_fw + l_retarget_bw + l_retarget_fw + l_novel_t_concate

            mask_novel = (data['novel_view']['mask']>0.66).float()
            data['novel_view']['img'] = data['novel_view']['img'] *mask_novel + (1-mask_novel)*(torch.tensor(bg_color).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-2)).to(data['imgs'])

            render_novel = predictions["novel_img"].squeeze()
            gt_novel = data['novel_view']['img']
            psnr_novel = psnr(render_novel, gt_novel).mean().double()
            l_novel = 0.8*l2_loss(render_novel, gt_novel) + 0.2*(1.0 - ssim(render_novel.reshape(-1, c, h, w), (gt_novel).reshape(-1, c, h, w))) + lambda_lpips* self.lpips(render_novel.reshape(-1, c, h, w), (gt_novel).reshape(-1, c, h, w)) 
            # l_novel = l2_loss(render_novel, gt_novel) + lambda_lpips* self.lpips(render_novel.reshape(-1, c, h, w), (gt_novel).reshape(-1, c, h, w)) 
            
            lambda_novel = 1 #if step > 20000 else 0.0
            lambda_input = 1
            lambda_cam = 1
            lambda_flow = self.cfg.loss.flow*1/h
            lambda_novel_t = 1
            loss = lambda_input*l_input + l_depth + l_pc + lambda_cam*l_cam + lambda_novel*l_novel + lambda_novel_t*l_novel_t + lambda_flow*l_flow
            # loss =  L2  + 0.5 * lpips#+ L_intr

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0 and self.accelerator.process_index == 0:
                mask = (data['masks']>0.99).float()
                image = torchvision.utils.make_grid(render_in.reshape(-1, render_in.shape[-3], render_in.shape[-2], render_in.shape[-1]))
                self.logger.writer.add_image("Rendered Train", image, global_step=self.total_steps)
                
                image = torchvision.utils.make_grid(gt_in.reshape(-1, gt_in.shape[-3], gt_in.shape[-2], gt_in.shape[-1])) 
                self.logger.writer.add_image("GT Train", image, global_step=self.total_steps)

                image = torchvision.utils.make_grid(render_novel.reshape(-1, render_novel.shape[-3], render_novel.shape[-2], render_novel.shape[-1]))
                self.logger.writer.add_image("Rendered Novel Train", image, global_step=self.total_steps)

                image = torchvision.utils.make_grid(retarget_novel_t_images_bw.reshape(-1, retarget_novel_t_images_bw.shape[-3], retarget_novel_t_images_bw.shape[-2], retarget_novel_t_images_bw.shape[-1]))
                self.logger.writer.add_image("Rendered Novel Time BW Train", image, global_step=self.total_steps)

                image = torchvision.utils.make_grid(retarget_novel_t_images_fw.reshape(-1, retarget_novel_t_images_fw.shape[-3], retarget_novel_t_images_fw.shape[-2], retarget_novel_t_images_fw.shape[-1]))
                self.logger.writer.add_image("Rendered Novel Time FW Train", image, global_step=self.total_steps)

                image = torchvision.utils.make_grid(retarget_novel_t_images.reshape(-1, retarget_novel_t_images.shape[-3], retarget_novel_t_images.shape[-2], retarget_novel_t_images.shape[-1]))
                self.logger.writer.add_image("Rendered Novel Time Train", image, global_step=self.total_steps)

                image = torchvision.utils.make_grid(retarget_novel_t_gt.reshape(-1, retarget_novel_t_gt.shape[-3], retarget_novel_t_gt.shape[-2], retarget_novel_t_gt.shape[-1]))
                self.logger.writer.add_image("GT Novel Time Train", image, global_step=self.total_steps)

                image = torchvision.utils.make_grid(retarget_images_bw.reshape(-1, retarget_images_bw.shape[-3], retarget_images_bw.shape[-2], retarget_images_bw.shape[-1]))
                self.logger.writer.add_image("Rendered Retargeted BW Train", image, global_step=self.total_steps)

                image = torchvision.utils.make_grid(retarget_images_fw.reshape(-1, retarget_images_fw.shape[-3], retarget_images_fw.shape[-2], retarget_images_fw.shape[-1]))
                self.logger.writer.add_image("Rendered Retargeted FW Train", image, global_step=self.total_steps)

                # image = torchvision.utils.make_grid(torch.abs(retarget_images-retarget_gt).reshape(-1, retarget_images.shape[-3], retarget_images.shape[-2], retarget_images.shape[-1]))
                # self.logger.writer.add_image("Retargeted Residual Train", image, global_step=self.total_steps)

                # image = torchvision.utils.make_grid(torch.abs(retarget_images- render_in[:, 1:]).reshape(-1, retarget_images.shape[-3], retarget_images.shape[-2], retarget_images.shape[-1]))
                # self.logger.writer.add_image("Retargeted Displace Train", image, global_step=self.total_steps)
                            
                image = torchvision.utils.make_grid(gt_novel.reshape(-1, gt_novel.shape[-3], gt_novel.shape[-2], gt_novel.shape[-1])) 
                self.logger.writer.add_image("GT Novel Train", image, global_step=self.total_steps)

                image = torchvision.utils.make_grid(flow_to_color(data["flow_bw"][:,1].reshape(-1, data["flow_bw"][:,1].shape[-3], data["flow_bw"][:,1].shape[-2], data["flow_bw"][:,1].shape[-1]), flow_mask_bw)) 
                self.logger.writer.add_image("GT Flow Bw", image, global_step=self.total_steps)

                image = torchvision.utils.make_grid(flow_to_color(predictions["flow_bw"].reshape(-1, predictions["flow_bw"].shape[-3], predictions["flow_bw"].shape[-2], predictions["flow_bw"].shape[-1]), flow_mask_bw)) 
                self.logger.writer.add_image("Pred Flow Bw", image, global_step=self.total_steps)

                image = torchvision.utils.make_grid(flow_to_color(data["flow_fw"][:,0].reshape(-1, data["flow_fw"][:,1].shape[-3], data["flow_fw"][:,1].shape[-2], data["flow_fw"][:,1].shape[-1]), flow_mask_fw)) 
                self.logger.writer.add_image("GT Flow FW", image, global_step=self.total_steps)

                image = torchvision.utils.make_grid(flow_to_color(predictions["flow_fw"].reshape(-1, predictions["flow_fw"].shape[-3], predictions["flow_fw"].shape[-2], predictions["flow_fw"].shape[-1]), flow_mask_fw)) 
                self.logger.writer.add_image("Pred Flow FW", image, global_step=self.total_steps)

                if self.cfg.pointmap:
                    image = torchvision.utils.make_grid(0.5*data['depth']) 
                    self.logger.writer.add_image("GT Depth Train", image, global_step=self.total_steps)

                    image = torchvision.utils.make_grid(((torch.abs(mask*predictions['world_points'])-torch.tensor([0.0, 0.0, 0.8]).cuda().unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)))*torch.tensor([3,3,3]).cuda().unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)) 
                    self.logger.writer.add_image("Rendered Pointmap Train", image, global_step=self.total_steps)

                    image = torchvision.utils.make_grid((torch.abs(mask*data['pointmap'])-torch.tensor([0., 0., 0.8]).cuda().unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1))*torch.tensor([3,3,3]).cuda().unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)) 
                    self.logger.writer.add_image("GT Pointmap Train", image, global_step=self.total_steps)

                    image = torchvision.utils.make_grid((0.5*mask*predictions['depth'])) 
                    self.logger.writer.add_image("Predicted Depth Train", image, global_step=self.total_steps)

                self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)

                if self.total_steps % (50*self.cfg.record.loss_freq) == 0:
                    self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)

            # self.scaler.scale(loss).backward()
            # self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.accelerator.backward(loss)
            self.optimizer.step()
            
            # self.scheduler.step()
            self.update_learning_rate(step)

            L2 = self.accelerator.gather(L2).mean()
            Lssim = self.accelerator.gather(Lssim).mean()
            psnr_value = self.accelerator.gather(psnr_value).mean()
            lpips = self.accelerator.gather(lpips).mean()
            # l_depth = self.accelerator.gather(l_depth)
            # l_pc = self.accelerator.gather(l_pc)
            l_flow = self.accelerator.gather(l_flow).mean()
            l_cam = self.accelerator.gather(l_cam).mean()
            l_novel = self.accelerator.gather(l_novel).mean()
            psnr_novel = self.accelerator.gather(psnr_novel).mean()
            psnr_retarget_fw = self.accelerator.gather(psnr_retarget_fw).mean()
            psnr_retarget_bw = self.accelerator.gather(psnr_retarget_bw).mean()
            psnr_retarget_novel_time_fw = self.accelerator.gather(psnr_retarget_novel_time_fw).mean()
            psnr_retarget_novel_time_bw = self.accelerator.gather(psnr_retarget_novel_time_bw).mean()
            psnr_retarget_novel_time = self.accelerator.gather(psnr_retarget_novel_time).mean()
            
            metrics.update({
                'loss': loss.item(),
                'ssim': Lssim.item(),
                'psnr': psnr_value.item(),
                'lpips': lpips.item(),
                'l_depth': l_depth.item(),
                'l_pt': l_pc.item(),
                'l_flow': l_flow.item(),
                'l_cam': l_cam.item(),
                'l_novel': l_novel.item(),
                'psnr_novel': psnr_novel.item(),
                'psnr_retarget_fw': psnr_retarget_fw.item(),
                'psnr_retarget_novel_time_fw': psnr_retarget_novel_time_fw.item(),
                'psnr_retarget_bw': psnr_retarget_bw.item(),
                'psnr_retarget_novel_time_bw': psnr_retarget_novel_time_bw.item(),
                'psnr_retarget_novel_time': psnr_retarget_novel_time.item(),
            })
            self.logger.push(metrics)

            if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0:
                self.model.eval()
                try:
                    self.run_eval()
                except Exception as e:
                    print("Error while evaluating the model", e)
                self.model.train()

            self.total_steps += 1

        self.model.eval()

        try:
            self.run_eval()
        except Exception as e:
            print("Error while evaluating the model", e)
        
        self.model.train()
        
        print("FINISHED TRAINING")
        self.logger.close()
        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))
         
    # @torch.amp.autocast(device_type='cuda', dtype=DTYPE)
    def run_eval(self):
        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        for index in range(len(self.val_set)):
                psnr_list, ssim_list = [], []
                psnr_novel_list, ssim_novel_list, lpips_novel_list = [], [], []
                val_psnr_retarget_novel_time_fw_list, val_psnr_retarget_fw_list = [], []
                val_psnr_retarget_novel_time_bw_list, val_psnr_retarget_bw_list = [], []
                val_psnr_retarget_novel_time_list, ssim_retarget_novel_time_list, lpips_retarget_novel_time_list = [], [], []
                val_psnr_retarget_novel_time_view_list, ssim_retarget_novel_time_view_list, lpips_retarget_novel_time_view_list = [], [], []
                flow_loss_list = []
            # for val_novel_id in self.val_novel_id:
            #     self.val_set[index].val_novel_id =  [val_novel_id]
                self.val_loader[index] = DataLoader(self.val_set[index], batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
                self.len_val = int(len(self.val_loader[index]))  # real length of val set
                self.val_iterator = iter(self.val_loader[index])
                show_idx = np.random.choice(list(range(self.len_val)), 1)
                for idx in tqdm(range(self.len_val)):
                    try:
                        data = self.fetch_data(phase='val', index=index)
                    except Exception as e:
                        print(e)
                        continue
                    with torch.no_grad():
                        mask = (data['masks']>0.66).float()#.expand_as(data['xyz'])
                        data['imgs'] = data['imgs'] *mask + (1-mask)*(torch.tensor(self.cfg.dataset.bg_color).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-2)).to(data['imgs'])
                        mask_novel = (data['novel_view']['mask']>0.66).float()
                        data['novel_view']['img'] = data['novel_view']['img'] *mask_novel + (1-mask_novel)*(torch.tensor(self.cfg.dataset.bg_color).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-2)).to(data['imgs'])                        
                        gt_novel = data['novel_view']['img']
                        novel_time_mask = (data['novel_time_mask']>0.66).float()
                        data['novel_time'] = data['novel_time'] *novel_time_mask + (1-novel_time_mask)*(torch.tensor(self.cfg.dataset.bg_color).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-2)).to(data['novel_time'])
                        novel_view_time_mask = data['novel_view']["novel_time_mask"]
                        data['novel_view']['novel_time'] = data['novel_view']['novel_time'] *novel_view_time_mask + (1-novel_view_time_mask)*(torch.tensor(self.cfg.dataset.bg_color).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-2)).to(data['novel_view']['novel_time'])

                        tmp_novel = gt_novel.reshape(-1, gt_novel.shape[-3], gt_novel.shape[-2], gt_novel.shape[-1]).detach()
                        Path('%s/gt_novel_view/' % (cfg.record.show_path)).mkdir(exist_ok=True, parents=True)
                        tmp_img_name = '%s/gt_novel_view/%s_%s_%d.png' % (cfg.record.show_path, data["scene"][0], index, idx)
                        save_image(tmp_novel, tmp_img_name, normalize=True)

                        tmp_novel = data['novel_view']['novel_time'].reshape(-1, data['novel_view']['novel_time'].shape[-3], data['novel_view']['novel_time'].shape[-2], data['novel_view']['novel_time'].shape[-1]).detach()
                        Path('%s/gt_novel_view_time/' % (cfg.record.show_path)).mkdir(exist_ok=True, parents=True)
                        tmp_img_name = '%s/gt_novel_view_time/%s_%s_%d.png' % (cfg.record.show_path, data["scene"][0], index, idx)
                        save_image(tmp_novel, tmp_img_name, normalize=True)
                        # continue

                        predictions  = self.model.module.forward_matching_fusion(data, data['imgs'], data['masks'], bg_color=self.cfg.dataset.bg_color, is_val=True)
                        # predictions  = self.model.module.forward_matching_fusion(data, data['imgs'], data['masks'], bg_color=self.cfg.dataset.bg_color, is_val=False)
                        
                        render_in = predictions["images"]
                        gt_in = data['imgs'] #data['novel_view']['img'].cuda()
                        c, h, w = render_in.shape[-3:]

                        render_novel = predictions["novel_img"].squeeze()
                        retarget_novel_t_gt = data['novel_time']
                        retarget_novel_t_images_bw = predictions['novel_time_bw']
                        retarget_novel_t_images_fw = predictions['novel_time_fw']
                        psnr_retarget_novel_time_bw = psnr(retarget_novel_t_gt, retarget_novel_t_images_bw).mean().double()
                        psnr_retarget_novel_time_fw = psnr(retarget_novel_t_gt, retarget_novel_t_images_fw).mean().double()
                        l_novel_t_bw = 0.8*l2_loss(retarget_novel_t_images_bw, retarget_novel_t_gt) + 0.2*(1.0 - ssim(retarget_novel_t_images_bw.reshape(-1, c, h, w), (retarget_novel_t_gt).reshape(-1, c, h, w))) + 0.2* self.lpips(retarget_novel_t_images_bw.reshape(-1, c, h, w), (retarget_novel_t_gt).reshape(-1, c, h, w)) 
                        l_novel_t_fw = 0.8*l2_loss(retarget_novel_t_images_fw, retarget_novel_t_gt) + 0.2*(1.0 - ssim(retarget_novel_t_images_fw.reshape(-1, c, h, w), (retarget_novel_t_gt).reshape(-1, c, h, w))) + 0.2* self.lpips(retarget_novel_t_images_fw.reshape(-1, c, h, w), (retarget_novel_t_gt).reshape(-1, c, h, w))
                        val_psnr_retarget_novel_time_fw_list.append(psnr_retarget_novel_time_fw.item())
                        val_psnr_retarget_novel_time_bw_list.append(psnr_retarget_novel_time_bw.item()) 
                        
                        retarget_novel_t_images = predictions['novel_time']
                        psnr_retarget_novel_time = psnr(retarget_novel_t_gt, retarget_novel_t_images).mean().double()
                        Lssim_retarget_novel_time = 1.0 - ssim(retarget_novel_t_images.reshape(-1, render_in.shape[-3], render_in.shape[-2], render_in.shape[-1]), (retarget_novel_t_gt).reshape(-1, gt_in.shape[-3], gt_in.shape[-2], gt_in.shape[-1]))
                        lpips_retarget_novel_time = self.lpips(retarget_novel_t_images.reshape(-1, render_in.shape[-3], render_in.shape[-2], render_in.shape[-1]), retarget_novel_t_gt.reshape(-1, gt_in.shape[-3], gt_in.shape[-2], gt_in.shape[-1])).mean().double()
                        val_psnr_retarget_novel_time_list.append(psnr_retarget_novel_time.item())
                        ssim_retarget_novel_time_list.append(Lssim_retarget_novel_time.mean().item())
                        lpips_retarget_novel_time_list.append(lpips_retarget_novel_time.item())

                        retarget_novel_t_novel_view_gt = data['novel_view']['novel_time']
                        retarget_novel_t_novel_view = predictions['novel_time_novel_view']
                        psnr_retarget_novel_time_view = psnr(retarget_novel_t_novel_view_gt, retarget_novel_t_novel_view).mean().double()
                        Lssim_retarget_novel_time_view = 1.0 - ssim(retarget_novel_t_novel_view.reshape(-1, render_in.shape[-3], render_in.shape[-2], render_in.shape[-1]), (retarget_novel_t_novel_view_gt).reshape(-1, gt_in.shape[-3], gt_in.shape[-2], gt_in.shape[-1]))
                        lpips_retarget_novel_time_view = self.lpips(retarget_novel_t_novel_view.reshape(-1, render_in.shape[-3], render_in.shape[-2], render_in.shape[-1]), retarget_novel_t_novel_view_gt.reshape(-1, gt_in.shape[-3], gt_in.shape[-2], gt_in.shape[-1])).mean().double()
                        val_psnr_retarget_novel_time_view_list.append(psnr_retarget_novel_time_view.item())
                        ssim_retarget_novel_time_view_list.append(Lssim_retarget_novel_time_view.mean().item())
                        lpips_retarget_novel_time_view_list.append(lpips_retarget_novel_time_view.item())

                        retarget_gt = data['imgs'][:, :-1]
                        retarget_images_bw = predictions['retarget_images_bw']
                        psnr_retarget_bw = psnr(retarget_gt, retarget_images_bw).mean().double()
                        val_psnr_retarget_bw_list.append(psnr_retarget_bw.item())
                        l_retarget_bw = 0.8*l2_loss(retarget_images_bw, retarget_gt) + 0.2*(1.0 - ssim(retarget_images_bw.reshape(-1, c, h, w), (retarget_gt).reshape(-1, c, h, w))) + 0.2* self.lpips(retarget_images_bw.reshape(-1, c, h, w), (retarget_gt).reshape(-1, c, h, w)) 
                        retarget_gt = data['imgs'][:, 1:]
                        retarget_images_fw = predictions['retarget_images_fw']
                        psnr_retarget_fw = psnr(retarget_gt, retarget_images_fw).mean().double()
                        val_psnr_retarget_fw_list.append(psnr_retarget_fw.item())
                        
                        Lssim = 1.0 - ssim(render_in.reshape(-1, render_in.shape[-3], render_in.shape[-2], render_in.shape[-1]), (gt_in).reshape(-1, gt_in.shape[-3], gt_in.shape[-2], gt_in.shape[-1]))
                        psnr_value = psnr(render_in, gt_in).mean().double()
                        psnr_list.append(psnr_value.item())
                        ssim_list.append(Lssim.mean().item())
                        
                        try:
                            l_flow_bw, _, flow_mask_bw = flow_loss(predictions["flow_bw"], data["flow_bw"][:,1], data["flow_fw"][:,0], (data['masks']>0.66).float()[:,1])
                            l_flow_fw, _, flow_mask_fw = flow_loss(predictions["flow_fw"], data["flow_fw"][:,0], data["flow_bw"][:,1], (data['masks']>0.66).float()[:,1])
                            l_flow = l_flow_bw + l_flow_fw
                        except:
                            l_flow = torch.tensor(0)
                        flow_loss_list.append(l_flow.item())
                        Lssim_novel = 1.0 - ssim(render_novel.reshape(-1, render_novel.shape[-3], render_novel.shape[-2], render_novel.shape[-1]), (gt_novel).reshape(-1, gt_novel.shape[-3], gt_novel.shape[-2], gt_novel.shape[-1]))
                        psnr_novel = psnr(render_novel, gt_novel).mean().double()
                        lpips_novel = self.lpips(render_novel.reshape(-1, render_in.shape[-3], render_in.shape[-2], render_in.shape[-1]), gt_novel.reshape(-1, gt_in.shape[-3], gt_in.shape[-2], gt_in.shape[-1])).mean().double()
                        psnr_novel_list.append(psnr_novel.item())
                        ssim_novel_list.append(Lssim_novel.mean().item())
                        lpips_novel_list.append(lpips_novel.mean().item())

                        if idx == show_idx:
                            try:
                                image = torchvision.utils.make_grid(flow_to_color(data["flow_bw"][:,1].reshape(-1, data["flow_bw"][:,1].shape[-3], data["flow_bw"][:,1].shape[-2], data["flow_bw"][:,1].shape[-1]), flow_mask_bw)) 
                                self.logger.writer.add_image("Val GT Flow Bw", image, global_step=self.total_steps)

                                image = torchvision.utils.make_grid(flow_to_color(predictions["flow_bw"].reshape(-1, predictions["flow_bw"].shape[-3], predictions["flow_bw"].shape[-2], predictions["flow_bw"].shape[-1]), flow_mask_bw)) 
                                self.logger.writer.add_image("Val Pred Flow Bw", image, global_step=self.total_steps)

                                image = torchvision.utils.make_grid(flow_to_color(data["flow_fw"][:,0].reshape(-1, data["flow_fw"][:,1].shape[-3], data["flow_fw"][:,1].shape[-2], data["flow_fw"][:,1].shape[-1]), flow_mask_fw)) 
                                self.logger.writer.add_image("Val GT Flow", image, global_step=self.total_steps)

                                image = torchvision.utils.make_grid(flow_to_color(predictions["flow_fw"].reshape(-1, predictions["flow_fw"].shape[-3], predictions["flow_fw"].shape[-2], predictions["flow_fw"].shape[-1]), flow_mask_fw)) 
                                self.logger.writer.add_image("Val Pred Flow", image, global_step=self.total_steps)
                            except:
                                pass

                            image = torchvision.utils.make_grid(render_in.reshape(-1, render_in.shape[-3], render_in.shape[-2], render_in.shape[-1]))
                            self.logger.writer.add_image(f"Rendered Val {index}", image, global_step=self.total_steps)
                    
                            image = torchvision.utils.make_grid(gt_in.reshape(-1, gt_in.shape[-3], gt_in.shape[-2], gt_in.shape[-1])) 
                            self.logger.writer.add_image(f"GT Val {index}", image, global_step=self.total_steps)

                            # image = torchvision.utils.make_grid(torch.abs(retarget_images-retarget_gt).reshape(-1, retarget_images.shape[-3], retarget_images.shape[-2], retarget_images.shape[-1]))
                            # self.logger.writer.add_image(f"Retargeted Residual Val {index}", image, global_step=self.total_steps)

                            # image = torchvision.utils.make_grid(torch.abs(retarget_images-render_in[:, 1:]).reshape(-1, retarget_images.shape[-3], retarget_images.shape[-2], retarget_images.shape[-1]))
                            # self.logger.writer.add_image(f"Retargeted Displace Val {index}", image, global_step=self.total_steps)

                            image = torchvision.utils.make_grid(retarget_novel_t_images_fw.reshape(-1, retarget_novel_t_images_fw.shape[-3], retarget_novel_t_images_fw.shape[-2], retarget_novel_t_images_fw.shape[-1]))
                            self.logger.writer.add_image(f"Rendered Novel Time Val FW {index}", image, global_step=self.total_steps)

                            image = torchvision.utils.make_grid(retarget_novel_t_images_bw.reshape(-1, retarget_novel_t_images_bw.shape[-3], retarget_novel_t_images_bw.shape[-2], retarget_novel_t_images_bw.shape[-1]))
                            self.logger.writer.add_image(f"Rendered Novel Time Val BW {index}", image, global_step=self.total_steps)

                            image = torchvision.utils.make_grid(retarget_novel_t_gt.reshape(-1, retarget_novel_t_gt.shape[-3], retarget_novel_t_gt.shape[-2], retarget_novel_t_gt.shape[-1]))
                            self.logger.writer.add_image(f"GT Novel Time Val {index}", image, global_step=self.total_steps)

                            image = torchvision.utils.make_grid(render_novel.reshape(-1, render_novel.shape[-3], render_novel.shape[-2], render_novel.shape[-1]))
                            self.logger.writer.add_image(f"Rendered Novel Val {index}", image, global_step=self.total_steps)

                            image = torchvision.utils.make_grid(retarget_images_fw.reshape(-1, retarget_images_fw.shape[-3], retarget_images_fw.shape[-2], retarget_images_fw.shape[-1]))
                            self.logger.writer.add_image(f"Rendered Retargeted FW Val {index}", image, global_step=self.total_steps)

                            image = torchvision.utils.make_grid(retarget_images_bw.reshape(-1, retarget_images_bw.shape[-3], retarget_images_bw.shape[-2], retarget_images_bw.shape[-1]))
                            self.logger.writer.add_image(f"Rendered Retargeted BW Val {index}", image, global_step=self.total_steps)
                            
                            image = torchvision.utils.make_grid(gt_novel.reshape(-1, gt_novel.shape[-3], gt_novel.shape[-2], gt_novel.shape[-1])) 
                            self.logger.writer.add_image(f"GT Novel Val {index}", image, global_step=self.total_steps)
                        tmp_novel = retarget_novel_t_novel_view.reshape(-1, retarget_novel_t_novel_view.shape[-3], retarget_novel_t_novel_view.shape[-2], retarget_novel_t_novel_view.shape[-1]).detach()
                        Path('%s/novel/' % (cfg.record.show_path)).mkdir(exist_ok=True, parents=True)
                        tmp_img_name = '%s/novel/%s_%s_%d.png' % (cfg.record.show_path, data["scene"][0], index, idx)
                        save_image(tmp_novel, tmp_img_name, normalize=True)

                        tmp_novel = gt_in.reshape(-1, gt_in.shape[-3], gt_in.shape[-2], gt_in.shape[-1]).detach()
                        Path('%s/input/' % (cfg.record.show_path)).mkdir(exist_ok=True, parents=True)
                        tmp_img_name = '%s/input/%s_%s_%d.png' % (cfg.record.show_path, data["scene"][0], index, idx)
                        save_image(tmp_novel, tmp_img_name, normalize=True)

                        tmp_novel = render_novel.reshape(-1, render_novel.shape[-3], render_novel.shape[-2], render_novel.shape[-1]).detach()
                        Path('%s/lastframes/' % (cfg.record.show_path)).mkdir(exist_ok=True, parents=True)
                        tmp_img_name = '%s/lastframes/%s_%s_%d.png' % (cfg.record.show_path, data["scene"][0], index, idx)
                        save_image(tmp_novel, tmp_img_name, normalize=True)


                val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
                val_ssim = np.round(np.mean(np.array(ssim_list)), 4)
                val_novel_psnr = np.round(np.mean(np.array(psnr_novel_list)), 4)
                val_novel_ssim = np.round(np.mean(np.array(ssim_novel_list)), 4)
                val_novel_lpips = np.round(np.mean(np.array(lpips_novel_list)), 4)
                val_psnr_retarget_fw= np.round(np.mean(np.array(val_psnr_retarget_fw_list)), 4)
                val_psnr_retarget_fw_novel_time= np.round(np.mean(np.array(val_psnr_retarget_novel_time_fw_list)), 4)
                val_psnr_retarget_bw= np.round(np.mean(np.array(val_psnr_retarget_bw_list)), 4)
                val_psnr_retarget_bw_novel_time= np.round(np.mean(np.array(val_psnr_retarget_novel_time_bw_list)), 4)
                val_psnr_retarget_novel_time = np.round(np.mean(np.array(val_psnr_retarget_novel_time_list)), 4)
                val_ssim_retarget_novel_time = np.round(np.mean(np.array(ssim_retarget_novel_time_list)), 4)
                val_lpips_retarget_novel_time = np.round(np.mean(np.array(lpips_retarget_novel_time_list)), 4)

                val_psnr_retarget_novel_time_view = np.round(np.mean(np.array(val_psnr_retarget_novel_time_view_list)), 4)
                val_ssim_retarget_novel_time_view = np.round(np.mean(np.array(ssim_retarget_novel_time_view_list)), 4)
                val_lpips_retarget_novel_time_view = np.round(np.mean(np.array(lpips_retarget_novel_time_view_list)), 4)
                
                flow_loss_val = np.round(np.mean(np.array(flow_loss_list)), 4)

                logging.info(f"Validation Metrics {index} ({self.total_steps}): psnr {val_psnr}, ssim {val_ssim}, psnr novel {val_novel_psnr}, ssim novel {val_novel_ssim}, lpips novel {val_novel_lpips}, val_psnr_retarget_novel_time {val_psnr_retarget_novel_time}, val_ssim_retarget_novel_time {val_ssim_retarget_novel_time}, val_lpips_retarget_novel_time {val_lpips_retarget_novel_time}, val_psnr_retarget_novel_time_view {val_psnr_retarget_novel_time_view}, val_ssim_retarget_novel_time_view {val_ssim_retarget_novel_time_view}, val_lpips_retarget_novel_time_view {val_lpips_retarget_novel_time_view},")
                self.logger.write_dict({f'val_ssim_{index}':val_ssim, f'val_psnr_{index}': val_psnr, f'val_novel_ssim_{index}':val_novel_ssim, f'val_novel_psnr_{index}': val_novel_psnr, f'val_psnr_retarget_fw_{index}': val_psnr_retarget_fw,
                        f'val_psnr_retarget_novel_time_fw_{index}': val_psnr_retarget_fw_novel_time, f'val_psnr_retarget_bw_{index}': val_psnr_retarget_bw,
                        f'val_psnr_retarget_novel_time_bw_{index}': val_psnr_retarget_bw_novel_time, f'val_flow_loss_{index}': flow_loss_val, f'val_psnr_retarget_novel_time_{index}': val_psnr_retarget_novel_time,
                        f'val_ssim_retarget_novel_time_{index}':val_ssim_retarget_novel_time, f'val_lpips_retarget_novel_time_{index}':val_lpips_retarget_novel_time, }, write_step=self.total_steps)
       
        torch.cuda.empty_cache()
    def fetch_data(self, phase, index=None):
        if phase == 'train':
            attr_list = ["imgs", "masks", "novel_time", "novel_time_mask", "flow_fw", "flow_bw"]
            data_idx = 0
            if self.multi_dataset:
                data_idx = random.choices(range(len(self.train_iterator)), weights=self.cfg.probs, k=1)[0]
            try:
                data = next(self.train_iterator[data_idx])
            except:
                self.train_iterator[data_idx] = iter(self.train_loader[data_idx])
                data = next(self.train_iterator[data_idx])
                    
        elif phase == 'val':
            attr_list = ["imgs", "masks", "novel_time", "novel_time_mask", "flow_fw", "flow_bw"]
            # try:
            data = next(self.val_iterator)
            # except:
            #     self.val_iterator = iter(self.val_loader[index])
            #     data = next(self.val_iterator)

        for item in attr_list:
            if item in data.keys():
                data[item] = data[item].to(self.accelerator.device)

        for item in ["novel_view", "camera"]:
            if item in data.keys():
                for k in data[item].keys():
                    data[item][k] = data[item][k].to(self.accelerator.device)
                
        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')

        self.model.load_state_dict(ckpt['network'], strict=strict)
        logging.info(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            logging.info(f"Optimizer loading done")

    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, save_path)
        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["motionhead", "cross blocks", "scene blocks", "scene token", "scene lambda", "fusion"]:
                lr = self.scheduler(iteration)
                param_group['lr'] = lr

            if param_group["name"] in ["gshead", "camhead", "ptshead", "aggregator"]:
                lr = self.scheduler_ft(iteration)
                param_group['lr'] = lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default="config/train_stage3_motion_fusion.yaml")
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

    cfg.defrost()
    dt = datetime.today()
    cfg.depth = args.depth
    cfg.exp_name = '%s_%s%s' % (cfg.name, str(dt.month).zfill(2), str(dt.day).zfill(2))
    cfg.record.ckpt_path = "%s/%s/ckpt" % (args.log_path, cfg.exp_name)
    cfg.record.show_path = "%s/%s/show" % (args.log_path, cfg.exp_name)
    cfg.record.logs_path = "%s/%s/logs" % (args.log_path, cfg.exp_name)
    cfg.record.file_path = "%s/%s/file" % (args.log_path, cfg.exp_name)

    cfg.freeze()

    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)

    file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    torch.manual_seed(1314)
    np.random.seed(1314)

    trainer = Trainer(cfg)
    trainer.train()