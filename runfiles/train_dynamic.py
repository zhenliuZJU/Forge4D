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
from config.transformer_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.loss import l1_loss, l2_loss, ssim, psnr, opacity_regular, scale_regular, PerceptualLoss
from lib.utils import compute_norm, compute_scale, get_lr_func, gradient

from forge4d.models.forge4d import Forge4D

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs

import torchvision
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


Dataset = {'Nersemble': NersembleDataset, 'DNA': DNADataset, 'DNAP3': DNADatasetPart3, 'DNAP4': DNADatasetPart4, 'DNAP5': DNADatasetPart5, 'DNAP6': DNADatasetPart6, 'HuMMan': HuMManDataset, 'DNA4K4D': DNADataset4K4D, 'Genebody':GenebodyDataset}
ValDataset = {'DNA': DNADataset, 'DNA4K4D': DNADataset4K4D, 'HuMMan': HuMManDataset, 'Genebody':GenebodyDataset}

class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file
        
        self.model = Forge4D(point_map=self.cfg.pointmap)
        self.lpips = PerceptualLoss(device='cuda')

        # for name, module in self.model.aggregator.named_children():
        #     if name not in ["scene_blocks", "lambda_t"]:  # 排除 blocks 子模块
        #         for param in module.parameters():
        #             param.requires_grad = False
        l = [
            {'params': self.model.gaussian_head.parameters(), 'lr': 0.000001, "name": "gaussian head"},
            {'params': [p for p in self.model.aggregator.parameters() 
                if p not in set(self.model.aggregator.scene_blocks.parameters())
                and p not in set(self.model.aggregator.lambda_t)], 'lr': 0.000001, "name": "aggregator"},
            {'params': self.model.aggregator.scene_blocks.parameters(), 'lr': self.cfg.lr, "name": "scene blocks"},
            # {'params': self.model.aggregator.lambda_t, 'lr': self.cfg.lr, "name": "scene lambda"},
            {'params': [self.model.state_token], 'lr': self.cfg.lr, "name": "state token"},
            {'params': self.model.camera_head.parameters(), 'lr': 0.000001, "name": "camhead"},
            {'params': self.model.point_head.parameters(), 'lr': 0.000001, "name": "ptshead"},
        ]

        self.optimizer = optim.AdamW(l, lr=0.0, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = get_lr_func(lr_init=self.cfg.lr, lr_final=1e-7, max_steps=self.cfg.num_steps, warmup_steps=5000)
        self.scheduler_ft = get_lr_func(lr_init=0.000001, lr_final=1e-7, max_steps=self.cfg.num_steps, warmup_steps=5000)

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0
        
        self.accelerator = Accelerator(gradient_accumulation_steps=12, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

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
            self.val_set.append(ValDataset[dataset_name](self.cfg.dataset, phase='val', sampling_rate=self.cfg.sampling_rate, timesteps=8))
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
        # self.model.eval()
        # self.run_eval()
        # self.model.train()
        # exit()
        for step in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()

            data = self.fetch_data(phase='train')

            metrics = {}
             
            mask = (data['masks']>0.99).float()#.expand_as(data['xyz'])
            bg_color = random.choice([[0.,0.,0.,],[0.5,0.5,0.5],[1.,1.,1.]])
            data['imgs'] = data['imgs'] *mask + (1-mask)*(torch.tensor(bg_color).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-2)).to(data['imgs'])
            predictions  = self.model(data, data['imgs'], data['masks'], bg_color=bg_color)
            
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
                # ((norm_pred/norm_gt-torch.mean(norm_pred/norm_gt, dim=1, keepdim=True))**2).mean()# exclude view 1
                # print(l_t, norm_gt/norm_pred, torch.mean(norm_gt/norm_pred, dim=1, keepdim=True))
                # exit()
                l_cam = l2_loss(pose_gt[:,:,:,3:], predictions['pose_encoding'][:,:,:,3:])+l_t

            lambda_lpips = 0.2 #if step > 20000 else 0.0
            l_input = 0.8 * L2 + 0.2 * Lssim + lambda_lpips* lpips #if step > 20000 else 0
             
            # lambda_lpips = 0.5 #if step > 1000 else 0.0
            # l_input = L2 + lambda_lpips* lpips 
            
            mask_novel = (data['novel_view']['mask']>0.99).float()
            data['novel_view']['img'] = data['novel_view']['img'] *mask_novel + (1-mask_novel)*(torch.tensor(bg_color).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-2)).to(data['imgs'])

            render_novel = predictions["novel_img"].squeeze()
            gt_novel = data['novel_view']['img']
            psnr_novel = psnr(render_novel, gt_novel).mean().double()
            l_novel = 0.8*l2_loss(render_novel, gt_novel) + 0.2*(1.0 - ssim(render_novel.reshape(-1, c, h, w), (gt_novel).reshape(-1, c, h, w))) + lambda_lpips* self.lpips(render_novel.reshape(-1, c, h, w), (gt_novel).reshape(-1, c, h, w)) 
            # l_novel = l2_loss(render_novel, gt_novel) + lambda_lpips* self.lpips(render_novel.reshape(-1, c, h, w), (gt_novel).reshape(-1, c, h, w)) 
            
            lambda_novel = 1 #if step > 20000 else 0.0
            lambda_input = 1
            lambda_cam = 1
            loss = lambda_input*l_input + l_depth + l_pc + lambda_cam*l_cam + lambda_novel*l_novel
            # loss =  L2  + 0.5 * lpips#+ L_intr

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                mask = (data['masks']>0.99).float()
                image = torchvision.utils.make_grid(render_in.reshape(-1, render_in.shape[-3], render_in.shape[-2], render_in.shape[-1]))
                self.logger.writer.add_image("Rendered Train", image, global_step=self.total_steps)
                
                image = torchvision.utils.make_grid(gt_in.reshape(-1, gt_in.shape[-3], gt_in.shape[-2], gt_in.shape[-1])) 
                self.logger.writer.add_image("GT Train", image, global_step=self.total_steps)

                image = torchvision.utils.make_grid(render_novel.reshape(-1, render_novel.shape[-3], render_novel.shape[-2], render_novel.shape[-1]))
                self.logger.writer.add_image("Rendered Novel Train", image, global_step=self.total_steps)
                            
                image = torchvision.utils.make_grid(gt_novel.reshape(-1, gt_novel.shape[-3], gt_novel.shape[-2], gt_novel.shape[-1])) 
                self.logger.writer.add_image("GT Novel Train", image, global_step=self.total_steps)

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

                if self.total_steps % (5*self.cfg.record.loss_freq) == 0:
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
            l_cam = self.accelerator.gather(l_cam).mean()
            l_novel = self.accelerator.gather(l_novel).mean()
            psnr_novel = self.accelerator.gather(psnr_novel).mean()
            
            metrics.update({
                'loss': loss.item(),
                'ssim': Lssim.item(),
                'psnr': psnr_value.item(),
                'lpips': lpips.item(),
                'l_depth': l_depth.item(),
                'l_pt': l_pc.item(),
                'l_cam': l_cam.item(),
                'l_novel': l_novel.item(),
                'psnr_novel': psnr_novel.item(),
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
        psnr_list, ssim_list = [], []
        psnr_novel_list, ssim_novel_list = [], []
        for index in range(len(self.val_set)):
                self.val_loader = DataLoader(self.val_set[index], batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
                self.len_val = int(len(self.val_loader) / self.val_set[index].val_boost)  # real length of val set
                self.val_iterator = iter(self.val_loader)
                show_idx = np.random.choice(list(range(self.len_val)), 1)
                for idx in range(self.len_val):
                    data = self.fetch_data(phase='val', index=index)
                    with torch.no_grad():
                        mask = (data['masks']>0.99).float()#.expand_as(data['xyz'])
                        data['imgs'] = data['imgs'] *mask + (1-mask)*(torch.tensor(self.cfg.dataset.bg_color).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-2)).to(data['imgs'])
                        predictions  = self.model(data, data['imgs'], data['masks'], bg_color=self.cfg.dataset.bg_color, is_val=True)
                
                        render_in = predictions["images"]
                        gt_in = data['imgs'] #data['novel_view']['img'].cuda()
                        
                        mask_novel = (data['novel_view']['mask']>0.99).float()
                        data['novel_view']['img'] = data['novel_view']['img'] *mask_novel + (1-mask_novel)*(torch.tensor(self.cfg.dataset.bg_color).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-2)).to(data['imgs'])
                        render_novel = predictions["novel_img"].squeeze()
                        gt_novel = data['novel_view']['img']
                        
                        Lssim = 1.0 - ssim(render_in.reshape(-1, render_in.shape[-3], render_in.shape[-2], render_in.shape[-1]), (gt_in).reshape(-1, gt_in.shape[-3], gt_in.shape[-2], gt_in.shape[-1]))
                        psnr_value = psnr(render_in, gt_in).mean().double()
                        psnr_list.append(psnr_value.item())
                        ssim_list.append(Lssim.mean().item())
                        
                        Lssim_novel = 1.0 - ssim(render_novel.reshape(-1, render_novel.shape[-3], render_novel.shape[-2], render_novel.shape[-1]), (gt_novel).reshape(-1, gt_novel.shape[-3], gt_novel.shape[-2], gt_novel.shape[-1]))
                        psnr_novel = psnr(render_novel, gt_novel).mean().double()
                        psnr_novel_list.append(psnr_novel.item())
                        ssim_list.append(Lssim_novel.mean().item())
                        if idx == show_idx:
                            image = torchvision.utils.make_grid(render_in.reshape(-1, render_in.shape[-3], render_in.shape[-2], render_in.shape[-1]))
                            self.logger.writer.add_image("Rendered Val", image, global_step=self.total_steps)
                    
                            image = torchvision.utils.make_grid(gt_in.reshape(-1, gt_in.shape[-3], gt_in.shape[-2], gt_in.shape[-1])) 
                            self.logger.writer.add_image("GT Val", image, global_step=self.total_steps)

                            image = torchvision.utils.make_grid(render_novel.reshape(-1, render_novel.shape[-3], render_novel.shape[-2], render_novel.shape[-1]))
                            self.logger.writer.add_image("Rendered Novel Val", image, global_step=self.total_steps)
                            
                            image = torchvision.utils.make_grid(gt_novel.reshape(-1, gt_novel.shape[-3], gt_novel.shape[-2], gt_novel.shape[-1])) 
                            self.logger.writer.add_image("GT Novel Val", image, global_step=self.total_steps)


        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        val_ssim = np.round(np.mean(np.array(ssim_list)), 4)
        val_novel_psnr = np.round(np.mean(np.array(psnr_novel_list)), 4)
        val_novel_ssim = np.round(np.mean(np.array(ssim_novel_list)), 4)

        logging.info(f"Validation Metrics ({self.total_steps}): psnr {val_psnr}, ssim {val_ssim}, psnr novel {val_novel_psnr}, ssim novel {val_novel_ssim}")
        self.logger.write_dict({'val_ssim':val_ssim, 'val_psnr': val_psnr, 'val_novel_ssim':val_novel_ssim, 'val_novel_psnr': val_novel_psnr}, write_step=self.total_steps)
       
        torch.cuda.empty_cache()

    def fetch_data(self, phase, index=None):
        if phase == 'train':
            data_idx = 0
            if self.multi_dataset:
                data_idx = random.choices(range(len(self.train_iterator)), weights=self.cfg.probs, k=1)[0]
            try:
                data = next(self.train_iterator[data_idx])
            except:
                self.train_iterator[data_idx] = iter(self.train_loader[data_idx])
                data = next(self.train_iterator[data_idx])
                    
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader[index])
                data = next(self.val_iterator)

        for item in ["imgs", "masks"]:
            data[item] = data[item].to(self.accelerator.device)

        for item in ["novel_view", "camera"]:
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
            if param_group["name"] in ["scene blocks", "scene token", "scene lambda"]:
                lr = self.scheduler(iteration)
                param_group['lr'] = lr

            if param_group["name"] in ["gshead", "camhead", "ptshead", "aggregator"]:
                lr = self.scheduler_ft(iteration)
                param_group['lr'] = lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default="config/train_stage2_streaming.yaml")
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