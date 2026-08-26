from yacs.config import CfgNode as CN


class ConfigStereoHuman:
    def __init__(self):
        self.cfg = CN()
        self.cfg.name = ''
        self.cfg.stage1_ckpt = None
        self.cfg.restore_ckpt = None
        self.cfg.pretrain_weight = None
        self.cfg.DataParallel = False
        self.cfg.lr = 0.0
        self.cfg.wdecay = 0.0
        self.cfg.batch_size = 0
        self.cfg.num_steps = 0
        self.cfg.dataname = ['DNA']
        self.cfg.val_dataname = ['DNA']
        self.cfg.probs = [1]
        self.cfg.pointmap = False
        self.cfg.timesteps = 8
        self.cfg.sampling_rate = 2
        self.cfg.scaling_factor = 1.0
        self.cfg.stage = "togther"

        self.cfg.loss = CN()
        self.cfg.loss.flow = 0.0
        self.cfg.loss.lpips = 0.2
        self.cfg.loss.scene_flow = 0.0
        self.cfg.loss.retarget = 1.0

        self.cfg.dataset = CN()
        self.cfg.dataset.resolution = 518
        self.cfg.dataset.novel_resolution = 518
        self.cfg.dataset.train_view_num = 4
        self.cfg.dataset.anchor = False
        self.cfg.dataset.source_id = None
        self.cfg.dataset.train_novel_id = None
        self.cfg.dataset.val_novel_id = None
        self.cfg.dataset.use_hr_img = None
        self.cfg.dataset.use_processed_data = None
        self.cfg.dataset.data_root = ''
        # gsussian render settings
        self.cfg.dataset.bg_color = [1.0, 1.0, 1.0]
        self.cfg.dataset.zfar = 100.0
        self.cfg.dataset.znear = 0.01
        self.cfg.dataset.trans = [0.0, 0.0, 0.0]
        self.cfg.dataset.scale = 1.0

        self.cfg.record = CN()
        self.cfg.record.ckpt_path = None
        self.cfg.record.show_path = None
        self.cfg.record.logs_path = None
        self.cfg.record.file_path = None
        self.cfg.record.loss_freq = 0
        self.cfg.record.eval_freq = 0

    def get_cfg(self):
        return self.cfg.clone()
    
    def load(self, config_file):
        self.cfg.defrost()
        self.cfg.merge_from_file(config_file)
        self.cfg.freeze()
