# Forge4D：代码与论文一致性核对报告

> 核对对象：arXiv:2509.24209（ECCV 2026）vs 本仓库开源代码。
> 核对日期：2026-08-26。

## A. 发现并已修复的问题

这些是开源化整理过程中核对出的、会让按 README 复现的结果与论文不一致的问题，已直接修复。

### A1. Stage-3 训练脚本中 motion head 被注释（最严重）
- 现象：`runfiles/train_matchedfusion.py` 的 `stage: 'motion'` 分支里，
  `{'params': motion_head, ...}` 被注释，冻结循环也只冻结 aggregator / point /
  camera / gaussian / state-token → **`motion` 模式只训练 Gaussian fusion，
  motion head 从不更新**。而 stage-2（`train_dynamic.py`）的优化器也不含
  motion head → 按 README 的 stage-2 → stage-3 链走完，motion head 全程是随机初始化，
  与论文"stage-3 训练 motion blocks / Gaussian fusion / motion head"直接矛盾。
- 根因（原版如此，非重构引入）：作者实际流程是先用 `stage: 'together'`（唯一训
  motion head 的分支）跑出 `dynamic_wflow_518_dual_fusion_alldata` 检查点，再用
  `stage: 'motion'` 只 finetune fusion；论文只描述了一个 stage-3，开源版按 README
  无法复现该链路。
- 修复：取消 `motion` 分支中 motion head 的注释 → `motion` 模式 = 训练
  motion head + fusion（其余冻结），与论文 stage-3 一致。

### A2. Stage-3 配置缺失 `loss:` 段 → L_flow 权重为 0
- 现象：`config/train_stage3_motion_fusion.yaml` 没有 `loss:` 段，
  `cfg.loss.flow` 走默认值 0 → `lambda_flow = 0 * 1/h = 0`，论文 Eq.(7) 的
  occlusion-aware optical flow loss 虽被计算但权重为 0（等于没训）。
  原版 `train_motion_dual_518.yaml` 同样无 `loss:` 段。
- 修复：补 `loss: {flow: 10., lpips: 0.2}`。flow 权重取仓库内
  `train_motion_518.yaml` 的先例值 10；注意代码 `lambda_flow = cfg.loss.flow * 1/h`
  会再除以高度（≈0.019），见 B9。
  若要复现原版"仅融合"训练，可设 `flow: 0.`。

### A3. Stage-2 配置 `stage` 字段误导
- 现象：`config/train_stage2_streaming.yaml` 的 `stage: 'static'`，而这是流式训练配置。
  该字段在 `train_dynamic.py` 中不被读取（inert）。
- 修复：改为 `stage: 'streaming'` 并注明该字段仅作说明。

### A4. README 中 motion 头通道数描述错误
- 现象：README 写 `--motion_dimension` "3: flow, 4: flow + opacity"，但生产
  `forward` / `forward_matching_fusion` 只用 `motions[:, :, :3]`，第 4 通道（opacity）
  只存在于已删除的实验变体 `forward_dual` 中。
- 修复：改为"dense 3D bidirectional motion"，并删去 MotionHead 描述里的
  "+ opacity control"。

## B. 原版代码与论文的差异（仅记录，未改动）

以下差异在原版代码中就已存在。为保持对原论文实验配置的忠实复现，未作修改，
此处如实列出，供复现与审稿对照。

| # | 主题 | 论文 | 代码（原版/本仓库） |
|---|---|---|---|
| B1 | **Metric gauge** | 学到的 scale head 预测 p̂gauge，推理时用其恢复 metric-scale 3D 点 | 无预测 gauge 头；只在训练/有 GT 相机时用 `scale = mean(‖T_pred‖/‖T_gt‖)` 缩放相机平移。`align_test_step`（测试时优化 novel-view 相机）存在 ✓ |
| B2 | **State token 数量** | "a learnable state token"（单数描述） | `num_state_tokens = 1024`（原版 `num_scene_tokens = 1024`） |
| B3 | **λ_SSIM** | 0.25 | 0.2（`0.8·L2 + 0.2·(1−SSIM)`，写死在损失公式） |
| B4 | **λ_LPIPS** | 0.25 | stage-1 硬编码 0.5（train.py）；stage-2/3 配置 0.2 |
| B5 | **各阶段 LR** | 每阶段 5e-5 | 原版生产配置：stage-1 1e-6、stage-2 5e-5（`train_DNA_518`）/ 1e-4（`train_abla_noflow_518` 消融）、stage-3 1e-6。本仓库 config 忠于原版 |
| B6 | **stage-1 迭代数** | 100k | 200k（`train_static_518` 原版如此） |
| B7 | **LR 衰减终点** | "线性降到 1e-5" | `get_lr_func(lr_final=1e-7)` |
| B8 | **zero-init** | scale / offset / color-offset 头与 motion blocks zero-initialized | `offset_init()` / `motion_init()` 方法存在但构造时调用被注释（`# self.offset_init()`），未实际 zero-init。如需对齐论文可取消注释 |
| B9 | **flow loss 归一化** | Eq.(7) 无系数求和 | `lib/loss.py::flow_loss` 返回 per-element **mean**，再乘 `1/h` → 双重归一化，λ_flow 量级偏小 |
| B10 | **stage-2 冻结范围** | state attention + state tokens + scale/Gaussian/pose heads @ 5e-5，其余冻结 | 只训 scene_blocks + state_token @ cfg.lr(5e-5)；Gaussian/pose/camera 冻结在 1e-6 |
| B11 | **stage-1 参数组** | Gaussian/scale 头 @ 5e-5，其他 @ 1e-5 | gs_head @ cfg.lr(1e-6)；aggregator @ 1e-6；camera/point @ 1e-5 |

### B0. 原始 stage-3 的两步链路（背景说明）
作者原版 pipeline 实际是：`stage-3a`（`stage: 'together'`，训出 motion head +
fusion，检查点 `dynamic_wflow_518_dual_fusion_alldata`）→ `stage-3b`
（`stage: 'motion'`，只 finetune fusion，检查点 `dynamic_wflow_518_matched_fusion_alldata`）。
论文将两者合并描述为一个 stage-3。本仓库经 A1/A2 修复后，单一 `stage: 'motion'`
即可自洽地训练 motion head + fusion + flow loss，与论文描述一致。

## C. 确认与论文一致的部分

- 输入 4 视角、分辨率 518×518（config `train_view_num: 4`、`resolution: 518`）。
- 遮挡感知融合阈值 **τ = 0.05**（`flow_matching_mask_fw = (dist > 0.05)`）。
- 双向运动场 **M ∈ R^(2×3×H×W)**（`motion_maps = motions[:, :, :3]`，fw/bw 各 3 通道）。
- 三阶段训练结构：stage-1 静态（`train.py`）、stage-2 流式 state token
  （`train_dynamic.py`）、stage-3 运动+融合（`train_matchedfusion.py`）。
- 三阶段损失项齐全：L_retarget（`l_novel_t` 中 retarget 项）、L_flow（`l_flow`）、
  L_fusion（novel-time 渲染项）。
- 3D Gaussian 属性 14 通道 = offset 3 + opacity 1 + color 3 + rot(quat) 4 + scale 3，
  对应论文的 P / O / C / Q / S。
- batch size 8 = 每 GPU 1 × 8 GPU（config `batch_size: 1` + Accelerate 分布式）。
- `align_test_step` 对应论文"pose-free 方法在评测时优化 novel-view 相机"。
- 数据集：DNA-Rendering Part 3–6 训练、Genebody/DNA 验证，与论文一致。
- 训练 LR 的**结构**与论文一致：仅活跃头以 `cfg.lr` 训练、其余冻结在 1e-6。
