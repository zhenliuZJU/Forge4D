<h1 align="center" style="font-size: 2.8em;">Forge4D</h1>

<h3 align="center">Feed-Forward 4D Human Reconstruction and Interpolation from Uncalibrated Sparse-view Videos</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2509.24209"><img src="https://img.shields.io/badge/arXiv-2509.24209-b31b1b" alt="arXiv Paper"></a>
  <a href="https://zhenliuzju.github.io/huyingdong/Forge4D"><img src="https://img.shields.io/badge/Project%20Page-Forge4D-orange" alt="Project Page"></a>
  <a href="https://github.com/zhenliuZJU/Forge4D"><img src="https://img.shields.io/badge/Code-GitHub-blue" alt="Code"></a>
</p>

<p align="center">
  <a href="https://zhenliuzju.github.io/huyingdong/">Yingdong Hu</a><sup>*</sup>,
  <a href="https://hyshkust.github.io/">Yisheng He</a><sup>*&dagger;</sup>,
  <a href="https://jinnan-chen.github.io/">Jinnan Chen</a>,
  <a href="https://weihao-yuan.com/">Weihao Yuan</a>,
  <a href="https://sites.google.com/site/kejieqiujack/home">Kejie Qiu</a>,
  <a href="https://zhlinup.github.io/">Zehong Lin</a>,
  <a href="https://sites.google.com/site/zhusiyucs/home">Siyu Zhu</a>,
  <a href="https://scholar.google.com/citations?user=GHOQKCwAAAAJ&hl=zh-CN&oi=ao">Zilong Dong</a>,
  <a href="https://eejzhang.people.ust.hk/">Jun Zhang</a>
</p>

<p align="center"><sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>&dagger;</sup>Corresponding author &nbsp;·&nbsp; ECCV 2026</p>

<div align="center">
  <img src="./assets/teaser.png" alt="Forge4D teaser" width="800">
</div>

Forge4D reconstructs a **dynamic 3D human** from a small number of **uncalibrated**
sparse-view videos in a single forward pass, and produces dense **novel-view and
novel-time** renderings without per-scene optimization.

Given 4 synchronized (but uncalibrated) videos of a person, the model streams
over the frames, fuses cross-view information with a learnable **state token**,
predicts dense **bidirectional scene flow** between consecutive frames, and
merges the per-frame 3D Gaussians with **occlusion-aware fusion** before
splatting them to arbitrary novel viewpoints and novel times.

---

## News

- [x] Release full training code (stage 1–3) and inference demo
- [ ] Release pretrained checkpoints (*coming soon*)

## Method overview

| Module | Role |
|---|---|
| `forge4d.models.Forge4D` | Feed-forward 4D reconstruction network (multi-frame D3D → D4D) |
| `forge4d.models.aggregator.StateTokenAggregator` | Cross-frame token aggregation with a learnable **state token** that carries temporal context |
| `forge4d.heads.CameraHead` | Predicts per-frame camera extrinsics / intrinsics |
| `forge4d.heads.DPTHead` (point head) | Predicts per-view dense 3D point maps (D3D) |
| `forge4d.heads.GaussianHead` | Predicts per-view 3D Gaussian attributes (rotation / scale / opacity / color) |
| `forge4d.heads.MotionHead` | Predicts dense bidirectional scene flow between consecutive frames |
| `forge4d.layers.fusion.GaussianFusion` | Occlusion-aware fusion of forward / backward Gaussians |

Training proceeds in three stages:

1. **Stage 1 — static reconstruction**: train the static multi-frame
   reconstruction backbone (single time step).
2. **Stage 2 — streaming with state tokens**: add the learnable state token and
   train on whole clips so temporal context accumulates across frames.
3. **Stage 3 — motion + fusion**: train dense motion prediction and the
   occlusion-aware Gaussian fusion head, enabling novel-time interpolation.

## Installation

```bash
conda create -n forge4d python=3.10 -y
conda activate forge4d
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

Notes:
- `gsplat` compiles CUDA kernels on first use. On newer NVIDIA GPUs you may
  need to set the architecture list, e.g.
  `export TORCH_CUDA_ARCH_LIST="8.9+PTX"` (Ada) or the appropriate value for
  your GPU before importing the renderer.
- `xformers` provides the efficient attention in the VGGT backbone and must
  match your PyTorch / CUDA version.
- A GPU with ≥ 24 GB VRAM is recommended for the 518×518 4-view configuration.

## Data preparation

### DNA-Rendering

Forge4D is trained and evaluated on
[DNA-Rendering](https://dna-rendering.github.io/). The dataset is expected
under a single root with the following layout (Part 3–6):

```
<data_root>/
└── DNA-Rendering/
    ├── Part3/   (and Part4 / Part5 / Part6)
    │   ├── calibration_full.json          # camera calibration (K, D, ext)
    │   └── main/
    │       └── <subject_id>/
    │           ├── Cam00/ 0001.jpg ...    # 4 synchronized views
    │           ├── Cam00/mask/ 0001.jpg ...
    │           ├── Cam01/ ...
    │           ├── Cam02/ ...
    │           └── Cam03/ ...
    └── (optional) flow/                   # precomputed optical flow for flow supervision
```

Point the configs at this root via `dataset.data_root` (or `--data_root` for
inference). Set `dataname` to `['DNA']`, `['DNAP3']`, … to select which parts
to load.

### Other supported datasets

The following dataset loaders are included (`lib/`):

| Dataset | Loader | Notes |
|---|---|---|
| [HuMMan](https://caizhongang.github.io/projects/HuMMan/) | `HuMManDataset` | multi-view 4D humans |
| [GeneBody](https://github.com/nii-yamagishilab/project-CVPR2022-GeneBody) | `GenebodyDataset` | |
| [Nerfstudio / NERSEMBLE](https://mobile-capture.github.io/) | `NersembleDataset` | |
| SelfCap / VVC / MHC / 4D4K | `SelfCapDataset`, `VVCDataset`, `MHCDataset`, `DNADataset4K4D` | additional 4D human sources |

### Synthetic data rendering

`prepare_data/render_data.py` (with the vendored `taichi_three`) renders
synthetic multi-view human data from meshes (SMPL-family), producing the same
image / mask / depth / camera layout consumed by the dataset loaders:

```bash
python prepare_data/render_data.py --help
```

## Checkpoints

| Model | Description | Download |
|---|---|---|
| Stage 1 | Static reconstruction (518px) | *coming soon* |
| Stage 2 | Streaming + state tokens (`forge4d_stage2_streaming`) | *coming soon* |
| Stage 3 | Motion + occlusion-aware fusion (`forge4d_stage3_motion_fusion`) | *coming soon* |

## Training

All training scripts share the interface
`--cfg_path <yaml> --log_path <dir>`. Checkpoints are written to
`<log_path>/<exp_name>/ckpt/`.

**Stage 1 — static reconstruction** (edit `config/train_stage1_static.yaml`
and set the pre-trained static VGGT checkpoint in `stage1_ckpt`):

```bash
python runfiles/train.py \
    --cfg_path config/train_stage1_static.yaml \
    --log_path ./runs
```

**Stage 2 — streaming with state tokens** (set `stage1_ckpt` to the Stage-1
checkpoint):

```bash
python runfiles/train_dynamic.py \
    --cfg_path config/train_stage2_streaming.yaml \
    --log_path ./runs
```

**Stage 3 — motion + fusion** (set `stage1_ckpt` to the Stage-2 checkpoint):

```bash
python runfiles/train_matchedfusion.py \
    --cfg_path config/train_stage3_motion_fusion.yaml \
    --log_path ./runs
```

Training uses [Accelerate](https://huggingface.co/docs/accelerate) and supports
distributed launch via `accelerate config` / `accelerate launch`.

## Inference

Run the feed-forward 4D reconstruction + novel-view / novel-time rendering demo
on a DNA-Rendering sequence with a trained Stage-3 checkpoint:

```bash
python runfiles/inference.py \
    --weight /path/to/forge4d_stage3_motion_fusion_latest.pth \
    --data_root /path/to/datasets \
    --test_ids 0007_04 \
    --output_dir ./output
```

Key flags (see `python runfiles/inference.py --help`):

| Flag | Default | Description |
|---|---|---|
| `--weight` | *(required)* | Trained Forge4D checkpoint (state dict with a `network` key) |
| `--cfg` | `config/inference.yaml` | Inference config |
| `--data_root` | cfg value | Dataset root (overrides `cfg.dataset.data_root`) |
| `--test_ids` | `0007_04` | Comma-separated subject ids, matched against `<data_root>/DNA-Rendering/main/` |
| `--output_dir` | `./output` | Rendered images (`3DGS/`, `render/`) and videos (`videos/`) |
| `--frames` | `50` | Frames to process per subject |
| `--extend_frames` | `2` | Novel-time interpolation steps between consecutive frames |
| `--motion_dimension` | `3` | Motion-head output channels (dense 3D bidirectional motion) |

### Novel-view video generation

`runfiles/generate_novel_video.py` renders a camera-trajectory video from a
static (Stage-1) reconstruction:

```bash
python runfiles/generate_novel_video.py \
    --cfg_path config/generate.yaml \
    --log_path ./runs
```

## Evaluation

The reconstructed per-frame 3D Gaussians can be exported to PLY meshes /
point clouds (`lib.utils.save_3dgs_ply`, `tools/pointcloud.py`) and evaluated
against ground-truth scans with standard geometric metrics (Chamfer distance,
P2S, normal consistency) and appearance metrics (PSNR / SSIM / LPIPS) at the
novel viewpoints used in the paper.

## Repository structure

```
Forge4D/
├── assets/                 # teaser / figures
├── config/                 # yacs configs + config builder
│   ├── train_stage1_static.yaml
│   ├── train_stage2_streaming.yaml
│   ├── train_stage3_motion_fusion.yaml
│   ├── inference.yaml
│   └── generate.yaml
├── runfiles/               # entry-point scripts
│   ├── train.py                 # Stage 1 (static)
│   ├── train_dynamic.py         # Stage 2 (streaming + state tokens)
│   ├── train_matchedfusion.py   # Stage 3 (motion + occlusion-aware fusion)
│   ├── inference.py             # reconstruction + novel view/time demo
│   └── generate_novel_video.py  # novel-view video from static model
├── forge4d/                # model package (Meta VGGT backbone + Forge4D heads)
│   ├── models/             # Forge4D, StateTokenAggregator, VGGT (static backbone)
│   ├── heads/              # CameraHead, DPTHead (point), GaussianHead, MotionHead
│   ├── layers/             # attention / transformer blocks / fusion
│   └── utils/              # pose encoding, geometry helpers
├── lib/                    # datasets, Gaussian rendering, losses, utils
├── gaussian_renderer/      # differentiable 3D Gaussian splatter (gsplat)
├── tools/                  # camera trajectory, point cloud helpers
└── prepare_data/           # synthetic data rendering (vendored taichi_three)
```

## Acknowledgements

This project builds on
[**VGGT**](https://github.com/facebookresearch/vggt) (Meta Platforms), the
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
renderer (Inria GRAPHDECO), and [taichi_three](https://github.com/taichi-dev/taichi_three).
See [LICENSE](LICENSE) for the third-party attribution and license terms.

## Citation

```bibtex
@misc{hu2025forge4dfeedforward4dhuman,
    title={Forge4D: Feed-Forward 4D Human Reconstruction and Interpolation from Uncalibrated Sparse-view Videos},
    author={Yingdong Hu and Yisheng He and Jinnan Chen and Weihao Yuan and Kejie Qiu and Zehong Lin and Siyu Zhu and Zilong Dong and Jun Zhang},
    year={2025},
    eprint={2509.24209},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2509.24209},
}
```
