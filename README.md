# FastSESR: Fast Surface Extraction and Super-Resolution

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.10+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

FastSESR 是一个基于深度学习的点云曲面重建框架，采用两阶段（S1/S2）训练策略，实现从点云到三角网格的高效重建。项目支持多种数据集，并提供完整的训练、评估和重建流程。

## 📋 目录

- [环境配置](#-环境配置)
- [依赖库](#-依赖库)
- [数据集准备](#-数据集准备)
- [Stage 1 训练](#-stage-1-训练)
- [Stage 2 训练](#-stage-2-训练)
- [预训练模型](#-预训练模型)
- [重建与评估](#-重建与评估)
- [项目结构](#-项目结构)

---

## 🛠 环境配置

### 1. 创建 Conda 虚拟环境

```bash
conda create -n fastsesr python=3.8
conda activate fastsesr
```

### 2. 安装 PyTorch

根据您的CUDA版本安装PyTorch（推荐 PyTorch 1.10+）：

```bash
# CUDA 11.3
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# 或使用 pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
```

### 3. 安装 PyTorch3D

PyTorch3D 用于 LOON-UNet 的 kNN 操作和 Chamfer 距离计算：

```bash
# 使用 conda (推荐)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

# 或从源码安装
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### 4. 安装其他依赖

```bash
pip install -r requirements.txt
```

---

## 📦 依赖库

项目所需的主要依赖库如下：

| 库名称 | 用途 | 最低版本 |
|--------|------|----------|
| `torch` | 深度学习框架 | 1.10+ |
| `pytorch3d` | 3D操作（kNN, FPS, Chamfer距离） | 0.6+ |
| `open3d` | 点云/网格IO与可视化 | 0.15+ |
| `numpy` | 数值计算 | 1.20+ |
| `tqdm` | 进度条显示 | 4.60+ |
| `timm` | 预训练模型组件（DropPath等） | 0.5+ |
| `wandb` | 训练日志记录 | 0.12+ |

创建 `requirements.txt` 文件：

```txt
torch>=1.10.0
numpy>=1.20.0
open3d>=0.15.0
tqdm>=4.60.0
timm>=0.5.0
wandb>=0.12.0
scipy
```

---

## 📁 数据集准备

### 数据集目录结构

项目期望的数据目录结构如下：

```
Data/
├── ABC/
│   ├── train/                    # ABC 训练集 (PLY 格式)
│   │   ├── 00000001.ply
│   │   ├── 00000002.ply
│   │   └── ...
│   └── test/                     # ABC 测试集
│       ├── 00000501.ply
│       └── ...
├── PointClouds/
│   ├── FAUST/                    # FAUST 数据集点云
│   │   ├── tr_reg_000.ply
│   │   └── ...
│   ├── MGN/                      # MGN 数据集点云
│   └── <其他数据集>/
└── GT_Meshes/
    ├── FAUST/                    # FAUST Ground Truth 网格
    │   ├── tr_reg_000.ply
    │   └── ...
    └── <其他数据集>/
```

### 数据集获取

| 数据集 | 描述 | 下载链接 |
|--------|------|----------|
| **ABC** | CAD模型数据集，用于 S1 训练 | [ABC Dataset](https://deep-geometry.github.io/abc-dataset/) |
| **FAUST** | 人体扫描数据集 | [FAUST Dataset](https://faust.is.tue.mpg.de/) |
| **MGN** | 多服装人体数据集 | [MGN Dataset](https://virtualhumans.mpi-inf.mpg.de/mgn/) |

### 数据预处理

确保所有点云文件为 `.ply` 格式。如需转换，可使用 Open3D：

```python
import open3d as o3d

mesh = o3d.io.read_triangle_mesh("input.obj")
pcd = mesh.sample_points_uniformly(number_of_points=100000)
o3d.io.write_point_cloud("output.ply", pcd)
```

---

## 🎯 Stage 1 训练

Stage 1 使用 ABC 数据集训练基础三角分类网络。

### 训练命令

```bash
python S1_train.py \
    --gpu 0 \
    --max_epoch 301 \
    --use_pair_lowrank 1 \
    --pair_rank 32 \
    --pair_alpha 0.5 \
    --pair_bias 0.0
```

### 参数说明

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--gpu` | 0 | 使用的 GPU 编号 |
| `--max_epoch` | 301 | 最大训练轮数 |
| `--ckpt_path` | None | 恢复训练的检查点路径 |
| `--use_pair_lowrank` | 0 | 是否使用低秩对偶分解 (0/1) |
| `--pair_rank` | 32 | 低秩分解的秩 |
| `--pair_alpha` | 0.5 | pair_alpha 初始值 |
| `--pair_bias` | 0.0 | pair_bias 初始值 |

### 训练输出

训练模型和日志保存在：

```
S1_training/
└── model_k50/
    ├── log_train.txt           # 训练日志
    ├── best_model              # 最佳模型检查点
    └── ckpt_epoch_*.pth        # 各轮次检查点
```

### 从检查点恢复训练

```bash
python S1_train.py \
    --gpu 0 \
    --max_epoch 301 \
    --ckpt_path S1_training/model_k50/ckpt_epoch_100.pth
```

---

## 🚀 Stage 2 训练

Stage 2 使用 LOON-UNet 进行多尺度偏移量学习。

### 数据集划分

#### Step 1: 生成固定划分配置

使用 `generate_fixed_splits.py` 生成可复现的 K-fold 划分：

```bash
python scripts/generate_fixed_splits.py \
    --data_root /path/to/Data \
    --datasets FAUST MGN \
    --split_names Split-A Split-B Split-C \
    --seeds 202401 202402 202403
```

这将在 `splits/<dataset>/` 目录下生成划分配置文件。

#### Step 2: 将 JSON 划分转换为文件列表

```bash
python scripts/convert_json_splits_to_kfold_lists.py --dataset FAUST
```

生成的目录结构：

```
splits/
└── FAUST/
    ├── Split-A/
    │   ├── fold_0.json
    │   └── ...
    ├── fold_Split-A_fold_0/
    │   ├── train_list.txt
    │   ├── val_list.txt
    │   └── test_list.txt
    └── ...
```

### K-Fold 交叉验证训练

```bash
python scripts/kfold_runner.py \
    --dataset FAUST \
    --data_root /path/to/Data \
    --epochs 30 \
    --gpu 0 \
    --splits_root splits \
    --train_script S2/S2_train_loon_unet.py \
    --chunk_size 2000
```

#### 参数说明

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--dataset` | (必需) | 数据集名称 (如 FAUST, MGN) |
| `--data_root` | (必需) | 数据根目录 |
| `--epochs` | 30 | 每个 fold 的训练轮数 |
| `--gpu` | 0 | GPU 编号 |
| `--splits_root` | splits | 划分配置目录 |
| `--chunk_size` | 2000 | 分块大小（降低显存占用） |
| `--use_loon_unet` | False | 重建时使用 LOON-UNet |
| `--resume` | False | 跳过已完成的 fold |

### LOSO (Leave-One-Subject-Out) 训练

适用于需要留一验证的场景：

```bash
python scripts/loso_runner.py \
    --dataset FAUST \
    --data_root /path/to/Data \
    --epochs 30 \
    --gpu 0 \
    --val_ratio 0.2
```

### 直接使用 S2_train_loon_unet.py 训练

对于 ABC 数据集或自定义训练，可直接调用训练脚本：

```bash
python S2/S2_train_loon_unet.py \
    --dataset ABC \
    --data_root /path/to/Data \
    --train_list /path/to/train_list.txt \
    --val_list /path/to/val_list.txt \
    --epochs 30 \
    --gpu 0 \
    --batch_size 1 \
    --lr 0.001 \
    --save_dir runs/ABC_train
```

#### 完整参数列表

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--dataset` | (必需) | 数据集名称 |
| `--data_root` | Data | 数据根目录 |
| `--train_list` | "" | 训练样本列表文件 |
| `--val_list` | "" | 验证样本列表文件 |
| `--test_list` | "" | 测试样本列表文件 |
| `--split_config` | "" | JSON 格式的划分配置文件 |
| `--epochs` | 30 | 训练轮数 |
| `--batch_size` | 1 | 批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--weight_decay` | 0.0 | 权重衰减 |
| `--gpu` | 0 | GPU 编号 |
| `--seed` | 42 | 随机种子 |
| `--delta` | 0.0 | 表面体素大小 |
| `--rescale_delta` | False | 是否根据模型尺度缩放 delta |
| `--unet_k` | 16 | DGCNN 编码器 K 近邻数 |
| `--unet_hidden` | 64 | 瓶颈层隐藏维度 |
| `--unet_T` | 3 | LOON 迭代步数 |
| `--unet_K` | 50 | 三角网络 KNN 数 |
| `--save_dir` | "" | 模型保存目录 |
| `--amp` | False | 启用混合精度训练 |

---

## 💾 预训练模型

预训练的 Stage 1 模型保存在 `trained_models/` 目录：

```
trained_models/
└── model_knn50.pth              # KNN=50 的预训练模型
```

### 模型加载

Stage 2 训练会自动从 `trained_models/model_knn{K}.pth` 加载预训练权重。确保该文件存在：

```python
# 检查预训练模型
import os
assert os.path.exists('trained_models/model_knn50.pth'), "预训练模型不存在!"
```

### 使用自训练的 S1 模型

如果使用自己训练的 S1 模型，S2 会自动查找 `S1_training/model_k{knn}/best_model`：

```bash
# S1 训练完成后，模型位于
S1_training/model_k50/best_model
```

---

## 🔍 重建与评估

### 使用 LOON-UNet 重建

```bash
python S2_reconstruct.py \
    --dataset FAUST \
    --data_root /path/to/Data \
    --use_loon_unet \
    --loon_unet_ckpt runs/kfold/FAUST/fold_0/save/loon_unet_best.pth \
    --gpu 0 \
    --chunk_size 2000 \
    --out_dir results/FAUST
```

### 使用 OffsetOPT 重建 (传统方法)

```bash
python S2_reconstruct.py \
    --dataset ABC \
    --data_root /path/to/Data \
    --gpu 0 \
    --out_dir results/ABC
```

### 评估重建质量

```bash
python main_eval_acc.py \
    --gt_path /path/to/Data/GT_Meshes/FAUST \
    --pred_path results/FAUST \
    --sample_num 100000
```

### 批量评估多个 fold

```bash
python scripts/eval_multi.py \
    --gt_path /path/to/Data/GT_Meshes/FAUST \
    --pred_paths results/FAUST_fold0 results/FAUST_fold1 results/FAUST_fold2 \
    --csv_out results/metrics.csv
```

---

## 📂 项目结构

```
FastSESR/
├── S1/                           # Stage 1 模块
│   ├── BaseNet.py                # S1 基础网络（DGCNN + GNN）
│   ├── loss_supervised.py        # 监督损失函数
│   └── fitModel.py               # 训练工具类
├── S2/                           # Stage 2 模块
│   ├── LoonUNet.py               # LOON-UNet 网络架构
│   ├── ReconNet.py               # 重建网络（继承自 S1）
│   ├── ExtractFace.py            # 三角面片提取
│   ├── offset_opt.py             # 偏移量优化器
│   ├── loss_unsupervised.py      # 无监督损失
│   └── S2_train_loon_unet.py     # S2 训练脚本
├── dataset/                      # 数据集加载器
│   ├── mesh_train.py             # 网格训练数据集
│   ├── pc_recon.py               # 点云重建数据集
│   └── pc_recon_with_gt.py       # 带 GT 的点云数据集
├── scripts/                      # 实用脚本
│   ├── generate_fixed_splits.py  # 生成划分配置
│   ├── convert_json_splits_to_kfold_lists.py  # 转换划分格式
│   ├── kfold_runner.py           # K-fold 训练编排
│   ├── loso_runner.py            # LOSO 训练编排
│   └── eval_multi.py             # 批量评估
├── trained_models/               # 预训练模型
│   └── model_knn50.pth
├── utils/                        # 工具函数
│   └── augmentor.py              # 数据增强
├── eval/                         # 评估工具
├── S1_train.py                   # Stage 1 训练入口
├── S2_reconstruct.py             # 重建入口
├── main_eval_acc.py              # 评估入口
└── README.md
```

---

## 📝 引用

如果您使用了本项目，请引用：

```bibtex
@article{fastsesr2024,
  title={FastSESR: Fast Surface Extraction and Super-Resolution},
  author={Your Name},
  year={2024}
}
```

---

## 📄 License

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
