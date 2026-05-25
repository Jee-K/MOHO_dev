# MOHO_dev

**Point cloud-based deep learning framework for predicting geometric shrinkage and deformation in ceramic tableware during kiln firing.**

---

## Method

The framework predicts per-point shrinkage directly from the as-designed
geometry. It supports three interchangeable point-cloud regression backbones,
selectable at runtime via `--model`:

| `--model`            | Architecture        | Key idea |
|----------------------|---------------------|----------|
| `dgcnn_regression`   | DGCNN (EdgeConv)     | k-NN graph features at each layer; original baseline |
| `pointnet2`          | PointNet++ (SSG)     | hierarchical FPS + ball-query encoder/decoder |
| `pointtransformer`   | Point Transformer    | local self-attention over k-NN with positional encoding |

`pointnet2` and `pointtransformer` normalize each piece to a unit sphere for
scale-consistent neighborhoods, then re-inject the per-sample scale into the
prediction head so the model can scale shrinkage with absolute piece size.

**Input:** `as-designed` 3D point cloud `(N × 3)`
**Output:** per-point scalar shrinkage value `(N × 1)`

---

## Repository Structure

```
MOHO_dev/
├── data.py             # Dataset class (MOHOReg) and point cloud loaders
├── model.py            # DGCNN / PointNet++ / Point Transformer regression models
├── main_regression.py  # Training and evaluation entry point
├── run_inference.py    # Batch inference -> per-point prediction .txt files
├── vis.py              # Threshold-based displacement visualization
├── util.py             # Loss functions and logging utilities
├── environment.yml     # Conda environment definition
├── prepare_data/       # Color tables + (unused) S3DIS/ScanNet prep scripts
├── pretrained/         # Pretrained DGCNN weights (for transfer learning)
└── outputs/            # Experiment logs, checkpoints, visualizations
```

---

## Data Format

Each sample is a `.txt` file with one point per row:

```
x  y  z  shrinkage_value
```

Points represent the as-designed CAD geometry. Ground-truth shrinkage values are
obtained by aligning post-firing 3D scans against the original CAD model
(done outside this repo).

The loader reads files from a fixed root (set in `data.py`):

```
/data/DGCNN_datatest/
├── trainval/   # training .txt files
└── test/       # test .txt files
```

Edit `DATA_DIR` in `data.py` to point at your own data location.

---

## Getting Started

### Installation

```bash
git clone https://github.com/Jee-K/MOHO_dev.git
cd MOHO_dev
conda env create -f environment.yml
conda activate moho
```

The default `environment.yml` targets an NVIDIA GPU; see the comments in that
file to switch CUDA versions or build CPU-only.

### Training

```bash
python main_regression.py \
    --exp_name moho_exp1 \
    --model pointnet2 \        # dgcnn_regression | pointnet2 | pointtransformer
    --num_points 4096 \
    --epochs 200 \
    --scheduler cos
```

### Evaluation

```bash
python main_regression.py \
    --exp_name moho_exp1 \
    --model pointnet2 \
    --eval True \
    --model_path outputs/moho_exp1/models/model.t7
```

### Inference

```bash
python run_inference.py \
    --model_path outputs/moho_exp1/models/model.t7 \
    --num_points 4096
```

---

## Current Status

- [x] DGCNN adapted for per-point scalar regression
- [x] PointNet++ and Point Transformer regression backbones (`--model`)
- [x] Per-sample scale feature for size-aware prediction
- [x] Fixed shrinkage-target truncation bug (int64 → float32) in data loader
- [x] Custom dataset loader for ceramic point cloud pairs
- [x] MSE-based training pipeline with cosine LR scheduling
- [x] Transfer learning support from pretrained DGCNN weights
- [x] Inference and visualization scripts
- [ ] Chamfer Distance / EMD evaluation metrics

---

## License

MIT
