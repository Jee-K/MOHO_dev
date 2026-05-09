# MOHO_dev

**Point cloud-based deep learning framework for predicting geometric shrinkage and deformation in ceramic tableware during kiln firing.**

---


## Method

The current approach adapts **DGCNN (Dynamic Graph CNN)** — originally designed for point cloud classification and part segmentation — into a **point-wise regression model**. The EdgeConv backbone captures local geometric relationships via k-nearest neighbor graphs at each feature layer, which is well-suited for modeling local deformation patterns.

**Input:** `as-designed` 3D point cloud `(N × 3)`  
**Output:** per-point scalar shrinkage value `(N × 1)`

---

## Repository Structure

```
MOHO_dev/
├── data.py             # Dataset class (MOHOReg) and point cloud loaders
├── model.py            # DGCNN_regression and baseline model definitions
├── main_regression.py  # Training and evaluation entry point
├── util.py             # Loss functions and logging utilities
├── data/
│   ├── train/          # Training .txt files (x y z shrinkage)
│   └── test/           # Test .txt files
└── outputs/            # Experiment logs, checkpoints, visualizations
```

---

## Data Format

Each sample is a `.txt` file with one point per row:

```
x  y  z  shrinkage_value
```

Points represent the as-designed CAD geometry. Ground truth shrinkage values are obtained by aligning post-firing 3D scans against the original CAD model.

---

## Getting Started

### Installation

```bash
git clone https://github.com/Jee-K/MOHO_dev.git
cd MOHO_dev
pip install torch numpy h5py plyfile scikit-learn
```

### Training

```bash
python main_regression.py \
    --exp_name moho_exp1 \
    --model dgcnn_regression \
    --num_points 4096 \
    --epochs 200 \
    --scheduler cos
```

### Evaluation

```bash
python main_regression.py \
    --exp_name moho_exp1 \
    --model dgcnn_regression \
    --eval True \
    --model_path outputs/moho_exp1/models/model.t7
```

---

## Current Status

- [x] DGCNN adapted for per-point scalar regression
- [x] Custom dataset loader for ceramic point cloud pairs
- [x] MSE-based training pipeline with cosine LR scheduling
- [x] Transfer learning support from pretrained DGCNN weights
- [ ] Training instability under investigation
- [ ] Chamfer Distance / EMD evaluation metrics
- [ ] Inference and visualization script
- [ ] Alternative architectures (PointNet++, PCT, etc.)

---

## License

MIT
