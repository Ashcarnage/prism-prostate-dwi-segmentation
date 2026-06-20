# PRISM — Prostate Reinforcement Image Segmentation Model

A reinforcement learning framework that automatically discovers efficient 3D U-Net architectures for prostate cancer segmentation from Diffusion-Weighted Imaging (DWI) MRI. The entire architecture search runs on a single GPU in under 2 days.

---

## What this does

Standard Neural Architecture Search for 3D medical images is prohibitively expensive — millions of candidate networks, each requiring hours of training. PRISM solves this with three key ideas:

| Innovation | Benefit |
|---|---|
| **Macro search space** | High-level decisions (patch size, dilation, pooling) instead of per-layer micro-ops → 172,800 vs 10^18 combinations |
| **Parameter sharing** | Shared weight bank across all child networks — no training from scratch → 8× speedup |
| **Element-wise skip connections** | Sum instead of concatenation in U-Net skip paths → 50% memory saving with comparable accuracy |

An LSTM controller generates architecture decisions, trains a candidate network using shared weights, evaluates its Dice score, and updates via REINFORCE. After 150–300 episodes the best-found architecture is used for full inference.

---

## Results

| Metric | Value |
|---|---|
| Average Dice (31 test volumes) | **0.6379** |
| Best single-volume Dice | 0.9150 |
| Model parameters | 5.0M |
| Inference time | 7.8ms / volume |
| Total search time | ~33.6 hours on RTX 5090 |

**Discovered architecture**: 64×64×32 patches · LeakyReLU · stride-2 pooling in stages 3–4 · dilation rate 1 · selective skip pattern

---

## Architecture overview

```
Prostate DWI MRI Volumes (.nii.gz / .nrrd)
         │
         ▼
  ┌─────────────────┐
  │  Preprocessor   │  Z-score normalize, binarize labels, save .pt
  └────────┬────────┘
           │  31 preprocessed samples (preprocessed_dwi_data/)
           ▼
  ┌──────────────────────────────────────────────────────┐
  │                PRISM Search Loop (150 episodes)      │
  │                                                      │
  │  ┌──────────────┐   9 decisions   ┌───────────────┐  │
  │  │ LSTM         │ ──────────────► │ ChildNetwork  │  │
  │  │ Controller   │                 │ (3D U-Net)    │  │
  │  │ (REINFORCE)  │ ◄── Dice score ─│               │  │
  │  └──────────────┘                 └───────┬───────┘  │
  │                                           │          │
  │                          ┌────────────────▼────────┐ │
  │                          │ Parameter Sharing Bank  │ │
  │                          │ (EMA weight updates)    │ │
  │                          └─────────────────────────┘ │
  └──────────────────────────────────────────────────────┘
           │
           │  best_architecture.json + best_prism_model.pt
           ▼
  ┌─────────────────┐
  │  Full Inference │  Export predictions as .nrrd for 3D Slicer
  └─────────────────┘
```

### Controller decisions (search space)

| Decision | Options | Notes |
|---|---|---|
| Patch size HW | 5 | `max(H,W)/S⁴ - S⁴×{0,1,2,3,4}` |
| Patch size Depth | 5 | `D/S⁴ - S⁴×{0,1,2,3,4}` |
| Pooling stride stage 3 | 2 | stride 1 or 2 |
| Pooling stride stage 4 | 2 | stride 1 or 2 |
| Dilation rate stages 2–4 | 3 each | rates 1, 2, 3 |
| Activation function | 3 | ReLU, LeakyReLU, ELU |
| Skip connections | 64 | 2⁶ binary combinations |

**Total search space: ~172,800 architectures**

### Child network architecture

```
Input [1, D, H, W]
    │
    ├── Encoder Stage 1 →  32 ch │ MaxPool (stride 2)
    ├── Encoder Stage 2 →  64 ch │ MaxPool (stride 2)
    ├── Encoder Stage 3 → 128 ch │ MaxPool (configurable stride)
    └── Encoder Stage 4 → 256 ch (bottleneck)
         │
    ┌────┘
    ├── Decoder Stage 1: ConvTranspose3d + element-wise sum (skip) + 2×Conv3d
    ├── Decoder Stage 2: ConvTranspose3d + element-wise sum (skip) + 2×Conv3d
    └── Decoder Stage 3: ConvTranspose3d + element-wise sum (skip) + 2×Conv3d → 1×1 Conv → Output

Deep supervision: intermediate 1×1 Conv heads during training (weight 0.4)
Loss: Dice loss on main output + 0.4 × Dice on deep supervision outputs
```

---

## File structure

```
prism-prostate-dwi-segmentation/
├── prism.py                         # Core: all model classes and trainer
│   ├── PRISMConfig                  # Hyperparameter configuration
│   ├── LSTMController               # RL controller (2-layer LSTM, REINFORCE)
│   ├── ParameterSharingManager      # Shared weight bank with EMA updates
│   ├── SearchSpace                  # Action space encoder / decoder
│   ├── ChildNetwork                 # Dynamic 3D U-Net builder
│   ├── EfficientSkipConnection      # Element-wise sum skip connections
│   ├── DiceLoss                     # Dice loss for segmentation
│   └── PRISMTrainer                 # Main search loop
│
├── train_prism.py                   # Training entry point
├── preprocess_dwi_data.py           # DWI → normalized .pt files
├── test_prism.py                    # Inference + .nrrd export
├── run_prism.py                     # Quick-start interactive CLI
├── search_space.py                  # Standalone search space utilities
├── visualize_training_results.py    # Generate interactive HTML dashboards
├── plot_training_metrics.py         # Generate static training plots
│
├── prism_config.yaml                # YAML configuration
├── requirements_prism.txt           # Python dependencies
├── dwi_dataset_pairs.json           # Image-label pair index
│
├── training_plots/                  # Static matplotlib PNGs
│   ├── reward_progression.png
│   ├── entropy_analysis.png
│   ├── controller_losses.png
│   └── training_dashboard.png
│
├── visualization_results/           # Interactive HTML dashboards
│   ├── dashboard.html
│   ├── training_progress.html
│   ├── convergence_analysis.html
│   └── architecture_analysis.html
│
└── RESULTS_SUMMARY.md               # Training results and performance summary
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements_prism.txt
```

Key dependencies: `torch`, `nibabel`, `pynrrd`, `numpy`, `wandb`, `tqdm`, `matplotlib`

### 2. Preprocess data

Place your prostate DWI MRI volumes (`.nii.gz` images + `.nrrd` segmentation labels) in `Extracted_DWI/`, then:

```bash
python preprocess_dwi_data.py
```

Reads image-label pairs from `dwi_dataset_pairs.json`, applies Z-score normalization, and writes `preprocessed_dwi_data/dwi_sample_XXXX.pt`.

### 3. Run architecture search

```bash
# Full search (150 episodes, ~33 hours on RTX 5090)
python train_prism.py --episodes 150 --batch_size 2

# Quick test (5 episodes)
python train_prism.py --quick_test

# Interactive launcher
python run_prism.py
```

Results are saved under `experiments/<exp_name>/`.

### 4. Evaluate best architecture

```bash
python test_prism.py \
    --model_path experiments/prism_search/best_prism_model.pt \
    --preprocessed_dir preprocessed_dwi_data \
    --output_dir segmentation_results
```

Opens both `prediction_*.nrrd` and `groundtruth_*.nrrd` in 3D Slicer for side-by-side comparison.

### 5. Custom search configuration

```python
from prism import PRISMConfig, PRISMTrainer

config = PRISMConfig()
config.episodes = 100
config.child_networks_per_episode = 15
config.child_lr = 0.0005

trainer = PRISMTrainer(config, device)
results = trainer.search(train_loader, val_loader)
```

---

## Training dynamics

The search proceeds in three phases:

| Phase | Episodes | Reward | Behavior |
|---|---|---|---|
| Learning | 1–50 | 0.02 → 0.48 | High entropy; controller learns basic patterns |
| Improvement | 50–150 | 0.48 → 0.52 | Entropy decreasing; transitioning to exploitation |
| Convergence | 150–300 | ~0.54 plateau | Stable low entropy; fine-tuning best regions |

Training and reward curves are logged via Weights & Biases (`wandb`) and exported as interactive HTML dashboards.

---

## Dataset

- **Modality**: Diffusion-Weighted Imaging (DWI) MRI
- **Task**: Prostate cancer lesion segmentation
- **Volumes**: 31 patients
- **Format**: Mixed `.nii.gz` (images) and `.nrrd` (segmentation labels)
- **Dimensions**: Variable — 168×100×20 to 224×224×35 (highly anisotropic)
- **Split**: 80% training (25 volumes), 20% validation (6 volumes)
- **Labels**: Binary — prostate cancer lesion vs background

---

## Key configuration defaults

| Parameter | Value |
|---|---|
| Episodes | 150 |
| Child networks per episode | 20 |
| Child epochs per episode | 3 |
| Controller LSTM size | 100 |
| Controller LSTM layers | 2 |
| Controller LR | 0.001 |
| Child LR | 0.001 |
| Encoder base channels | 32 |
| Number of encoder stages | 4 |
| Mixed precision (AMP) | Yes |
| Parameter sharing | Yes |
| Element-wise skip connections | Yes |

---

## Visualizing results

Open the HTML files in any browser — no server needed:

```
visualization_results/dashboard.html            # Full training dashboard
visualization_results/training_progress.html    # Episode-by-episode rewards
visualization_results/convergence_analysis.html # Entropy + loss curves
visualization_results/architecture_analysis.html # Architecture frequency
```

Or view static PNG plots in `training_plots/`.

---

## License

MIT
