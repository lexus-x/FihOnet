# FishOnet 🐟

> **CV4Ecology 2026 Fish Species Recognition Challenge**  
> An open-set fish species classification framework utilizing BioCLIP 2.5, Calibrated Maximum Logit Score (MLS) Gating, and Description-guided Zero-Shot Retrieval.

---

## 🌟 Overview

FishOnet is an open-set classification pipeline designed to address the challenge of recognizing fish species in natural marine environments. In ecological surveys, systems must classify known species (seen in training data) while detecting and handling unknown, novel species. 

FishOnet addresses this by routing inputs dynamically using a **Maximum Logit Score (MLS)** gate between a fine-tuned **Closed-Set Classifier** (for seen species) and a **Zero-Shot Text Matcher** (for unseen species) powered by **BioCLIP 2.5 Huge** features.

### Key Strategy Components:
1. **Backbone**: Pretrained BioCLIP 2.5 Huge (ViT-H/14) features, specialized in TreeOfLife and marine taxonomy representation.
2. **MLS Gating**: Uses raw logit magnitude rather than normalized softmax probability to detect out-of-distribution (OOD) novel species.
3. **Seen Route**: Proto + α×Class-Max-Sim blended scoring (α=2.0) for ~70% seen accuracy.
4. **Unseen Route**: Column-wise z-score debiased zero-shot name matching for robust retrieval.
5. **Squeeze**: Prompt-template ensemble, backbone fusion (ViT-H + ViT-L), TTA (horizontal flip), and Qwen2.5-VL reranking pilot.

---

## 📊 Project Artifacts

### 1. System Architecture
Our routing architecture processes incoming fish images, extracts feature vectors, runs gating logic, and routes to the appropriate classification head.

![System Architecture](docs/images/system_architecture.png)
*Figure 1: High-level routing flowchart of the FishOnet architecture.*

### 2. MLS Gate Calibration
A visual concept of how the Maximum Logit Score separating threshold $\theta$ isolates known seen species distributions from novel unseen species distributions.

![MLS Calibration](docs/images/mls_calibration.png)
*Figure 2: Probability density of maximum logit scores for known vs. unknown species.*

### 3. Interactive Planning Dashboard & MLS Simulator
We built a premium, glassmorphic project dashboard containing an interactive MLS simulator. You can adjust the threshold $\theta$ slider to see how it affects classification metrics (TPR, FPR, Accuracy) across a set of simulated fish samples.

![Dashboard Walkthrough](docs/videos/dashboard_demo.webp)
*Figure 3: Screencast demonstrating the interactive simulator and timeline checklist.*

---

## 📂 Repository Layout

```
fishonet/
├── docs/
│   ├── images/
│   │   ├── system_architecture.png     # System flow diagram
│   │   └── mls_calibration.png         # Gating score distributions
│   ├── videos/
│   │   └── dashboard_demo.webp         # Interactive dashboard screencast
│   └── project_plan.md                 # Detailed paper-style spec & results
├── src/
│   ├── embed.py                        # BioCLIP ViT-L image embedder
│   ├── embed_any.py                    # Generic open_clip embedder (ViT-H, etc.) + TTA
│   ├── embed_dino.py                   # DINOv2 ViT-L/14 image embedder
│   ├── build_text_embeddings.py        # Single-template text encoder
│   ├── text_any.py                     # Text encoder for any open_clip model
│   ├── text_ensemble.py                # Multi-template prompt ensemble text encoder
│   ├── baseline.py                     # Zero-shot + proto blend baseline predictor
│   ├── predict_v2.py                   # MLS-gated routing predictor (ViT-L)
│   ├── predict_vith.py                 # Best predictor: ViT-H + debiased unseen
│   ├── route_v2.py                     # MLS gating + threshold calibration
│   ├── validate.py                     # Basic validation harness
│   ├── validate_gate.py                # Threshold sweep validation
│   ├── eval_backbone.py                # Backbone comparison (seen + unseen)
│   ├── eval_ensemble.py                # BioCLIP vs DINOv2 blend evaluation
│   ├── ensemble_test.py                # ViT-H + ViT-L score fusion
│   ├── seen_methods.py                 # Seen-route scoring method comparison
│   ├── seen_sweep.py                   # Proto + α×class-max-sim sweep
│   ├── analyze_unseen.py               # Unseen-route debiasing analysis
│   ├── cheap_wins.py                   # Quick improvement experiments
│   ├── rerank_qwen.py                  # Qwen2.5-VL second-stage reranker
│   ├── resolve_taxonomy.py             # GBIF genus resolver
│   ├── resolve_taxonomy_full.py        # Full GBIF taxonomy (kingdom→genus)
│   ├── extract_embeddings.py           # Feature extraction utilities
│   ├── sanity.py                       # Sanity check script
│   └── zeroshot_baseline.py            # Pure zero-shot text matcher
├── scripts/
│   ├── setup_env.sh                    # Conda environment setup
│   └── gated_embed.sh                  # GPU-gated batch embedding pipeline
├── notes/
│   └── playbook.md                     # Verified strategy playbook
├── index.html                          # Interactive dashboard HTML
├── index.css                           # Glassmorphism dark-mode style
├── index.js                            # Simulator logic & tab switcher
├── environment.yml                     # Conda environment spec
└── README.md                           # Project landing page (this file)
```

---

## 🔬 Current Results (Hard Simulation Split)

| Route | Method | Accuracy |
| :--- | :--- | :---: |
| **Seen** | Proto + 2.0×Class-Max-Sim (ViT-H) | **~70%** |
| **Unseen** | Z-score debiased name matching | **Best** |
| **Projected Overall** | Weighted seen×0.566 + unseen×0.434 | Under optimization |

---

## 📑 Detailed Plan
For a full mathematical formulation of the MLS gate, experimental results, literature citations, and validation details, read our **[Detailed Paper-Style Project Plan](docs/project_plan.md)**.

---

## 🚀 Quick Start

### Running the Interactive Dashboard
```bash
python -m http.server 8000
# Navigate to http://localhost:8000
```

### Environment Setup (GPU Server)
```bash
bash scripts/setup_env.sh
conda activate onet
```

### Embedding Pipeline
```bash
# Embed training images with BioCLIP ViT-H
python src/embed_any.py --model hf-hub:imageomics/bioclip-2.5-vith14 \
       --split train --out outputs/emb_train_h.pt --hflip 1

# Precompute text embeddings
python src/text_any.py --model hf-hub:imageomics/bioclip-2.5-vith14 \
       --out outputs/text_emb_h.pt

# Generate predictions
python src/predict_vith.py --alpha 2.0 --debias 1
```

### Evaluation
```bash
# Compare backbones
python src/eval_backbone.py --train outputs/emb_train_h.pt --text outputs/text_emb_h.pt

# Sweep seen-route blending weight
python src/seen_sweep.py

# Analyze unseen debiasing
python src/analyze_unseen.py
```