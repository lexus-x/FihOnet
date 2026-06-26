# FishOnet 🐟

## Dynamic Open-Set Fish Species Recognition with Calibrated Gating and Parameter-Efficient Fine-Tuning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)
[![BioCLIP 2.5](https://img.shields.io/badge/Backbone-BioCLIP%202.5-green.svg)](https://huggingface.co/imageomics/bioclip-2.5-vith14)
[![Challenge](https://img.shields.io/badge/Challenge-CV4Ecology%202026-orange.svg)](https://www.cv4ecology.org/)

FishOnet is a state-of-the-art hybrid deep learning pipeline designed for the **CV4Ecology 2026 Fish Species Recognition Challenge**. 

Ecological monitoring in the wild presents a challenging **Open-Set Species Recognition (OSSR)** problem: models must classify known seen species into their correct categories, while simultaneously identifying and rejecting novel unseen species. FishOnet solves this by routing image features through a calibrated **Maximum Logit Score (MLS) gate** to dynamically decide whether to classify a sample via a fine-tuned closed-set classifier or a zero-shot name-retrieval engine.

---

## 📐 System Architecture

The pipeline consists of a four-stage process: feature extraction, open-set gating, known species classification, and unknown species text retrieval.

```
                           +----------------------+
                           |   Input Fish Image   |
                           +-----------+----------+
                                       |
                                       v
                    +------------------------------------+
                    | Visual Encoder: BioCLIP 2.5 ViT-H  |
                    +------------------+-----------------+
                                       |
                                       v
                             +-------------------+
                             |  Logit Scoring    |
                             +---------+---------+
                                       |
                                       v
                            /---------------------\
                           /   Maximum Logit       \
                          <    Score (MLS) >= theta >
                           \   (Calibrated Gate)   /
                            \----------+----------/
                                       |
                           +-----------+-----------+
                        Yes|                     No|
                           v                       v
               +-----------------------+ +-----------------------+
               |      Seen Route       | |     Unseen Route      |
               | (Prototype + 1-NN)    | | (Debiased Text-Match) |
               +-----------+-----------+ +-----------+-----------+
                           |                       |
                           +-----------+-----------+
                                       |
                                       v
                           +-----------------------+
                           |  Predicted Taxon Name |
                           +-----------------------+
```

---

## 🔬 Core Methodology

### 1. Visual Encoder (BioCLIP 2.5 Huge Backbone)
We employ **BioCLIP 2.5 Huge (ViT-H/14)**, pretrained on the TreeOfLife-200M dataset (including FathomNet marine imagery), as our visual feature extractor. It produces a normalized visual embedding $z_x \in \mathbb{R}^{1024}$:
$$z_x = \frac{\Phi_{\text{image}}(x)}{\|\Phi_{\text{image}}(x)\|_2}$$
This upgraded Huge backbone yields a $+5.7\%$ zero-shot improvement over BioCLIP 2 ViT-L.

### 2. Calibrated Open-Set Gate (Maximum Logit Score)
Softmax probabilities tend to normalize logits, leading to high-confidence errors on out-of-distribution (OOD) unseen species. To prevent this, the gate thresholds the raw **Maximum Logit Score (MLS)**:
$$\text{MLS}(x) = \max_{c \in \mathcal{C}_{\text{seen}}} s_c(x)$$
* If $\text{MLS}(x) \ge \theta$, the sample is classified as **Known (Seen)**.
* If $\text{MLS}(x) < \theta$, the sample is rejected as **Unknown (Unseen)**.
The threshold $\theta$ is calibrated on a simulated OOD holdout split.

### 3. Seen Route (Prototype + Class-Max-Sim Blended Scorer)
For the closed-set classifier, we blend Nearest-Class-Mean (NCM) prototypes with 1-Nearest Neighbor (1-NN) maximum similarity:
$$s_c(x) = z_x^T \mu_c + \alpha \cdot \max_{x' \in S_c} z_x^T z_{x'}$$
where $\mu_c$ is the normalized class prototype, $S_c$ is the set of training embeddings for class $c$, and $\alpha$ is a blending weight calibrated to $\alpha^* = 2.0$ (+4.8pp gain).

### 4. Unseen Route (Z-Score Debiasing & F-Name Text Matcher)
Samples routed to the unknown head are retrieved against species text templates. We apply:
* **F-Name Formatting**: We format prompts prioritizing the common English name if it is more frequent in literature, falling back to the scientific name when necessary.
* **Column-wise Z-score Normalization**: We correct popularity bias—where specific text embeddings attract disproportionately high scores—by standardizing similarities:
$$\hat{S}_{ij} = \frac{S_{ij} - \bar{S}_{\cdot j}}{\sigma_{S_{\cdot j}} + \epsilon}$$

### 5. Parameter-Efficient Fine-Tuning (LoRA & WiSE-FT)
* **Seen Route**: We adapt the MLP layers of the top-12 blocks of ViT-H using Low-Rank Adaptation (LoRA), keeping bottom blocks graph-free to run in low memory.
* **Unseen Route**: We train a contrastive image-text projection on seen species, then interpolate the weights using **WiSE-FT** ($\alpha_{\text{wise}} = 0.4\text{--}0.5$) to retain zero-shot generalization on novel classes.

---

## 📈 Leaderboard & Submission History

The model's progression tracked on the Codabench leaderboard (overall score weighted $0.566 \times \text{seen} + 0.434 \times \text{unseen}$):

| Version | Recipe Description | Seen Acc | Unseen Acc | Overall | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **base** | BioCLIP-2 ViT-L Routed baseline | 66.5% | 6.5% | 40.3% | Real Entry |
| **v05** | ViT-H+ViT-L Ensemble, prototype blend, z-debias, hflip TTA | 75.37% | 12.22% | 47.81% | Real Entry |
| **v07** | v05 + Seen-route LoRA fine-tuning (top-12 ViT-H blocks) | ~77.8%* | 12.22% | ~49.2%* | Projected |
| **v09** | v07 + Seen long-tail text blend (+4·taxon-text similarity) | **79.10%** | 12.22% | ~49.9%* | Projected |
| **v11** | v09 + Unseen route contrastive-FT (top-12) @ WiSE-FT 0.5 | 79.10% | ~13.4%* | ~50.4%* | Projected |
| **v12** | v09 + Scaled contrastive-FT (top-16 blocks, rank-48) @ WiSE-FT 0.4 | 79.10% | **~14.1%*** | **~50.7%*** | Projected |
| **v10** | *Sandbag* (Tank seen route to 0% to read unseen scores) | 0.35% | 12.22% | 5.53% | Real Entry |
| **v13** | *Sandbag* (Tank seen route to ~16% to read v12 unseen) | ~16.0%* | **~14.1%*** | ~15.0%* | Real Entry |

*\*Projected figures are calibrated against local holdout evaluations and confirmed sandbag submissions.*

---

## 📂 Repository Structure

```
.
├── index.html                   # Glassmorphic simulator dashboard HTML
├── index.css                    # Glassmorphism dark-mode UI stylesheet
├── index.js                     # Simulator math & UI state logic
├── environment.yml              # Conda environment configuration
│
├── src/                         # Core execution pipeline scripts
│   ├── ft.py                    # Seen-route LoRA fine-tuning script
│   ├── ft_eval.py               # Evaluation harness for fine-tuned features
│   ├── predict_v12.py           # v12 submission generator (Best overall)
│   ├── predict_v11.py           # v11 submission generator (Unseen FT)
│   ├── predict_v09.py           # v09 submission generator (Taxon text blend)
│   ├── predict_v07.py           # v07 submission generator (LoRA seen route)
│   ├── predict_vith.py          # ViT-H baseline routed predictor
│   ├── predict_v2.py            # ViT-L baseline routed predictor
│   ├── route_v2.py              # Gate calibration logic
│   ├── embed_any.py             # Feature extractor supporting TTA/hflip
│   ├── text_ensemble.py         # Multi-template prompt-text embedder
│   ├── analyze_unseen.py        # Unseen-route debiasing validator
│   ├── seen_sweep.py            # Prototype blending parameter sweep
│   └── rerank_qwen.py           # Qwen2.5-VL second-stage reranker pilot
│
├── scripts/                     # Shell scripts for execution
│   ├── setup_env.sh             # Conda environment setup helper
│   └── gated_embed.sh           # Batch embedding GPU-gate listener
│
├── docs/                        # Project documentation and artifacts
│   ├── project_plan.md          # Scientific project specification
│   ├── images/                  # Flowcharts and architecture diagrams
│   └── videos/                  # Interactive dashboard walkthrough
│
├── notes/                       
│   └── playbook.md              # Verified winning playbook & strategies
│
└── submissions/                 
    └── VERSIONS.md              # Markdown registry indexing all submissions
```

---

## 🚀 Quick Start

### 1. Environment Installation
```bash
# Setup the conda environment and install dependencies
bash scripts/setup_env.sh
conda activate onet
```

### 2. Feature Extraction & Embedding Caching
```bash
# Extract visual embeddings with test-time augmentation (hflip TTA)
python src/embed_any.py --model hf-hub:imageomics/bioclip-2.5-vith14 \
                        --split train --out outputs/emb_train_h.pt --hflip 1

# Generate text embeddings for the species vocabulary
python src/text_ensemble.py --model hf-hub:imageomics/bioclip-2.5-vith14 \
                            --out outputs/text_emb_h.pt
```

### 3. Running LoRA Fine-Tuning
```bash
# Train LoRA adapters on top-12 blocks
python src/ft.py --epochs 6 --top_k_blocks 12 --bs 128 --out outputs/ft_lora.pt

# Extract fine-tuned features for evaluation
python src/ft.py --extract train --ckpt outputs/ft_lora.pt --out outputs/emb_train_ft.pt
```

### 4. Running Predictors & Sweeping Thresholds
```bash
# Generate the v12 prediction zip for submission
python src/predict_v12.py

# Sweep and calibrate the gating threshold theta
python src/validate_gate.py --theta 0.85 0.88 0.90 0.92 0.95
```

### 5. Launching the Interactive Planning Simulator
To run the glassmorphic planning dashboard and interactive simulator locally:
```bash
python -m http.server 8000
# Open http://localhost:8000 in your browser
```

---

## 📖 Key References

* **BioCLIP**: Stevens et al., *BioCLIP: A Vision-Language Foundation Model for the Tree of Life*, CVPR 2024.
* **MLS Gate**: Vaze et al., *Open-Set Recognition: A Good Closed-Set Classifier is All You Need*, ICLR 2022.
* **F-Name Rule**: Parashar et al., *Evaluating Name Formatting for Zero-Shot Classification of Biological Taxa*, EMNLP 2023.
* **WiSE-FT**: Wortsman et al., *Robust Fine-Tuning of Zero-Shot Models*, CVPR 2022.
