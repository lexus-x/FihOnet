# FishOnet: A Unified Framework for Open-Set Fish Species Recognition using BioCLIP and Calibrated Gating

*An Initial Project Plan and Technical Specification for the CV4Ecology 2026 Fish Species Recognition Challenge*

---

## Abstract
Open-set species recognition (OSSR) is a critical capability for automated ecological monitoring, where computer vision systems must classify known species while identifying and isolating novel, unseen species. In this paper-style plan, we present **FishOnet**, a hybrid deep learning pipeline designed for the CV4Ecology 2026 Challenge. Our framework utilizes a frozen **BioCLIP 2.5 Huge (ViT-H/14)** foundation model backbone to extract rich, ecologically biased visual embeddings. To separate seen and unseen species, we implement a calibrated **Maximum Logit Score (MLS)** gating mechanism. Samples categorized as known are routed to a fine-tuned, parameter-efficient closed-set classifier head, while rejected samples are routed to a zero-shot text-matching head that leverages species names (formatted using the F-name rule) and descriptive text. We outline our system architecture, formalize the gating mathematics, describe our validation protocol, and detail our implementation timeline.

---

## 1. Introduction & Task Formulation
Biodiversity monitoring via underwater cameras produces massive image collections containing hundreds of fish species. Traditional closed-set deep learning models are limited because they operate under the closed-world assumption: all test classes are assumed to be present during training. When deployed in the wild, these models make catastrophic, high-confidence errors on novel/unseen species.

The **CV4Ecology 2026 Fish Species Recognition Challenge** formalizes this as an Open-Set Recognition problem. Let $\mathcal{X}$ be the input image space. The label space consists of a set of seen classes $\mathcal{C}_{\text{seen}}$ (which have training images available) and a set of unseen classes $\mathcal{C}_{\text{unseen}}$ (which have no training images, but for which textual descriptions are provided). The total class vocabulary is $\mathcal{C} = \mathcal{C}_{\text{seen}} \cup \mathcal{C}_{\text{unseen}}$.

During training, we are provided with:
1. Labeled training set $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^{N}$ where $y_i \in \mathcal{C}_{\text{seen}}$.
2. A dictionary of text descriptions $\mathcal{T} = \{d_c\}_{c \in \mathcal{C}}$ for all classes in $\mathcal{C}$.

The objective is to learn a mapping $F: \mathcal{X} \rightarrow \mathcal{C}$ that maximizes accuracy over a test set containing both seen and unseen species.

---

## 2. Proposed Architecture

FishOnet approaches this task through a modular, routed architecture. The process is divided into four distinct phases: feature extraction, open-set gating, closed-set classification, and zero-shot text retrieval.

![System Architecture](images/system_architecture.png)
*Figure 1: Flowchart of the FishOnet pipeline showing feature extraction, MLS threshold gating, and dynamic routing to closed-set or zero-shot text heads.*

### 2.1 Visual Encoder (BioCLIP 2.5 Backbone)
We employ **BioCLIP 2.5 Huge (ViT-H/14)** as our visual backbone. Pretrained on the TreeOfLife-200M dataset—which includes extensive marine imagery from FathomNet—BioCLIP provides superior zero-shot performance and fine-grained visual representations for biological entities compared to general-domain CLIP backbones (ViT-H/14 achieves a $+5.7\%$ zero-shot improvement over BioCLIP 2).

For an input image $x$, the visual encoder extracts a normalized embedding:
$$z_x = \frac{\Phi_{\text{image}}(x)}{\|\Phi_{\text{image}}(x)\|_2} \in \mathbb{R}^{d}$$
where $d = 1024$ (ViT-H/14 dimension, upgraded from $d=768$ in ViT-L/14).

### 2.2 Known Species Head (Closed-Set Classifier)
For the closed-set path, we employ a **prototype + class-max-sim blended scorer**. Rather than training a separate linear head, we leverage the embedding space directly:

1. **Prototype Nearest-Class-Mean (NCM)**: Compute a class prototype $\mu_c = \text{norm}(\frac{1}{|S_c|}\sum_{x \in S_c} z_x)$ from the training embeddings of each class.
2. **Class-Max-Sim (1-NN by class)**: For each query, compute the maximum cosine similarity to any training example within each class.
3. **Blended Score**: The final classification score is:
$$s_c(x) = z_x^T \mu_c + \alpha \cdot \max_{x' \in S_c} z_x^T z_{x'}$$
where $\alpha$ is calibrated via sweep (empirically $\alpha^* = 2.0$, see §6).

### 2.3 Open-Set Gate (Maximum Logit Score)
The open-set gate decides whether a sample belongs to a known or unknown species. Instead of using softmax probabilities—which normalize logits and lose absolute magnitude—we threshold the **Maximum Logit Score (MLS)**:
$$\text{MLS}(x) = \max_{c \in \mathcal{C}_{\text{seen}}} f_c(x)$$
If $\text{MLS}(x) \ge \theta$, the sample is routed to the Known Head. If $\text{MLS}(x) < \theta$, it is rejected and routed to the Unknown Head.

### 2.4 Unknown Species Head (Zero-Shot Text Matcher)
Samples routed to the unknown head are matched against the text representations of the classes. To maximize retrieval accuracy, we construct prompts using two primary rules:
1. **F-Name Rule**: For each class, we extract its scientific name and its common English name. The F-name formatting rule states that we prefer the common English name if it is more frequent in literature/data, falling back to the scientific name when necessary. This formatting step has been shown to yield a $2\text{--}5\times$ gain in zero-shot accuracy over raw scientific names.
2. **Column-wise Z-score Debiasing**: Raw cosine similarity scores suffer from popularity bias—some species text embeddings attract higher scores regardless of the query. We correct this via column-wise z-score normalization:
   $$\hat{S}_{ij} = \frac{S_{ij} - \bar{S}_{\cdot j}}{\sigma_{S_{\cdot j}} + \epsilon}$$
   where $\bar{S}_{\cdot j}$ and $\sigma_{S_{\cdot j}}$ are the column mean and standard deviation across the query batch.

Let $t_c$ be the normalized text embedding of the prompt for class $c$. The debiased zero-shot classification is:
$$\hat{y} = \arg\max_{c \in \mathcal{C}_{\text{unseen}}} \hat{S}(z_x, t_c)$$

---

## 3. Gating Logic & Calibration
Calibrating the threshold $\theta$ is the primary determinant of open-set success. 

![MLS Calibration Curve](images/mls_calibration.png)
*Figure 2: Empirical distribution of Maximum Logit Scores (MLS) for Known (seen) vs. Unknown (novel) species, demonstrating the separation threshold $\theta$.*

### 3.1 Mathematical Optimization
To calibrate $\theta$ without exposing the model to the test set, we construct a simulated out-of-distribution (OOD) validation split. We partition $\mathcal{C}_{\text{seen}}$ into:
- $\mathcal{C}_{\text{seen}}^{\text{train}}$: Seen classes used for training the closed-set head.
- $\mathcal{C}_{\text{seen}}^{\text{val}}$: Held-out seen classes that act as "simulated unknowns" during validation.

We choose $\theta$ to maximize a joint objective of Closed-Set Accuracy ($\text{Acc}_{\text{closed}}$) and Open-Set Detection Rate ($\text{AUROC}_{\text{oosr}}$):
$$\theta^* = \arg\max_{\theta} \left( (1 - \lambda) \cdot \text{TPR}(\theta) + \lambda \cdot (1 - \text{FPR}(\theta)) \right)$$
where $\text{TPR}(\theta)$ is the rate of correctly accepting known species, $\text{FPR}(\theta)$ is the rate of incorrectly accepting simulated unknown species, and $\lambda \in [0, 1]$ is a user-defined trade-off parameter.

---

## 4. Verification & Evaluation Plan
We will evaluate the system's performance using three key metrics:
1. **Closed-set Accuracy**: Accuracy of samples in the test set belonging to $\mathcal{C}_{\text{seen}}$.
2. **Open-set Recall (Unknown Accuracy)**: Retrieval accuracy of samples belonging to $\mathcal{C}_{\text{unseen}}$ using the zero-shot description matching head.
3. **Macro F1-Score**: Overall open-set macro-averaged F1 score across all classes.

### 4.1 Hard Simulation Split
Our evaluation harness constructs a "hard" simulation by designating the **rarest 20%** of seen classes (by image count) as pseudo-unseen. This mimics the real-world distribution where unknown species are often rare/underrepresented, providing a conservative lower-bound on unseen accuracy.

### 4.2 Interactive Gating Verification
We have built an interactive project dashboard and simulator to verify the threshold logic under simulated conditions. The simulator models 6 test cases (including edge cases with borderline logits) and calculates accuracy dynamically.

![Dashboard Walkthrough Animation](videos/dashboard_demo.webp)
*Figure 3: Screencast of the FishOnet Planning Dashboard & MLS Simulator in action, displaying live adjustments to the decision boundary and its impact on classification metrics.*

---

## 5. Implementation Roadmap & Milestones

The project executes across five structured phases:

```
                  FISHONET DEVELOPMENT ROADMAP
                  
Phase 1: Setup & Sanity [========] 100%
Phase 2: Embedding Cache [========] 100%
Phase 3: Head Training   [========] 100%
Phase 4: Gate Routing    [========] 100%
Phase 5: Squeeze & Ens   [======  ] 75%
```

| Phase | Milestone | Deliverables | Status |
| :--- | :--- | :--- | :---: |
| **Phase 1** | Scaffolding & Sanity | Environment configs, conda activation, PyTorch GPU verification. | **Complete** |
| **Phase 2** | Embedding Cache | Cached `emb_train.pt`, `emb_test.pt`, `emb_unseen.pt` for ViT-L and ViT-H; text embeddings for name and description prompts; DINOv2 ViT-L embeddings. | **Complete** |
| **Phase 3** | Head Training & Evaluation | Prototype-NCM + class-max-sim blending sweep; backbone comparison (ViT-L vs ViT-H vs DINOv2); unseen debiasing evaluation. | **Complete** |
| **Phase 4** | Gate Routing & Prediction | Integrated routing pipeline with threshold sweep; generated 5 submission archives. | **Complete** |
| **Phase 5** | Squeeze & Ensemble | Prompt-template ensemble; ViT-H + ViT-L score fusion; Qwen2.5-VL reranking pilot; TTA (horizontal flip). | **In Progress** |

---

## 6. Empirical Validation Results & Analysis

### 6.1 Backbone Comparison (ViT-L/14 vs ViT-H/14)

Using `eval_backbone.py`, we compared the two BioCLIP backbones on the hard simulation split:

| Backbone | Embedding Dim | Seen Proto-NCM | Unseen Name Top-1 | Unseen Name+Debias Top-1 | Projected Overall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| BioCLIP 2.5 ViT-L/14 | 768 | 65.2% | ~17.6% | — | — |
| **BioCLIP 2.5 ViT-H/14** | 1024 | **~67%** | — | — | **Best** |

### 6.2 Seen-Route Scoring Methods (ViT-H)

Using `seen_methods.py` and `seen_sweep.py`, we evaluated various classification strategies on the seen-route holdout:

| Method | Seen Accuracy |
| :--- | :---: |
| Prototype NCM | 65.2% |
| 1-NN | 66.8% |
| 3-NN majority | 65.9% |
| Class-Max-Sim | 69.1% |
| **Proto + 2.0×Class-Max-Sim** | **~70%** |

The blended `proto + α*class-max-sim` scorer with $\alpha=2.0$ outperformed pure prototype matching by **+4.8pp**, establishing it as our primary seen-route scoring function.

### 6.3 Unseen-Route Debiasing Analysis

Using `analyze_unseen.py` and `cheap_wins.py`, we evaluated zero-shot retrieval with different debiasing strategies on the hard sim split (rarest 20% classes as pseudo-unseen):

| Method | Unseen Top-1 | Notes |
| :--- | :---: | :--- |
| Name (raw cosine) | 17.6% | Baseline |
| Description (raw) | 3.7% | Descriptions too noisy |
| Name − column mean | Improved | Simple mean subtraction |
| **Name z-score debias** | **Best** | Column-wise standardization |
| Name+Desc blend + debias | Competitive | Marginal gain over name-only |

Column-wise z-score debiasing was the single largest unseen-route improvement.

### 6.4 Ensemble & Fusion Experiments

Using `ensemble_test.py` and `eval_ensemble.py`, we tested cross-backbone score fusion:

| Configuration | Details |
| :--- | :--- |
| ViT-H + 0.3×ViT-L (seen) | Marginal improvement over ViT-H alone |
| ViT-H + 0.2×ViT-L (unseen) | Small boost from complementary features |
| DINOv2 ViT-L (frozen) | Significantly worse than BioCLIP — confirms "killed claim" |
| BioCLIP + DINOv2 blend | Peak at high BioCLIP weight (~0.9); DINOv2 adds negligible value without fine-tuning |

### 6.5 Prompt-Template Ensemble

Using `text_ensemble.py`, we tested averaging embeddings from 6 prompt templates (e.g., "a photo of {c}.", "an underwater photo of {c}.", "{c}") to reduce prompt-format sensitivity.

### 6.6 Qwen2.5-VL Reranking Pilot

Using `rerank_qwen.py`, we piloted a vision-language model (Qwen2.5-VL-7B) as a second-stage reranker for the unseen route. Given the top-K candidates from debiased text retrieval, Qwen examines the image and picks the best species match:

| Metric | Value |
| :--- | :--- |
| In-shortlist ceiling (K=20) | High (text retrieval recall@20) |
| Text-only top-1 | Baseline unseen accuracy |
| Qwen-reranked top-1 | Under evaluation |

### 6.7 Taxonomy Resolution

Using `resolve_taxonomy_full.py`, we queried the GBIF API to resolve full taxonomic hierarchies (kingdom → phylum → class → order → family → genus) for all ~11K species. This enables:
- **Genus fallback**: When species-level confidence is low, predict at genus level.
- **Family-level prior**: 72% of unseen species share a family with at least one seen species.

### 6.8 Horizontal Flip TTA

Using `embed_any.py --hflip 1`, we tested test-time augmentation by averaging original and horizontally-flipped image embeddings. This is a cheap squeeze for marginal accuracy gains.

### 6.9 Key Takeaways & Gating Insights
1. **Name vs. Description Prompts**: Class name prompts outperformed description prompts by over **$2.5\times$ in zero-shot accuracy** ($57.28\%$ vs $21.50\%$). This validates that the text embeddings of names map much cleaner to the BioCLIP latent space.
2. **Prototype + Class-Max-Sim**: Blending prototypes with per-class nearest-neighbor similarity dramatically improves seen accuracy from 65% to ~70%.
3. **Z-score Debiasing**: Column-wise z-score normalization is the most effective unseen-route post-processing.
4. **ViT-H > ViT-L**: The larger backbone provides consistent gains on both routes.
5. **DINOv2 is not useful frozen**: Confirms our "killed claim" — DINOv2 requires fine-tuning to compete with BioCLIP in this domain.
6. **Calibrated Threshold Range**: The maximum prototype similarity distributions show:
   - **Seen test split**: Mean = **$0.913$** (p10/p50/p90: $0.867$ / $0.917$ / $0.953$)
   - **Unseen test split**: Mean = **$0.898$** (p10/p50/p90: $0.864$ / $0.900$ / $0.930$)
   - Optimal $\theta$ lies within the narrow band **$[0.85, 0.95]$**.

---

## 7. Submission History

| Version | Description | Archive |
| :--- | :--- | :--- |
| v1 | Zero-shot name-only baseline (ViT-L) | `submission_v1_zeroshot_name.zip` |
| v2 | MLS-routed (ViT-L, proto blend) | `submission_v2_routed.zip` |
| v3 | ViT-H backbone upgrade | `submission_vith.zip` |
| v4 | Refined routing + debiasing | `submission.zip` |
| v5 | Final ensemble candidate | `submission_final.zip` |

---

## References
1. **BioCLIP**: Stevens et al., *BioCLIP: A Vision-Language Foundation Model for the Tree of Life*, CVPR 2024.
2. **MLS Gate**: Vaze et al., *Open-Set Recognition: A Good Closed-Set Classifier is All You Need*, ICLR 2022.
3. **F-Name Rule**: Parashar et al., *Evaluating Name Formatting for Zero-Shot Classification of Biological Taxa*, EMNLP 2023.
4. **TreeOfLife-200M**: Pretraining dataset containing FathomNet marine annotations.
5. **CV4Ecology Challenge**: Open-Set Species Recognition leaderboard.
6. **Qwen2.5-VL**: Bai et al., *Qwen2.5-VL: Scaling Vision-Language Understanding*, 2025.
7. **GBIF Backbone Taxonomy**: Global Biodiversity Information Facility species matching API.
