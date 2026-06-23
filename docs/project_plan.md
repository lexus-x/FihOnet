# FishOnet: A Unified Framework for Open-Set Fish Species Recognition using BioCLIP and Calibrated Gating

*An Initial Project Plan and Technical Specification for the CV4Ecology 2026 Fish Species Recognition Challenge*

---

## Abstract
Open-set species recognition (OSSR) is a critical capability for automated ecological monitoring, where computer vision systems must classify known species while identifying and isolating novel, unseen species. In this paper-style plan, we present **FishOnet**, a hybrid deep learning pipeline designed for the CV4Ecology 2026 Challenge. Our framework utilizes a frozen **BioCLIP 2.5 Huge (ViT-H/14)** foundation model backbone to extract rich, ecologically biased visual embeddings. To separate seen and unseen species, we implement a calibrated **Maximum Logit Score (MLS)** gating mechanism. Samples categorized as known are routed to a fine-tuned, parameter-efficient closed-set classifier head, while rejected samples are routed to a zero-shot text-matching head that leverages species names (formatted using the F-name rule) and descriptive text. We outline our system architecture, formalize the gating mathematics, describe our validation protocol, and detail our 8-week implementation timeline.

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
where $d = 768$ is the latent dimension.

### 2.2 Known Species Head (Closed-Set Classifier)
For the closed-set path, we train a lightweight linear probe or fine-tune a parameter-efficient adapter (e.g., LoRA) on top of the BioCLIP embeddings. This head is optimized using a cross-entropy loss over $\mathcal{C}_{\text{seen}}$:
$$\mathcal{L}_{\text{ce}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{e^{w_{y_i}^T z_{x_i}}}{\sum_{j \in \mathcal{C}_{\text{seen}}} e^{w_j^T z_{x_i}}}$$
The closed-set classifier outputs a set of logits $f_c(x) = w_c^T z_x$ for each seen class $c \in \mathcal{C}_{\text{seen}}$.

### 2.3 Open-Set Gate (Maximum Logit Score)
The open-set gate decides whether a sample belongs to a known or unknown species. Instead of using softmax probabilities—which normalize logits and lose absolute magnitude—we threshold the **Maximum Logit Score (MLS)**:
$$\text{MLS}(x) = \max_{c \in \mathcal{C}_{\text{seen}}} f_c(x)$$
If $\text{MLS}(x) \ge \theta$, the sample is routed to the Known Head. If $\text{MLS}(x) < \theta$, it is rejected and routed to the Unknown Head.

### 2.4 Unknown Species Head (Zero-Shot Text Matcher)
Samples routed to the unknown head are matched against the text representations of the classes. To maximize retrieval accuracy, we construct prompts using two primary rules:
1. **F-Name Rule**: For each class, we extract its scientific name and its common English name. The F-name formatting rule states that we prefer the common English name if it is more frequent in literature/data, falling back to the scientific name when necessary. This formatting step has been shown to yield a $2\text{--}5\times$ gain in zero-shot accuracy over raw scientific names.
2. **Context Enrichment**: We append the provided textual description from `descriptions.json` to enrich the context:
   $$\text{prompt}_c = \text{"a photo of } \text{Name}_c \text{, a species of fish described as: } d_c\text{"}$$
   
Let $t_c$ be the normalized text embedding of $\text{prompt}_c$ generated by BioCLIP's text encoder. The zero-shot classification for an unknown sample $x$ is:
$$\hat{y} = \arg\max_{c \in \mathcal{C}} (z_x^T t_c)$$

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

### 4.1 Interactive Gating Verification
We have built an interactive project dashboard and simulator to verify the threshold logic under simulated conditions. The simulator models 6 test cases (including edge cases with borderline logits) and calculates accuracy dynamically.

![Dashboard Walkthrough Animation](videos/dashboard_demo.webp)
*Figure 3: Screencast of the FishOnet Planning Dashboard & MLS Simulator in action, displaying live adjustments to the decision boundary and its impact on classification metrics.*

---

## 5. Implementation Roadmap & Milestones

The project will execute across five structured phases over an 8-week timeline:

```
                  FISHONET DEVELOPMENT ROADMAP
                  
Phase 1: Setup & Sanity [========] 100% (Week 1 - Current)
Phase 2: Embedding Cache [====    ] 50%  (Week 2)
Phase 3: Head Training   [        ] 0%   (Week 3-4)
Phase 4: Gate Routing    [        ] 0%   (Week 5)
Phase 5: Squeeze & Ens   [        ] 0%   (Week 6-8)
```

| Phase | Milestone | Focus Area | Deliverables | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Phase 1** | Scaffolding & Sanity | Environment setup and hardware verification | Environment configs, conda activation validation, PyTorch GPU verification. | **Complete** |
| **Phase 2** | Embedding Cache | Precomputing representation tensors | Cache image embeddings (`emb_train.pt`, `emb_test.pt`, `emb_unseen.pt`) and pre-encode text descriptions. | **In Progress** |
| **Phase 3** | Head Training | Closed-set training & OOSR simulation | Fine-tune the seen classifier, structure the simulated OOD val split, generate logit distributions. | **Planned** |
| **Phase 4** | Gate Routing | Integrated routing pipeline | Combine MLS gate with dual classification heads, finalize decision boundary routing. | **Planned** |
| **Phase 5** | Squeeze & Ens | Ensembling & Test-Time Augmentation | Apply multi-crop TTA, blend fine-tuned and zero-shot logits, evaluate genus fallback logic. | **Planned** |

---

## References
1. **BioCLIP**: Stevens et al., *BioCLIP: A Vision-Language Foundation Model for the Tree of Life*, CVPR 2024.
2. **MLS Gate**: Vaze et al., *Open-Set Recognition: A Good Closed-Set Classifier is All You Need*, ICLR 2022.
3. **F-Name Rule**: Parashar et al., *Evaluating Name Formatting for Zero-Shot Classification of Biological Taxa*, EMNLP 2023.
4. **TreeOfLife-200M**: Pretraining dataset containing FathomNet marine annotations.
5. **CV4Ecology Challenge**: Open-Set Species Recognition leaderboard.
