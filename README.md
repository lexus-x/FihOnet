# FishOnet 🐟

## Open-Set Fish Species Recognition — CV4Ecology 2026

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)
[![Challenge](https://img.shields.io/badge/Challenge-CV4Ecology%202026-orange.svg)](https://www.cv4ecology.org/)

FishOnet is our entry for the **CV4Ecology 2026 Fish Species Recognition Challenge** — an
**open-set recognition** problem in which a model must classify known fish species while
also recognizing *novel* species that were never seen during training.

This repository hosts the project's **interactive dashboard**: a small, dependency-free web
app that explains the open-set recognition problem and lets you experiment with a
Maximum-Logit-Score (MLS) gating simulator.

> ℹ️ **Note:** The training pipeline, model weights, evaluation harness, and methodology
> are maintained in a private working tree and are intentionally **not** part of this public
> repository while the challenge is ongoing.

---

## 🖥️ The Interactive Dashboard

The dashboard has three sections:

| Section | What it shows |
| :--- | :--- |
| **Overview** | The open-set challenge and the general known/unknown routing idea. |
| **MLS Simulator** | An interactive threshold slider that demonstrates how a Maximum-Logit-Score gate separates known and unknown samples. |
| **Concept** | A plain-language walkthrough of the four conceptual stages of an open-set recognition pipeline. |

### Run it locally

```bash
# From the repository root
python -m http.server 8000
# then open http://localhost:8000
```

Or simply open `index.html` directly in a browser.

---

## 📂 Repository Structure

```text
.
├── index.html        # Dashboard markup
├── index.css         # Glassmorphism dark-mode stylesheet
├── index.js          # Simulator math & UI state logic
├── environment.yml   # Reference Python environment
└── README.md
```

---

## 🏷️ About

**RAISE Lab** · 책임 인공지능 연구실 · Responsible AI Lab
Advisor: **Prof. Cheng-Yaw Low**
Department of AI Convergence Engineering, Changwon National University, Republic of Korea 🇰🇷

---

## 📖 References

* **CV4Ecology 2026** — [cv4ecology.org](https://www.cv4ecology.org/)
* **Open-Set Recognition** — Vaze et al., *Open-Set Recognition: A Good Closed-Set Classifier is All You Need*, ICLR 2022.
