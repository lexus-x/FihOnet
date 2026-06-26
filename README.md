# FishOnet

Open-set fish species classification for the CV4Ecology 2026 challenge. Routes between a closed-set classifier (known species) and a zero-shot matcher (unknown species) using a calibrated MLS gate.

## Approach

The core problem: classify fish species from images, but some test species weren't in training data. Standard classifiers choke on unseen classes.

Solution pipeline:
1. **BioCLIP 2.5 Huge** features — pretrained on TreeOfLife data, good marine representations
2. **MLS gate** — Maximum Logit Score separates known vs unknown distributions better than softmax probability
3. **Known route** — Proto + class-max-sim blend (α=2.0) for ~70% seen accuracy
4. **Unknown route** — Column-wise z-score debiased zero-shot name matching
5. **Squeeze** — Prompt ensemble, backbone fusion (ViT-H + ViT-L), horizontal flip TTA

The gate threshold θ is calibrated on a held-out split to balance TPR/FPR.

## Results

On the challenge eval set, the routing architecture handles both seen and unseen species without catastrophic failure on the unknowns.

## Structure

```
index.html / .css / .js   # Interactive planning dashboard + MLS simulator
environment.yml           # Conda env (onet)

src/                      # Pipeline — run as `python src/<file>.py`
├── embed.py, embed_any.py, embed_dino.py            # Image encoders
├── build_text_embeddings.py, text_*.py              # Text encoders
├── baseline.py, predict_v2.py, predict_vith.py      # Predictors
├── predict_v07.py                                   # Submission generator (fine-tuned seen route)
├── ft.py, ft_eval.py                                # LoRA fine-tune of seen route + calibrated eval
├── route_v2.py                                       # Gate calibration
├── validate.py, validate_gate.py                    # Evaluation
├── eval_backbone.py, eval_ensemble.py               # Backbone comparisons
├── seen_methods.py, seen_sweep.py                   # Ablations
└── analyze_unseen.py, rerank_qwen.py                # Error analysis / experiments

scripts/                  # setup_env.sh, gated_embed.sh   (GPU-box helpers)
docs/                     # project_plan.md, images/ (diagrams), videos/ (demo)
notes/                    # playbook.md
competition/              # official Codabench files: starting_kit, public_data, class_image_counts.csv   (gitignored)
submissions/              # versioned prediction zips + VERSIONS.md index + results/   (zips gitignored)
onet_backup/              # local snapshot mirror of the GPU box   (gitignored)
```

## Quick start

```bash
# Extract features
python src/embed_any.py --model ViT-H-14

# Run gated prediction
python src/predict_vith.py

# Sweep gate threshold
python src/validate_gate.py --theta 0.3 0.4 0.5 0.6
```

## Interactive dashboard

There's a glassmorphic planning dashboard with an MLS simulator — adjust the θ slider to see how it affects metrics in real time. See `docs/videos/dashboard_demo.webp`.
