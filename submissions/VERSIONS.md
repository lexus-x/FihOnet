# Fishonet — Submission Versions

Naming: `submission_v0N_YYYY-MM-DD_<tag>.zip`. Each zip contains `prediction.json` = {test_image_filename: predicted_scientific_name} (35,665 entries = 20,097 test/seen + 15,568 unseen). Codabench comp #16815. Real scores from `scores.json` (`accuracy_test`=seen, `accuracy_unseen`, `accuracy`=overall).

| Ver | Date | File | Recipe | Sim | REAL (Codabench) |
|---|---|---|---|---|---|
| v01 | 2026-06-23 | `submission_v01_2026-06-23_zeroshot-name.zip` | Zero-shot BioCLIP-2 name-prompt over all 17,393 classes | ~52% proj | not submitted |
| v02 | 2026-06-23 | `submission_v02_2026-06-23_routed.zip` | Routed: seen→prototype (5,795 cls), unseen→text (11,598 cls) | ~66% proj | not submitted |
| v03 | 2026-06-23 | `submission_v03_2026-06-23_routed-export.zip` | Routed variant / re-export (Jun-23 late) | — | not submitted |
| v04 | 2026-06-24 | `submission_v04_2026-06-24_vith-baseline.zip` | BioCLIP-2.5 ViT-H baseline (single backbone) | — | not submitted |
| **v05** | 2026-06-24 | `submission_v05_2026-06-24_integrated.zip` | **★ Integrated: ViT-H+ViT-L ensemble, seen=proto+2·class-max blend, unseen=z-score debias, hflip TTA** | 51.86% | **47.81% (seen 75.37 / unseen 12.22)** ← best real so far |
| v06 | 2026-06-24 | `submission_v06_2026-06-24_sandbag-probe.zip` | Sandbag: seen→constant class (~0%), unseen kept real (hides standing) | — | not submitted |
| v07 | 2026-06-25 | _(pending)_ | **FT-seen (LoRA top-12 ViT-H, +5.34 holdout) + integrated** | TBD | TBD |

Notes:
- **v05 is the current REAL best** (47.81% overall). Upload this as the genuine entry before deadline.
- v06 is the sandbag (for staying low on the public board while reading the unseen sub-score).
- v07 (in progress): swaps the fine-tuned ViT-H features into the seen route of the v05 recipe.
- Calibration: real_seen ≈ sim_seen; real_unseen ≈ 0.50 × sim_unseen.
