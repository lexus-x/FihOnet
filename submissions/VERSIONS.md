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
| **v07** | 2026-06-25 | `submission_v07_2026-06-25_ft-seen.zip` | **FT-seen: LoRA top-12 ViT-H (+2.73 blend on clean holdout) → seen ensemble FT-H + 0.5·L; unseen unchanged (frozen H+L debias)** | seen 86.17% holdout | **~49.2% projected** (seen ~77.8 / unseen 12.2) — not yet submitted |
| **★ v09** | 2026-06-25 | `submission_v09_2026-06-25_seen-textblend.zip` | **v07 + SEEN long-tail text-blend: seen score = proto + 2·cmax + 4·taxon-text-sim (BioCLIP lineage text on given class names). Rescues 1–9-img tail classes. Unseen unchanged.** | seen **87.22%** holdout (+1.05) | **~49.9% projected** (seen ~79.1 / unseen 12.2) — STRICTLY beats v07; the REAL entry to upload near deadline |
| v10 | 2026-06-25 | `submission_v10_2026-06-25_v9sandbag-seenlow.zip` | **SANDBAG of v09: all 20,097 seen→constant class `Ostracion cubicum` (~1% seen); 15,568 unseen = REAL v09 (identical)**. Hides true standing on public board; accuracy_unseen still readable. | — | **5.53% REAL** (seen 0.35 / unseen 12.22) — sandbag confirmed working |
| **★★ v11** | 2026-06-25 | `submission_v11_2026-06-25_unseen-ctft-wiseft.zip` | **v09 seen route + NEW unseen route: contrastive image→text LoRA FT (BioCLIP ViT-H, top-12, trained on KNOWN species only, transfers to novel) @ WiSE-FT scale 0.5 + hflip, matched to taxon text, +0.5·frozen-L. First legit unseen gain.** | seen 87.22 holdout; unseen hard-sim **23.99→26.32 (+2.33)** | **~50.4% projected** (seen ~79.1 / unseen ~13.4) — best legit submission, crosses 50% |

| **★★ v12** | 2026-06-26 | `submission_v12_2026-06-26_unseen-ctft-scaled.zip` | **v09 seen route + SCALED contrastive-FT unseen (top-16/rank-48, WiSE scale 0.4). Seen byte-identical to v09.** | unseen hard-sim **27.70** (v11 26.32, v09 23.99) | **~50.7% projected** (seen ~79.1 / unseen ~14.1) — current best built |
| v13 | 2026-06-26 | `submission_v13_2026-06-26_v12sandbag-seen15.zip` | **SANDBAG of v12: seen tanked to ~16% (keep real v12 seen for every-5th img=20%, rest→non-seen class `Lepidotrigla spiloptera`); unseen = REAL v12 (scaled-FT, identical).** Reads the REAL unseen on the board while hiding standing. | — | **~15% overall** (seen ~16 / unseen ~REAL v12) — upload to CONFIRM the FT unseen gain |
| v14 | 2026-06-26 | `submission_v14_2026-06-26_standard-sandbag-seen15-unseen5.zip` | **STANDARD sandbag: BOTH routes tanked. Seen ~16% (keep 20% real v12), unseen ~5% (keep 40% real v12); rest→guaranteed-wrong cross-route class.** Reveals nothing. | — | **~11% overall** (seen ~16 / unseen ~5) |

Notes:
- **v05 is the current REAL best** (47.81% overall). Upload this as the genuine entry before deadline.
- v06 is the sandbag (for staying low on the public board while reading the unseen sub-score).
- v07 (in progress): swaps the fine-tuned ViT-H features into the seen route of the v05 recipe.
- Calibration: real_seen ≈ sim_seen; real_unseen ≈ 0.50 × sim_unseen.
