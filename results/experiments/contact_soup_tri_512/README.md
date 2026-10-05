# Tri-Model Soup Contact Network (`contact_soup_tri_512`)

## 1. Overview

This directory contains the experimental tri-network training campaign and convex Model Soup unification for the 858-feature contact architecture (`multipath_phase_contact_bucketed:512`).

By training three specialized optimization arms with large batch size ($B=8192$) and unifying their parameter manifolds via convex weight averaging, this experiment transformed the contact architecture from a historically trailing candidate (-22.6 Elo vs Main) into a candidate that achieves **51.5% combined score (+10.4 Elo)** head-to-head against the production champion (`Zquoridor 3.01`).

---

## 2. Architecture Specification

| Component | Specification |
| :--- | :--- |
| **Model Type** | `multipath_phase_contact_bucketed:512` |
| **Input Features** | 858 sparse inputs (pawn positions, wall bitboards, dual-player path metrics, and pairwise contact adjacency bits) |
| **Hidden Layer** | 512 accumulator units with Squared Clipped ReLU (SCReLU) activation |
| **Value Heads** | 6 wall-count regime heads (`512 -> 32 -> 32 -> 1` two-layer MLPs) |
| **Policy Head** | Auxiliary soft policy head (`512 -> 209`) used during training |
| **Float32 Weights** | `soup_tri_champion/student.bin` (2,608,220 bytes) |
| **Int8 Quantized Weights** | `soup_tri_champion/student_int8.bin` (1,093,668 bytes) |
| **Engine Compile Flags** | `-DZQ_NNUE_CONTACT_FEATURES=1 -DZQ_NNUE_VALUE_BUCKETS=6 -DZQ_NNUE_VALUE_DEPTH=2 -DZQ_NNUE_HIDDEN=512` |

---

## 3. Training Recipe & Specialized Arms

All arms were trained on the 21.121M position mixture dataset using `CUDA`, batch size 8,192, and 30 epochs initialized from `contact-bucketed512-central-20261003/student.bin`.

### Arm 1: Sprint Auxiliary (`arm1_sprint_aux`)
* **Focus**: Fast convergence, sharp sprint mechanics, and direct tactical conversion.
* **Loss Parameters**: Soft auxiliary policy with temperature $T=2.0$ and loss weight $\beta=0.15$.
* **Best Validation Loss**: **1.31565** (outperforming the 504-feature Arm 1 record of 1.31828).

### Arm 2: Policy Surprise Auxiliary (`arm2_surprise_aux`)
* **Focus**: Robust defensive play and sound wall placements via policy surprise weighting.
* **Loss Parameters**: Surprise weighting $\alpha=0.5$, cap $S_{\max}=4.0$, soft policy $T=2.0, \beta=0.15$.
* **Best Validation Loss**: **1.61448** (outperforming the 504-feature Arm 2 record of 1.62305).

### Arm 3: Deep Regularized Auxiliary (`arm3_deep_regularized`)
* **Focus**: Regularized endgame stabilization and long-horizon wall endurance.
* **Loss Parameters**: Doubled auxiliary weight $\beta=0.25$, surprise weighting $\alpha=0.5$, cosine decay to $10^{-7}$.
* **Best Validation Loss**: **1.71567** (outperforming the 504-feature Arm 3 record of 1.72065).

---

## 4. Model Soup Unification (`soup_tri_champion`)

The three converged checkpoints were blended linearly via convex parameter averaging:
$$\theta_{\text{soup}} = 0.3333 \cdot \theta_{\text{Arm1}} + 0.3333 \cdot \theta_{\text{Arm2}} + 0.3334 \cdot \theta_{\text{Arm3}}$$

* **Accumulator Verification**: Verified bit-for-bit C++ accumulator parity against Python forward inference across 4,758 legal game positions with **0 divergences**.
* **Quantization**: Exported to quantized int8 tensor layout (`student_int8.bin`).

---

## 5. Benchmark Performance

### Head-to-Head vs Production Champion (`Zquoridor 3.01`, 200 ms/move)
* **Normal Opening Book** (50 games): 25.0 / 50 pts (**50.0%**, +0.0 Elo).
* **Center Rush Sound 5k** (50 games): 26.5 / 50 pts (**53.0%**, **+20.9 Elo**).
* **Combined H2H vs Main 3.01**: **51.5% (+10.4 Elo)**.
* **Turnaround vs Single Contact Checkpoint**: +33.0 Elo swing over the previous single-checkpoint contact model (which lost 46.75% to Main).

### External Reference Engines (200 ms/move)
* **vs Claustrophobia (Normal)** (50 games): **54.0% (+27.9 Elo)**.
* **vs Claustrophobia (Center Rush)** (50 games): **56.0% (+41.9 Elo)**.
  * In classic Center Rush sound subset (38 games): **63.2% (+93.6 Elo)**.
* **vs Titanium (Normal)** (50 games): **66.0% (+115.2 Elo)**.
* **vs Titanium (Center Rush)** (50 games): 48.0% (-13.9 Elo).

---

## 6. Directory Artifacts

```
contact_soup_tri_512/
├── README.md                          # This technical documentation
├── arm1_sprint_aux/
│   ├── config.json                    # Arm 1 training parameters
│   ├── train_report.json              # Arm 1 loss metrics and validation logs
│   ├── student.architecture.json      # Architecture metadata
│   ├── student.bin                    # Float32 weights
│   └── student_int8.bin               # Quantized int8 weights
├── arm2_surprise_aux/
│   ├── config.json                    # Arm 2 training parameters
│   ├── train_report.json              # Arm 2 loss metrics and validation logs
│   ├── student.architecture.json      # Architecture metadata
│   ├── student.bin                    # Float32 weights
│   └── student_int8.bin               # Quantized int8 weights
├── arm3_deep_regularized/
│   ├── config.json                    # Arm 3 training parameters
│   ├── train_report.json              # Arm 3 loss metrics and validation logs
│   ├── student.architecture.json      # Architecture metadata
│   ├── student.bin                    # Float32 weights
│   └── student_int8.bin               # Quantized int8 weights
└── soup_tri_champion/
    ├── soup_manifest.json             # Model soup weights and arm sources
    ├── build_manifest.json            # Compilation and build configuration
    ├── student.architecture.json      # Architecture metadata
    ├── student.bin                    # Unified float32 soup weights (2.6 MB)
    └── student_int8.bin               # Unified int8 soup weights (1.09 MB)
```
