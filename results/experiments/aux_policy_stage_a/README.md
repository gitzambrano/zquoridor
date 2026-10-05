# Auxiliary Policy & Tri-Model Soup Stage A (`aux_policy_stage_a`)

## 1. Overview

This directory contains the training arms, policy ablations, and Model Soup combinations that generated the current **Production Champion: Zquoridor 3.01** (`Soup_Tri_Equal`).

The core innovation of this campaign was introducing a training-only auxiliary soft policy head ($T=2.0, \beta=0.15$) combined with policy surprise loss weighting ($\alpha=0.5, S_{\max}=4.0$), followed by a convex Model Soup of three specialized arms.

---

## 2. Architecture Specification

| Component | Specification |
| :--- | :--- |
| **Model Type** | `multipath_phase_bucketed:512` |
| **Input Features** | 504 sparse inputs (pawn locations, wall bitboards, path distance differences) |
| **Hidden Layer** | 512 units with Squared Clipped ReLU (SCReLU) |
| **Value Heads** | 6 wall-count regime heads (`512 -> 32 -> 32 -> 1`) |
| **Policy Head** | Auxiliary soft policy head (`512 -> 209`) used during training |
| **Production Weights** | Promoted to `data/nnue/nnue_weights.bin` and `data/nnue/nnue_weights_int8.bin` (731,172 bytes) |
| **Engine Compile Flags** | `-DZQ_NNUE_VALUE_BUCKETS=6 -DZQ_NNUE_VALUE_DEPTH=2 -DZQ_NNUE_HIDDEN=512` |

---

## 3. Training Arms & Checkpoints

All arms trained on 21.121M positions (75% search + 25% master) starting from frozen V3 weights:

### 1. `A0_control` (Control Baseline)
* Standard value-only loss (policy disabled), 20 epochs.
* Val Loss: 1.19130. Regressed on Center Rush openings (48.75%).

### 2. `A1_aux_t20_b15` (Auxiliary Soft Policy)
* Soft policy distillation ($T=2.0, \beta=0.15$), 20 epochs.
* Val Loss: 1.31828. Beaten baseline on Center Rush (+26.1 Elo swing).

### 3. `AS1_surprise_aux_t20_b15` (Policy Surprise Weighting)
* Added surprise weighting ($\alpha=0.5, S_{\max}=4.0$), 20 epochs.
* Val Loss: 1.62305. +45.3 Elo turnaround on Claustrophobia Normal.

### 4. `AS2_surprise_v2_t20_b25` (Epoch 14 Checkpoint)
* Increased auxiliary weight ($\beta=0.25$), cosine annealing towards $10^{-7}$.
* Achieved project validation loss record: **1.72065**.

---

## 4. Production Champion: `Soup_Tri_Equal`

Constructed by convex parameter averaging of the three best student arms:
$$\theta_{\text{production}} = 0.3333 \cdot \theta_{\text{A1}} + 0.3333 \cdot \theta_{\text{AS1}} + 0.3334 \cdot \theta_{\text{AS2-ep14}}$$

### Promotion Match Record (Zquoridor 3.01 Baseline)

| Opponent / Sub-suite | Opening Book | Score % | Elo Delta vs Previous Champion | Status |
| :--- | :--- | :---: | :---: | :--- |
| **vs Claustrophobia** | Normal Book | **65.0%** | **+42.5 Elo** | All-time record |
| **vs Claustrophobia** | Center Rush Book | **66.0%** | **+52.0 Elo** | All-time record |
| **vs Titanium** | Normal Book | **74.0%** | **+34.5 Elo** | Outperforms baseline |
| **vs Titanium** | Center Rush Book | **56.0%** | **+28.0 Elo** | Outperforms baseline |
| **Head-to-Head vs Main** | Normal Book | **53.0%** | **+20.9 Elo** | Direct victory |
| **Head-to-Head vs Main** | Center Rush Book | **53.0%** | **+20.9 Elo** | Direct victory |

**Verdict**: Officially promoted as **Zquoridor 3.01** (`data/nnue/nnue_weights_int8.bin`, sha256: `f29b4bdde846191a747166885b1523ece5198f067beaa86688e5462e18232792`).
