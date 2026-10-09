# Contact-Bucketed Single Checkpoint Candidate (`contact-bucketed512-central-20261003`)

## 1. Overview

This directory contains the initial single-checkpoint training run of the 858-feature contact architecture (`multipath_phase_contact_bucketed:512`).

It introduced pairwise contact adjacency bits on top of the 504 multipath-phase features, paired with 6 wall-regime value heads and 2-layer MLPs.

---

## 2. Architecture & Training Details

| Component | Specification |
| :--- | :--- |
| **Model Type** | `multipath_phase_contact_bucketed:512` |
| **Input Features** | 858 sparse inputs |
| **Hidden Layer** | 512 units with SCReLU |
| **Value Heads** | 6 wall-regime heads (`512 -> 32 -> 32 -> 1`) |
| **Training Mixture** | 21.121M positions (75% search + 25% master) |
| **Schedule** | 112 epochs, QAT, cosine learning rate decay |
| **Best Val Loss** | **1.18894** (lowest single-checkpoint loss at the time) |
| **Weights** | `student_int8.bin` (1,093,668 bytes) and `student.bin` (2,608,220 bytes) |

---

## 3. Benchmark Outcome & Findings

Evaluated across 600 games at 200 ms/move:
* **vs Main Champion (Normal)** (100 games): 46.5% (-24.4 Elo).
* **vs Main Champion (Center Rush)** (100 games): 47.0% (-20.9 Elo).
* **vs Claustrophobia (Normal)** (100 games): 55.5% (+38.4 Elo).
* **vs Claustrophobia (Center Rush)** (100 games): 46.0% (-27.9 Elo).
* **vs Titanium (Normal)** (100 games): 61.0% (+77.7 Elo).
* **vs Titanium (Center Rush)** (100 games): 54.0% (+27.9 Elo).
* **Combined H2H vs Main**: 46.75% (-22.6 Elo).

### Conclusion & Legacy
While it achieved strong static evaluation loss and won external normal matches, it suffered in head-to-head tactical play against Main (-22.6 Elo). This checkpoint served as the initial warm-start weight (`init_from`) for the subsequent `contact_soup_tri_512` campaign, which resolved the H2H deficit using convex Model Soup.
