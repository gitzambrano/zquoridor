# Policy Policy25 baseline — controlled promotion handoff (20 October 2026)

## Identity

The user-provided `zQuoridor401_policy_ordering.zip` contains a complete compatible **INT8 NNUE checkpoint**, not a new model architecture or a value retraining. Its target is `zq401_policy_policy25_int8.bin`, SHA-256:

`fe8473255fd219fab247b1a7d44be25384a6cfc5daa5c301def49109badb7f2b`

The original production 4.01/4.02 network SHA-256 is:

`45fa3ea7db3127a0b907a23d268b0c55c2020e1018513d54bb920cdbe134abe0`

Both checkpoints have exactly **1,093,668 bytes**. Bytes 0–985,823 (feature layer and all value heads) are **bit-for-bit identical**. Policy changes only the INT8 policy-head segment from byte 985,824 onward. Their architecture remains `multipath_phase_contact_bucketed`, 858 features, 512 hidden, 6 value buckets, 209 policy output logits, QA=255 and QB=64. The original float32 NNUE file is **not** a float32 version of Policy; never run quantization on it over the promoted Policy weights.

## Existing independent benchmark, and its limits

The controlled local Linux 200 ms screen ran both checkpoint files through the same **4.01 search executable**, not the 4.02 Lazy+Sparse engine. It reports:

| 200 ms comparison | Policy | Original | N per arm |
|---|---:|---:|---:|
| H2H normal | 60.0% | 40.0% | 20 total games |
| H2H center | 55.0% | 45.0% | 20 total games |
| External Titanium normal | 100.0% | 87.5% | 8 each |
| External Titanium center | 87.5% | 62.5% | 8 each |
| External Claustrophobia normal | 62.5% | 75.0% | 8 each |
| External Claustrophobia center | 62.5% | 37.5% | 8 each |

**Overall H2H: 57.5% (23/40 points), paired bootstrap 95% [47.5%, 67.5%].** The interval includes 50%, normal Claustrophobia regressed in the small screen, and no 3+2 games on the Policy checkpoint have been completed. The ZIP README's 212-game 64.4% claim was supplied by its author and has not been independently reproduced by this repo. The operator nevertheless specifically instructed promotion to a new baseline; that is an explicit policy decision, **not proof of improvement under every time control**.

## 4.02 compatibility

Search **Lazy/Eager** and the **Sparse Policy** quantized evaluator are not coupled to particular training targets or weight bytes. The policy head is computed over the same 209 canonical actions, restricted to legal moves, with perspective mirroring. Policy Policy25 is therefore binary- and semantically compatible with 4.02 without changing `src/mcab.hpp` or `src/nnue.hpp`.

Other new NNUE policy weights also work **if they retain the deployed 858/512/209 architecture, quantization parameters and action mapping**. New 768-hidden networks, 1,370-feature networks, Transformers or policies with a different action vocabulary are **not drop-in compatible**; they require an architecture-aware runtime change, rebuilding, and policy parity tests.

## Synchronized deployment (required)

Run the integrity-checked installer on a complete checkout:

```bash
python tools/promote_policy_policy25.py /path/to/zQuoridor401_policy_ordering.zip --dry-run
python tools/promote_policy_policy25.py /path/to/zQuoridor401_policy_ordering.zip
```

The script must replace **both** `data/nnue/nnue_weights_int8.bin` and `gui_web/zquoridor.data`, archive the original, update `data/nnue/nnue_weights.architecture.json` and `data/nnue/release.json`, run `gui_web/build_standalone.py` to regenerate **both** `gui_web/zquoridor.html` and `docs/index.html`, and verify the embedded data bytes. Both native and browser must advertise the new checkpoint SHA-256 after promotion.

Do **not** merge this promotion as complete based only on README or metadata changes. The required binary files and standalone bundles must actually be present in the commit. No remote GitHub Actions tournament was launched for the promotion.
