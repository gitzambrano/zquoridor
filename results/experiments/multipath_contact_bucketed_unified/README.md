# Multipath Contact Bucketed Unified Experiment (`multipath_contact_bucketed_unified`)

## Overview

This experiment evaluated the preliminary unification of contact features (858 inputs) with multi-head wall-count bucketing (6 buckets) and 2-layer MLPs on clean master datasets.

## Specification
* **Architecture**: `multipath_phase_contact_bucketed:512`
* **Features**: 858 sparse inputs
* **Hidden Width**: 512 units (SCReLU)
* **Value Heads**: 6 regime heads (`512 -> 32 -> 32 -> 1`)
* **Role**: Provided early structural verification of the 6-bucket contact evaluator before the 21M central fine-tune and subsequent Model Soup campaigns.
