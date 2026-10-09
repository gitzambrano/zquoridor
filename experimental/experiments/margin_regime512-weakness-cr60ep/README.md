# Margin Regime 512 Weakness CR 60ep (`margin_regime512-weakness-cr60ep`)

## Overview

This experiment evaluated an expanded 588-feature regime model on weakness and Center Rush curriculum data across 60 epochs with QAT.

## Specification
* **Architecture**: `margin_regime:512`
* **Features**: 588 sparse inputs
* **Hidden Width**: 512 units (SCReLU)
* **Outcome**: Verified that expanding input features without wall-count value head bucketing was insufficient to surpass the lighter multipath baseline (scored 49.0% vs Claustrophobia across 600 games), guiding the adoption of wall-count bucketing.
