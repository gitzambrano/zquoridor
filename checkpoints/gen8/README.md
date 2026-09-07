# Gen8 production checkpoint
Exact Gen8 baseline promoted to main.

- Source workflow run: `33973129847`
- Source artifact: `gen8-search-ci-completed` (`9971548567`)
- Search baseline merged in main: `fix/strong-selfplay-wandering` at `12ea0136e0c1d73ac63847b4d72d0171b4ff9473`
- Float SHA-256: `add43055e32154f76d376c594718cdf1bb09d8f777f55481061e486c1705393c`
- Int8 SHA-256: `b03f392ab6d13974149320ec793472b957f613b16070a95863f29e5604359818`
- Architecture: 354 inputs, hidden 256, policy 209, QA 255, QB 64.

`nnue_weights.bin` is the recoverable float checkpoint and `nnue_weights_int8.bin` is the production quantized network. The two `train_config_*.json` files are the checkpoint metadata preserved by the original Gen8 run. The original workflow did not upload optimizer/PyTorch state, so no optimizer state is fabricated here.
