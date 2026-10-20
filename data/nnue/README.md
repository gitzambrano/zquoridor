# NNUE Checkpoint — zQuoridor 4.10

Architecture: 858 sparse features, 512 hidden neurons, six bucketed value heads and 209 policy outputs.

Production INT8 SHA-256: `fe8473255fd219fab247b1a7d44be25384a6cfc5daa5c301def49109badb7f2b`.

Only the quantized policy tail changed relative to 4.03. The float32 file is a retained historical value and policy reference; it does not reproduce the active policy if re-quantized.

The production browser payload, standalone HTML and native engine must use this same INT8 checksum. See [release notes](../../docs/releases/4.10.md).
