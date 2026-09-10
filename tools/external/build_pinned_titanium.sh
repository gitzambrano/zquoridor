#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-titanium-bin}"
TITANIUM_REPO="https://github.com/titaniummachine1/titanium-engine.git"
TITANIUM_SHA="1ac94755f69799f01b2e06869403e701bbf0bd51"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone --quiet "$TITANIUM_REPO" "$TMP/titanium"
git -C "$TMP/titanium" checkout --quiet --detach "$TITANIUM_SHA"
ACTUAL="$(git -C "$TMP/titanium" rev-parse HEAD)"
test "$ACTUAL" = "$TITANIUM_SHA"
RUSTFLAGS='-C target-cpu=native' cargo build --release \
  --manifest-path "$TMP/titanium/Cargo.toml" --bin titanium

mkdir -p "$OUT_DIR"
cp "$TMP/titanium/target/release/titanium" "$OUT_DIR/titanium"
printf '%s\n' "$ACTUAL" > "$OUT_DIR/titanium-sha.txt"
chmod +x "$OUT_DIR/titanium"
echo "Pinned Titanium built: $ACTUAL -> $OUT_DIR/titanium"
