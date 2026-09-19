# Clean race384-search10 ablation

This branch creates the width-control that the existing historical
`race384-search10-ft` cannot provide.

The candidate uses exactly the current conservative search10 fine-tune recipe:

- dataset: `historical2m-search10-conservative`;
- 40 epochs, batch 4096;
- LR 3e-5, 2-epoch warmup, cosine to 3e-6;
- QAT throughout;
- trunk LR scale 0.1;
- policy:value loss 1:1.

Only hidden width is 384 instead of 512. Initialization comes from the
corresponding historical `race384-anneal` direct checkpoint.

Heavy training cannot run in GitHub Actions until the 2M dataset is exposed to
the runner; the recipe is pinned here so the eventual comparison is clean.
