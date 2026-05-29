Low-Rank Decay (LRD)

An optimizer-side spectral decay step for scale-invariant Transformers.

Work in progress — this repo holds the code and early experiments for an
ongoing draft. Results here are preliminary.


## Why
Modern Transformers use RMSNorm and QK-Norm, which make parts of the model
scale-invariant: multiplying a weight matrix by a positive scalar doesn't
change what the layer computes. In that setting, ordinary weight decay just
shrinks the Frobenius norm — a purely radial move that doesn't actually
simplify the function. The task gradient lives on the sphere (tangential),
weight decay pulls toward the origin (radial), and the two barely interact.

So if radial decay doesn't do much for a scale-invariant layer, what should
decay actually act on? Our answer: the singular-value spectrum.

## What LRD does

After a normal optimizer step, LRD applies one extra step:

    W ← W - η·λ·polar(W)

where polar(W) = UVᵀ for W = UΣVᵀ. This subtracts a constant from each
singular value (soft-thresholding), so small singular directions get pushed
to zero and the matrix drifts toward lower rank. We approximate polar(W)
with a few Newton-Schulz iterations, so there's no SVD in the training loop.

Contrast:
- AdamW: σᵢ → (1-ηλ)σᵢ   (shrinks everything proportionally, same spectral shape)
- LRD:   σᵢ → max(σᵢ-ηλ, 0)  (soft-thresholding, reshapes the spectrum)

## Early observations (modular addition, p=113)

- LRD drives the stable rank of Q/K matrices down sharply *before* test
  accuracy rises — spectral collapse seems to precede generalization.
- LRD appears to widen the low-data regime where grokking still happens,
  compared to L2/AdamW.

These are single-run observations on one task. Multi-seed runs, learning-rate
controls, stronger baselines, and causal tests are still needed before any of
this is solid. Treat it as a direction, not a result.

## Positioning

LRD probably shouldn't be an always-on pretraining optimizer — collapsing rank
too early can throw away directions you'll need later. It's meant more for
mid-training / SFT / CoT-SFT, where broad representations already exist and the
goal is to consolidate task structure and drop memorization-like high-rank
noise.


## Status / TODO

- [ ] Multi-seed runs with error bars
- [ ] Learning-rate matched baselines
- [ ] More tasks (modular multiplication, permutation composition)
- [ ] Causal intervention: does rank collapse *cause* grokking?
- [ ] Compare exact-SVD nuclear decay vs Newton-Schulz LRD

## References

Power et al. 2022 (grokking); Nanda et al. 2023 (mechanistic interp);
Loshchilov & Hutter 2019 (AdamW / decoupled decay); Higham 2008 (Newton-Schulz).
