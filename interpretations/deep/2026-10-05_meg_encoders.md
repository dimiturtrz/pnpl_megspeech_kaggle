# MEG encoders: what we run, what we tried, what the field uses (2026-10-05)

Scores are official val macro BAcc@10, n=2512, unless marked; the seed band of the base recipe is ~0.60-0.63
(no EMA) and the base single model `sent40-ioi-ema9998-s0` scores 0.6453. Sources: each run's `context.json`,
`megspeech/model/words/conv.py`, `megspeech/model/sentence.py`, and the closing notes of the cited beads issues.

## The encoder we run

Per word (`ConvWords`), from `sent40-ioi-ema9998-s0/context.json`:

1. 1 s of 306 sensors from the word's onset, decimated 250 -> 50 Hz (50 samples).
2. Robust per-sensor scaling: training median / IQR, clamped (`Scaler`).
3. 1x1 convolution mixing 306 sensors into 256 channels.
4. Six dilated residual temporal conv blocks, dilations 1-2-4-8-1-2, kernel 3, dropout 0.3 (receptive field ~0.7 s).
5. Mean + max pool over the window -> one embedding per word; a linear word head reads the 50 words (~0.36 alone).

Across the sentence (`SentenceNet`): embeddings projected to 256, a 4-layer 4-head bidirectional transformer over
the span with onset-position and log inter-onset-interval inputs, and a 50-word softmax context head.

## Alternatives tried — none cleared the 1.5-point bar

| Alternative | Issue | Result |
|---|---|---|
| MEG-XL frozen, pooled features | 7gf.5.1 | 0.358 |
| MEG-XL frozen, channel-resolved (PCA 16) | 7gf.5.2 | 0.278 |
| MEG-XL frozen, continuous span segments | 7gf.5.4 | 0.366 |
| MEG-XL LoRA fine-tune | 7gf.5.3 | diverged at epoch 1 (recipe failure, not refuted); ~93 min/epoch |
| Conformer word encoder (MEGConformer ladder) | qi7.3 | 0.594 / conv+conformer 0.586, at/below seed band |
| Conformer mixing in the sentence model | 4te.7 | 0.595 vs 0.619 |
| Bidirectional SSM (S4D) after the convs | 7gf.27 | 0.614, in band |
| Spatial-attention merger + spatial dropout (Défossez 2023) | 7gf.55 | 0.6472 vs 0.6453 on EMA recipe (+0.013 without EMA was variance reduction) |
| GLU-gated trunk (d'Ascoli activation) | 7gf.29.7 | 0.6338 vs 0.6453 |
| Attention-pooled readout (2 queries) | 7gf.45 | 0.6146 vs 0.6205 |
| Hidden 512 | 4te.6 | 0.606, in band |
| Per-window instance norm (MEGConformer) | 7gf.25.14 | 0.6154, in band; kept as a regulariser for the test shift |
| d'Ascoli 2025 recipe, faithful full-data replication | 7gf.51 | 0.6408 vs 0.6453: matches, adds nothing as ensemble member |
| d'Ascoli 16-layer/1024 context transformer | 7gf.51.1 | 0.6234 vs 0.6205 |
| Two-stage: encoder trained on every word, then context model | 7gf.50 | every-word supervision 3 nulls (lexical 0.602, semantic 0.607, phonetic) |

Reading: at ~307k labelled words from one subject, capacity and front-end variants sit inside seed noise; the
plain CNN is not the bottleneck it looks like. MEG-XL frozen reads lose ~0.28 because time-averaging drops the
within-window time course the CNN reads (7gf.5.2); a fine-tuned MEG-XL is untested, not refuted. Self-supervised
pretraining on subject-0's own continuous MEG is untested (7gf.109.4).

## Field architectures (from the papers; mechanism summary, not re-verified here)

- **Défossez et al. 2023, Nat Mach Intell** ("Decoding speech perception from non-invasive brain recordings"):
  spatial attention over sensor 2-D positions -> subject-specific 1x1 layer -> 5 dilated residual conv blocks
  with GLU -> CLIP contrastive loss to wav2vec 2.0 features of 3 s segments. Our CNN descends from this trunk.
- **d'Ascoli et al. 2025, Nat Commun** (PMC12658044): Défossez-style conv trunk per word, then a deep transformer
  across the sentence's words, SigLIP to T5 word embeddings. Replicated here (7gf.51).
- **MEGConformer** (LibriBrain 2025 phoneme-track winner): conformer encoder (self-attention + depthwise conv),
  per-window instance normalisation. Ported in qi7.
- **MEG-XL** (`pnpl/MEG-XL`, arXiv 2602.02494): a large transformer pretrained on long (~150 s) multi-subject MEG
  context, tokenising each sensor's signal into 512-d time tokens (per-sensor tokens observed in our port, 7gf.5).
  Its Fig. 3: pretraining matches supervised only up to ~2.5 h of target data; subject 0 has ~50 h.
- **Jayalath et al. 2024** (arXiv 2410.19986): self-supervised pretext tasks (band rejection, phase shift,
  amplitude scaling) for MEG speech decoding across datasets; naive dataset pooling hurt.
- **EEG foundation models** (BIOT, LaBraM): per-channel patch tokens + transformer, masked-code pretraining; EEG,
  not used here.
