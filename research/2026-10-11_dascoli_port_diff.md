# d'Ascoli port vs our replica vs our best Deep recipe — 2026-10-11

Lever #1 of `2026-10-11_frontier_delta.md`. d'Ascoli Deep public .7309 vs ours .6888.

**Port**: `external/dascoli-word-decoding` (gitignored, `.gitignore:25`), commit
`e1262ee36aa2fbaa6965e5bfb893f6aaee0dd693` (2026-07-01, "Update default experiment"; no newer upstream).
**LICENSE: CC BY-NC 4.0**, not OSI. We can run it locally; we can't vendor it, and a non-commercial clause
attaches to anything derived from it. All cites below are relative to the port root.

**Replica**: bead `megspeech-7gf.51` (closed). Run B = the port's own code with these changes: all sub-0 PNPL
train runs, MOCHA dropped, **duration 1.0 s**, 15 epochs. It scored .6408 on our ruler (n=2512; S1 .7014 /
Moth .5802), vs our ema9998 single at .6453. Arm 7gf.51.1 put their 16L/1024 context on our encoder and scored
.6234 (null).

**Our best**: stem `ema-cd25-xsp10sem05-s4` (aux-only pretrain: semantic .5 + speech 1.0, heads 0), then
`-ft-s5` (heads 1.0). This is the best Deep single at val .6320.

## Port recipe (default grid)

| Axis | Port | Cite |
|---|---|---|
| Entry | `python -m sentence_decoding.grids.libribrain100` | grids/libribrain100.py:135-145 |
| Data query | Sherlock1 sub-0 ses1-10 run1 train · ses11 run2 val · ses12 run2 test | libribrain100.py:80-97,105 |
| Alt query | `subject == 'LibriBrain100/0'` → PNPL partitions (validation→val) | libribrain100.py:128-132; studies/libribrain100.py:117-144 |
| Window | Word event, start 0.0, **duration 3.0 s** | libribrain100.py:99-124 |
| Preproc | MNE filter 0.1-40 Hz → 50 Hz → RobustScaler per **recording** → baseline mean 0-0.5 s → clamp ±5 | defaults.py:187-202; neuralset/features/neuro.py:94-114,170-175,226-227 |
| Target | t5-large **encoder**, word in isolation, token-mean, hidden_state 12/25, 1024-d, lowercased | defaults.py:203-215; neuralset/features/text.py:310-312,395-401,442; utils.py:247-257 |
| Encoder | ChannelMerger(270ch, Fourier pos 2048, per-subject, spatial drop .2) → 1x1 512 → SubjectLayers → 5-layer GLU dilated CNN hidden 160 → 1024 → attention time-pool | defaults.py:221-241; neuraltrain/models/simpleconv.py:223-309,385-386 |
| Context | x-transformers Encoder 16L/1024/16h, rotary, scalenorm, **bidirectional over the sentence** | defaults.py:244; transformer.py:23-72; pl_module.py:71-110 |
| Loss | SigLIP on **every heard word**, transformer output only (CNN loss logged only); learnable temp init 10, bias -10; dup words = positives (thr .999) | pl_module.py:126-147; losses.py:129-218 |
| Optim | AdamW lr 1e-4, wd 0, cosine per epoch T=50, 50 ep, patience 10, no clip, no EMA | defaults.py:175-182; main.py:51-59; pl_module.py:168-185 |
| Batch | 128 consecutive words = whole sentences, sentence blocks shuffled per epoch | utils.py:220-245; main.py:230-243,410 |
| Select | `val_retrieval_acc10_vocab=libribrain50_macro_0` (top-10 balanced, 50 words) | libribrain100.py:27-78,114; main.py:401-423 |
| Infer | nearest t5 prototype among the vocab words, per instance, no averaging, no LM unless `lm_path` | callbacks.py:298-380; main.py:467-468 |

Their metric matches ours: top-10 macro over the same 50-word vocab (`VOCAB_SIZE = 50`,
`megspeech/data/labels/vocabulary.py:14`).

## Diff, ranked by plausible effect on the public score

| # | Item | Port | Replica (Run B) | Our best | Plausible effect |
|---|---|---|---|---|---|
| 1 | **Window + baseline** | 3.0 s, baseline = mean of 0-0.5 s post-onset | 1.0 s (changed) | 1.0 s, no baseline (`baseline 0`) | **Untested faithfully.** A 3 s window reaches the 2-4 following words. Ours at 1.5 s (w15) was public −.042, but that was our arch with no every-word context. Estimate ±.02-.05: the biggest unknown |
| 2 | **Training data** | default Sherlock1 ses1-10 only; .7309 setup unknown | all sub-0 PNPL train | all sub-0, VAL carved 10% | Run A (S1 only) .506 vs Run B .6408. Data dominates. The .7309 surely uses more than the default query. Unknown for 2026 releases |
| 3 | **Objective** | SigLIP to t5 of every heard word through a bidirectional 16L sentence transformer; nearest-prototype readout | same | softmax over 50 + word head, tau prior; t5 semantic SigLIP only as aux in pretrain (sem .5) | Transplant 7gf.51.1 null (.6234), so the context alone doesn't carry it. Readout / prior is ±.01 |
| 4 | **Encoder** | merger + subject layer + 160-hidden GLU CNN + attention pool; RobustScaler per recording; 0.1-40 Hz | same | 256-hidden 6-layer conv, linear head, no merger; scaler fit on train windows (`model/input/scaler.py`), same clamp 5, 0.1-125 Hz + decimation AA <25 Hz | Run B single ≈ ours (.6408 vs .6453) at 1 s, so the encoder ≈ parity. Per-recording scaling is a free session-drift fix, ≤.01 |
| 5 | **Optim / selection** | lr 1e-4, wd 0, 50 ep + patience 10, batch 128 sentence-contiguous, no EMA | 15 ep cosine (changed) | lr 1e-3, wd .05, ls .1, EMA .9998, batch 64, 40 ep | Run B peaked by ep4-6 and stayed flat, so more epochs ≈ 0. EMA is ours and a small win |
| — | Augment | input dropout .1 + spatial merger dropout | same | aug 0 | ≤.005 |
| — | Ensembling / LM | none in code (LM optional via `lm_path`) | — | 8-ens .6626 + blend | .7309 may include an ensemble or LM; not shipped |

The 1 s replica already matches our single, so the port's code at 1 s is **not** where the .04 public gap lives.
The remaining candidates are, in order: (a) the 3 s window with post-onset baseline, the only faithful axis Run B
changed; (b) undisclosed data / ensembling behind .7309.

## Eval split

- **Default grid: no match.** Val/test are single Sherlock1 runs (ses11 run2 / ses12 run2).
- **With `subject == 'LibriBrain100/0'`:** val = pnpl validation (selects epoch, `main.py:401-423`) and test =
  pnpl test. Our local test pools pnpl validation+test, so their selection set is half our local test. Score
  through our own ruler instead: the saved per-word `retrieval_outputs/{split}_{idx}.pt` (`main.py:110-123`) or a
  re-scorer. Run B's `score_ref.py`/`score_dev.py` were lost with the old scratchpad and need rewriting as a
  committed `python -m` entrypoint.
- For a clean comparison, select on our VAL: change the query so that val = our carved 10% session list and
  test = pnpl validation ∪ test.

## Run command (not run)

Separate env inside the checkout. Don't install into our `.venv`. Pins come from the port README.

```powershell
cd D:/personal_projects/pnpl_megspeech_kaggle/external/dascoli-word-decoding
uv venv --python 3.11 .venv
uv pip install --python .venv torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv -e neuralset[dev] -e neuraltrain pnpl lightning wandb x_transformers transformers sentencepiece
# kenlm only if lm_path is used. neuraltrain pulls braindecode@git
$env:SAVEPATH = "D:/data/neural/processed/pnpl_libribrain/dascoli_ref"
$env:DATAPATH = "D:/data/neural"
$env:LIBRIBRAIN100_PATH = "<pnpl HF root under D:/data/neural/raw>"   # verify the dir Run B used
.venv/Scripts/python -m sentence_decoding.grids.libribrain100
```

Grid overrides go in the gitignored checkout (`libribrain100.py:99-124` / `defaults.py:175-182`), never in our
repo:
- `data.query = "subject == 'LibriBrain100/0'"`, or our VAL/local-test session lists
- `"use_wandb": False`
- MOCHA excluded (leak, as in Run B)
- `duration` kept at **3.0** (the point of the run)
- `num_workers` raised (Run B was CPU-loader-bound at 7% GPU)

## Cost estimate

| | Run B (1 s, measured) | Faithful 3 s (estimated) |
|---|---|---|
| Setup + preproc | ~54 min + 33 min | similar, RAM ~3× (keep_in_ram) |
| Per epoch | 68-80 min, GPU util 7%, loader-bound | ~1.5-3 h if loader scales with samples |
| Epochs | 15 | 50 cap, patience 10; Run B peaked ep4-6, so ~15-20 in practice |
| Wall | 8.9 h | **~25-50 h**; a 10-epoch cap gives ~15-30 h |
| Peak VRAM | 6.6 GiB | ~8-12 GiB (3× time axis on the CNN; transformer unchanged) |

Wall-clock ÷ gain: ~20-50 h for a coin-flip on ±.02-.05. Cheaper instrument first: our own encoder at a 3 s
window with post-onset baseline, scored with our ruler (our pipeline, ~hours). It answers whether a 3 s window
plus every-word context carries signal before the port's loader-bound run is paid for.

## Verdict (2026-10-11)

Lever closed on the window axis: the Kaggle test releases 1 s windows only (`docs/competition/overview.md:138`, `meg: (B, 306, 250)` at 250 Hz), so a 3 s window cannot be applied at test time, and the .7309 entry must also decode from 1 s windows. The port at 1 s (Run B) matches our single model (.6408 vs .6453), so the gap to .7309 is not in the shipped code: it rests on data, ensembling or assets the port does not ship. Cheap free fix kept on the list: per-recording robust scaling (<=.01, directional).
