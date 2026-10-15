# viz — sample viewer

Browse a random, stratified set of sentences (default 1000, drawn the way the balanced VAL partition is: `megspeech/data/stratify.py`; every labelled word in them is a sample): MEG, audio and text in the middle, each
selected model's output on the right. Pattern follows `mindscape/neuroviz`: Python exports static
files, a vanilla-JS canvas app reads them, no build step.

```bash
uv run python -m megspeech.evaluation.viz export --models <run> [<run> ...] [--n 1000] [--seed 0] [--device cuda]
uv run python -m megspeech.evaluation.viz serve [--port 8000]   # http://localhost:8000
uv run python -m megspeech.evaluation.viz agree [--name default]   # logs the export's saliency agreement report (agreement.py)
```

Exports land under the data root (`paths.yaml`), `viz/<export-name>/` — never in git. `serve`
mounts `web/` at `/` and the export at `/data/`.

## Data contract (`/data/`)

`meg.file` / `audio.file` are paths relative to `/data/` (e.g. `"meg/<id>.f32"`); `models/<name>.json` uses the exact model name.

| file | content |
|---|---|
| `manifest.json` | `{"name", "seed", "strata": [str], "models": [{"name", "track", "labels": [str], "metric": str, "score": float (the model's own metric on THIS sample), "metrics": {name: float}, "saliency": {"file", "channels"}}], "channel_names": [str] (optional), "samples": "samples.json"}` |
| `samples.json` | list of `{"id", "source", "subject", "session", "text", "target": [start, end] (char span of the scored item in `text`), "label", "sentence" (`run/sentence` key; the list groups samples by it), "target_t": [t_start, t_end] (seconds, same clock as `t0`; the scored word in the window), "strata": {str: str}, "meg": {"file", "channels", "times", "sfreq", "t0"}, "audio": {"file", "sfreq", "t0"} \| null}` |
| `meg/<id>.f32` | little-endian float32, row-major `[channels, times]`, per-channel z-scored |
| `audio/<id>.wav` | 16-bit PCM mono, the stimulus window aligned to the MEG window (`t0` in seconds, same clock) |
| `models/<name>.json` | `{"name", "projections": {"pca": {"explained": [float, float]}, "umap": {"explained": []}}, "samples": {id: {"pred", "correct", "p_true", "error" (= 1 - p_true, the mass on the other classes), "rank", "ranked": [[label, prob]] (all 50, most probable first), "emb": {"pca": [x, y], "umap": [x, y]}}}}` (`explained` = variance share per axis, empty where the projector has none) |
| `models/<name>.saliency.u8` | uint8, samples × channels, row-major in `samples.json` order: each word's integrated-gradients magnitude per sensor (`megspeech/evaluation/saliency.py`, label log-probability vs a zero window, summed over its own window), over that row's peak × 255 |

Multiclass outputs have no positive/negative split, so the right panel ranks all 50 words for
the selected sample (scrolls; a dashed line marks where top-10 ends) and a 2-D embedding scatter (the context head's input embedding → UMAP or PCA, picked in the
panel header) coloured by word class (default: pronoun, determiner, preposition, conjunction, auxiliary, adverb, content word, in the colour-blind-safe Okabe-Ito palette, each word in its dominant Penn/UD class), label, word, source, position in the sentence or error, not a score KDE. A legend under the
scatter lists the categories by count; hovering a chip highlights its points, clicking pins it. Only the active model's
prediction is shown. UMAP (cosine, 15 neighbours, min_dist 0.1,
seed 0) keeps local neighbourhoods, so word clusters show; PCA's two axes explain little of the variance (the header
shows how much). Both are fitted per model, so scatters are not comparable across models. Filters (source, sentence length in words from/to) open in a popup next to the
search box; the button shows how many are set. The left panel is a word table (id, word, pred, error = 1 − p(true), rank of the true word among all 50; the row's left edge is the source colour) for the active model; click a header to sort (error/rank start descending, so the largest errors come first), and ↑/↓ follow the table order. The top bar holds the model select and its metric chips: the competition score on this sample, then macro bAcc@10, bAcc@1, F1@1, mAP (samples ranked by each word's probability) and MRR over the sampled words (hover a chip for its definition). Pred is green only on an exact top-1 hit; the sentence card gives the true word's rank. The audio track folds away where there is none. The middle panel stacks sentence, MEG and audio at 1:2:1; drag the gutters between them to resize (remembered per browser), and each part scrolls on its own. The raw MEG panel is collapsible. Its rows are labelled `#<location> <sensor>`: the 3-digit Neuromag helmet location, then its magnetometer `mag` or one of its two orthogonal planar gradiometers `grad1`/`grad2`; a location's three sensors are always drawn together, a faint line between locations. The picker spreads locations evenly, ranks them word-evoked (by their best channel's RMS inside the target word over the whole window), by model weight (the active model's saliency for this word, summed over the location), or keeps an MNE Vectorview lobe (temporal ≈ auditory cortex); by kind it draws single channels. At the header's top right a helmet map (from above, nose up, MNE `Vectorview-all` layout positions in `manifest.sensors.xy`) colours each location by its summed weight for this word and rings those drawn; click a dot to draw it and its nearest neighbours. With saliency exported, every trace is coloured by its percentile within that word's map, cold to hot (key in the panel header), and the panel scrolls once rows would fall under 14 px: where the model's evidence sits, not that the sensor causes the word. Stimulus audio exists only for Sherlock: pnpl ships no `stimuli/` for TheMoth or MOCHATIMIT.
