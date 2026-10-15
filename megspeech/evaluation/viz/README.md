# viz — sample viewer

Browse a random, stratified set of sentences (default 1000, drawn the way the balanced VAL partition is: `megspeech/data/stratify.py`; every labelled word in them is a sample): MEG, audio and text in the middle, each
selected model's output on the right. Pattern follows `mindscape/neuroviz`: Python exports static
files, a vanilla-JS canvas app reads them, no build step.

```bash
uv run python -m megspeech.evaluation.viz export --models <run> [<run> ...] [--n 1000] [--seed 0]
uv run python -m megspeech.evaluation.viz serve [--port 8000]   # http://localhost:8000
```

Exports land under the data root (`paths.yaml`), `viz/<export-name>/` — never in git. `serve`
mounts `web/` at `/` and the export at `/data/`.

## Data contract (`/data/`)

`meg.file` / `audio.file` are paths relative to `/data/` (e.g. `"meg/<id>.f32"`); `models/<name>.json` uses the exact model name.

| file | content |
|---|---|
| `manifest.json` | `{"name", "seed", "strata": [str], "models": [{"name", "track", "labels": [str], "metric": str, "score": float (the model's own metric on THIS sample)}], "channel_names": [str] (optional), "samples": "samples.json"}` |
| `samples.json` | list of `{"id", "source", "subject", "session", "text", "target": [start, end] (char span of the scored item in `text`), "label", "sentence" (`run/sentence` key; the list groups samples by it), "target_t": [t_start, t_end] (seconds, same clock as `t0`; the scored word in the window), "strata": {str: str}, "meg": {"file", "channels", "times", "sfreq", "t0"}, "audio": {"file", "sfreq", "t0"} \| null}` |
| `meg/<id>.f32` | little-endian float32, row-major `[channels, times]`, per-channel z-scored |
| `audio/<id>.wav` | 16-bit PCM mono, the stimulus window aligned to the MEG window (`t0` in seconds, same clock) |
| `models/<name>.json` | `{"name", "projections": {"pca": {"explained": [float, float]}, "umap": {"explained": []}}, "samples": {id: {"pred", "correct", "p_true", "error" (= 1 - p_true, the mass on the other classes), "topk": [[label, prob]], "emb": {"pca": [x, y], "umap": [x, y]}}}}` (`explained` = variance share per axis, empty where the projector has none) |

Multiclass outputs have no positive/negative split, so the right panel shows the top-k bar list for
the selected sample and a 2-D embedding scatter (the context head's input embedding → UMAP or PCA, picked in the
panel header) coloured by word class (default: pronoun, determiner, preposition, conjunction, auxiliary, adverb, content word, in the colour-blind-safe Okabe-Ito palette, each word in its dominant Penn/UD class), label, word, source, position in the sentence or error, not a score KDE. A legend under the
scatter lists the categories by count; hovering a chip highlights its points, clicking pins it. Only the active model's
prediction is shown. UMAP (cosine, 15 neighbours, min_dist 0.1,
seed 0) keeps local neighbourhoods, so word clusters show; PCA's two axes explain little of the variance (the header
shows how much). Both are fitted per model, so scatters are not comparable across models. Filters (source, sentence length in words from/to) open in a popup next to the
search box; the button shows how many are set. The left panel is a word table (id, word, pred, error = 1 − p(true), rank of the true word among all 50, source) for the active model; click a header to sort (error/rank start descending, so the largest errors come first), and ↑/↓ follow the table order. Under the model select: macro bAcc@10, bAcc@1, F1@1, mAP (samples ranked by each word's probability) and MRR over the sampled words. The raw MEG panel is collapsible; its channel picker draws them spread by index, word-evoked (RMS inside the target word over the whole window, ranked), by kind (each Neuromag location = 1 magnetometer `m` + 2 planar gradiometers `g`) or by MNE Vectorview lobe (temporal ≈ auditory cortex). Stimulus audio exists only for Sherlock: pnpl ships no `stimuli/` for TheMoth or MOCHATIMIT.
