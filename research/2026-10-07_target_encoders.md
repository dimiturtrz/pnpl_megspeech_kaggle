# Frozen target encoders for the auxiliary alignment losses — prior art, candidates, shortlist

Scope: which frozen pretrained model should give the per-word / per-sentence targets for the MEG encoder's
auxiliary losses. The current choices (wav2vec2-xlsr-53 L15-18, t5-large h12, whisper-medium L24, gpt2-large ctx)
came from cited papers and were never compared. Tags: **[V]** checked against a primary source this session,
**[S]** from a search snippet or secondary page only (not verified), **[K]** from memory, not fetched this session,
**[I]** our inference.

## 1. Prior art

| Work | Frozen target(s) | What it measured / found |
|---|---|---|
| Défossez et al. 2023, Nat Mach Intell — https://arxiv.org/abs/2208.12266 | wav2vec2-large-xlsr-53 vs mel vs a learned "deep mel" | Segment top-10 with wav2vec2: 17.7 / 25.7 (EEG), 70.7 / 67.5 (MEG). Mel regression got 1.0-20.6, deep mel 15.4-64.4. The ar5iv text says "the activations of the **last four layers**" averaged [V, via ar5iv fetch]. **Mismatch:** `speech.py:33` uses `slice(15, 19)` (layers 15-18 of 24) and credits Défossez for it. Check the paper's methods text, or brainmagick's config, before trusting either one. |
| d'Ascoli et al. 2024/25 — https://arxiv.org/abs/2412.17829 (Nat Commun: https://www.nature.com/articles/s41467-025-65499-0) | t5-large, middle layer | Only one target. The arXiv text compares architectures, not target models [V, arXiv]. Whether the Nat Commun version adds a target ablation was not checked (paywall redirect) [unverified]. Top-10 of 250 words, up to 37%. |
| PNPL 2026 competition paper — https://arxiv.org/abs/2609.03231 | none | Deep-track baseline is a d'Ascoli-style supervised decoder at 73.23% BAcc. The broad track uses MEG-XL (pretrained on MEG only), 42.96%. No target-encoder comparison [V]. |
| Competitor write-ups (our `research/2026-10-06_munich_competitor_prior_art.md`) | T5 with D-SigLIP; CLIP to wav2vec2 | D-SigLIP beats CLIP when the target is T5 (18.2 vs 15.7 top-10 of 250). No team compared target models (per that doc). |
| Antonello et al. 2023, NeurIPS — https://arxiv.org/abs/2305.11863 | OPT/LLaMA 125M-30B; HuBERT, WavLM, Whisper (fMRI) | Encoding quality grows with log(size), for text models and for audio (r=0.991, ~+32% per decade of params). The Whisper family was the best audio encoder; its largest encoder (637M) was the top audio model [V, html fetch, summarised]. |
| Goldstein et al. 2025, Nat Hum Behav — https://hassonlab.princeton.edu/sites/g/files/toruqf3591/files/documents/Goldstein_et_al_Nature_HB_2025.pdf | Whisper acoustic / speech-encoder / decoder-word embeddings (ECoG, 100 h) | One model's three levels each predict their own stage of the cortical hierarchy on unseen conversations [S]. |
| Whisper → ECoG, 2026 — https://arxiv.org/abs/2606.02305 | Whisper layers | **Intermediate** Whisper layers predict best [S]. Our `whisper-medium-L24` is the last layer. |
| Vaidya et al. 2022, ICML — https://proceedings.mlr.press/v162/vaidya22a/vaidya22a.pdf; Millet et al. 2022 NeurIPS [K]; Li et al. 2023 Nat Neurosci [K] | APC, wav2vec, wav2vec2, HuBERT (fMRI/ECoG) | Lower layers track low-level areas; **upper-middle** layers best for phonetic/semantic areas [S]. HuBERT-class SSL beats supervised and filter-bank models [K]. |
| Tuckute et al. 2023, PLOS Biol — https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002366 | 19 audio DNNs | Most trained models beat filter-bank baselines, but "not all" do. Middle stages fit primary AC, deep stages non-primary AC [S]. |
| Oota et al. 2024 ACL, "Speech language models lack important brain-relevant semantics" [K] | wav2vec2 / HuBERT / Whisper vs text LMs | Speech-model alignment in late language areas mostly comes from low-level features; text LMs carry the semantics [K, unverified]. |
| Brain-tuning, Moussa & Toneva — https://arxiv.org/abs/2410.09230 | wav2vec2 / HuBERT / Whisper, ~90M each | Fine-tuning on fMRI improves semantic alignment, so frozen speech models under-use semantics [S]. |
| Imagined-speech MEG, 2026 — https://arxiv.org/abs/2605.08075 | BERT, Whisper, wav2vec2, BERT+wav2vec2 word targets | BERT and BERT+wav2vec2 were best for MEG word retrieval [S]. |
| Chinese MEG decoding — https://arxiv.org/abs/2506.12817 | wav2vec2/HuBERT on synthetic speech vs text features | Speech-feature help > text-feature help [S]. |
| Caucheteux & King 2022, Commun Biol [K] | GPT-2 layers | The middle layers of a causal LM map best onto MEG/fMRI [K]. |

**Our own evidence** (`bd show megspeech-99y`, alignment run on `sent40-ioi-ema9998-s0`):
- Every teacher has only **11-24 family-wise significant canonical components** out of 512, against 1024-1280 target dims.
- Every teacher-probe ceiling (BAcc@10 0.20-0.26) is **below** the MEG embedding's own 0.307.
- t5 (official val): top-1/5/20 canonical correlations 0.435/0.154/0.006, 14 components.
- Audio teachers are scored only on held-out TRAIN runs, where the encoder is in-sample, so they are upper bounds.

So the bottleneck is how much of the target MEG can predict, not the target's size [I].

## 2. Candidates
Notes on columns:
- Licences come from the HF API `cardData.license` (queried 2026-10-07) [V].
- Params and dims are from model cards [K unless marked].
- "Brain" = alignment evidence.
- "Fit" = frozen, ≤~1B, loads in HF transformers, can be precomputed once.

### Audio (word: 0.5 s after onset, mean-pooled frames; sentence: pool over the span)
| Model (HF id) | Params | Licence | Dim | Input | Brain | Fit |
|---|---|---|---|---|---|---|
| facebook/wav2vec2-large-xlsr-53 (current) | 317M | apache-2.0 | 1024 | 16 kHz raw | Défossez 2023 | yes |
| facebook/hubert-large-ll60k | 317M | apache-2.0 | 1024 | 16 kHz raw | Vaidya, Li, Antonello | yes. Trained on Libri-Light = LibriVox audiobooks, the same source domain as LibriBrain [I: possibly overlapping recordings, unverified] |
| microsoft/wavlm-large | 317M | **card silent** (unilm repo is MIT [K]) | 1024 | 16 kHz | Antonello | yes. Confirm the weight licence before use |
| facebook/w2v-bert-2.0 | 600M | mit | 1024 | 16 kHz → 80-mel fbank, Conformer | none found for brain | yes. 4.5M h of pretraining [S] |
| facebook/data2vec-audio-large | 313M | apache-2.0 | 1024 | 16 kHz | weak / none found | yes |
| openai/whisper-medium (current, L24) | 769M (encoder ~305M) | apache-2.0 (HF card) | 1024 | log-mel, 30 s padded window | Antonello, Goldstein, 2606.02305 (mid layers) | yes |
| openai/whisper-large-v3 encoder | 1.55B (encoder ~635M) | apache-2.0 (HF card) | 1280 | log-mel 128-bin, 30 s window | Antonello (largest = best audio) | yes (load the encoder only) |
| facebook/mms-1b | 1B | **cc-by-nc-4.0 — NC, excluded** | 1280 | 16 kHz | — | — |
| nvidia/parakeet-tdt-0.6b-v2 | 0.6B | cc-by-4.0 | — | NeMo, not transformers | none | no (NeMo) |

### Text (word type, word in context, sentence)
| Model | Params | Licence | Dim | Brain | Fit |
|---|---|---|---|---|---|
| google-t5/t5-large (current, h12) | 738M | apache-2.0 | 1024 | d'Ascoli | yes |
| google/flan-t5-large | 783M | apache-2.0 | 1024 | none direct | yes. Drop-in swap of the T5 layer |
| openai-community/gpt2-large (owner-gated) | 774M | mit | 1280 | Caucheteux & King | yes |
| Qwen/Qwen3-0.6B (mid layers) | 0.6B | apache-2.0 | 1024 | the Antonello scaling argument [I] | yes. A contextual target falls under the same owner gate as gpt2 |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | apache-2.0 | 1024 (Matryoshka [K]) | none | yes. Sentence-level |
| google/embeddinggemma-300m (v1) | 308M | **gemma terms** (not NC; gated, has a prohibited-use policy) | 768 | none | yes |
| meta-llama/Llama-3.2-1B | 1.2B | llama3.2 community licence (not NC; has restrictions) | 2048 | the Antonello family | borderline |
| facebook/SONAR (text + speech encoders) | — | **cc-by-nc-4.0 — NC, excluded** | 1024 | — | — |

### Joint audio-text (one shared space)
| Model | Params | Licence | Dim | Input | Notes |
|---|---|---|---|---|---|
| **google/embeddinggemma-2** | 740M = 270M text (130M transformer + 140M embedder) + 300M audio + 170M vision [V model card] | **apache-2.0** [V: HF API + model card; usage also falls under the Gemma Prohibited Use Policy] | 768, Matryoshka 512/256/128 [V] | mono 16 kHz, 25 tokens/s (40 ms), ≤~327 s within the 8192-token shared context [V] | See the facts below the table |
| laion/clap-htsat-unfused, laion/larger_clap_general | ~150-200M | apache-2.0 | 512 | 48 kHz, 10 s | Trained on sound-event captions. Little spoken lexical content [K]; no brain evidence |
| microsoft/msclap | — | ms-pl (permissive, not NC) | 1024 | 44.1 kHz | Same caveat as CLAP; not in transformers |

EmbeddingGemma 2 facts:
- Text, image, video and audio all map into one 768-d space; pooling is mean pooling [V].
- Released 2026-09-14 (HF repo creation date) [V].
- The audio tower is a 12-layer conformer-like stack: `model_type: gemma4_audio`, hidden 1024, conv kernel 5, chunked attention with left context 13 and **right context 0**, output projection 1536. Its tokens then run through the text backbone before pooling [V `config.json`; "streaming/causal" is our reading, I].
- Audio benchmarks: MSEB retrieval 69.54 MRR@10, MAEB mean 49.39 [V].
- Text inputs want task prefixes; audio takes none [V].
- Training data cutoff January 2025 [V].
- Loads via `AutoModel` or SentenceTransformers [V].
- The unsloth page is secondary and matches the card (it gives no Google links).
- Primary sources: https://huggingface.co/google/embeddinggemma-2 and https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2.

## 3. Licence flags
- **NC, cannot use for submission targets:** facebook/mms-1b (cc-by-nc-4.0) and facebook/SONAR (cc-by-nc-4.0, text and speech).
- **Restricted but not NC:**
  - gemma terms: embeddinggemma-300m v1, gemma-3n.
  - llama3.2 community licence.
  - Whether the rules allow these is not settled here; check `docs/competition/rules.md` before use. That check is ours; no rule line is cited.
- **Unclear:** microsoft/wavlm-large has no licence field on the HF card. Resolve before use.
- **Clean:** wav2vec2, HuBERT, data2vec (apache-2.0); w2v-BERT 2.0, gpt2 (mit); whisper, t5/flan-t5, Qwen3, CLAP (apache-2.0); **EmbeddingGemma 2 (apache-2.0)**; parakeet (cc-by-4.0).

## 4. Ranked shortlist (≤4)
Expected size [I]:
- 99y's own prior is that teacher gains are ensemble-scale, +1-2 pt.
- Four single-teacher nulls already sit inside seed noise.
- So a teacher swap is a knob unless the proxy shows a larger predictable subspace.
- Gate each arm on the proxy below; run training only for a candidate that beats the incumbent of its modality on that gate.

1. **Whisper mid layers** (whisper-medium L10-16 first, then the large-v3 encoder mid layers).
   - Mechanism: Antonello found Whisper the best audio family. 2606.02305 and Vaidya both put the best layers in the upper-middle range, and we use **L24, the last layer**.
   - Cheapest arm: the precompute code already exists in `whisper.py`; only the layer slice changes.
   - Proxy: L24 gave 12 family-wise components and 0.691/0.210/0.022. A mid layer should beat 12 and the top-5 of 0.210.
2. **HuBERT-large-ll60k** (layers ~12-18), with **w2v-BERT 2.0** as the alternative.
   - Mechanism: a masked-prediction SSL model with the strongest brain-encoding record among frame-level SSL models.
   - Libri-Light is LibriVox audiobook speech, the same domain as LibriBrain's audiobook stimuli [I]. That should matter more than xlsr-53's 53-language spread for one English narrator.
   - It is a drop-in replacement for wav2vec2 (1024 dims, 16 kHz, same `speech.py` pipeline).
   - Proxy: beat wav2vec2's 15 components and 0.142 top-5.
   - Also fix the wav2vec2 layer mismatch from §1. It is a free check of the incumbent.
3. **EmbeddingGemma 2**, text side and audio side.
   - Mechanism (a), dimension: the Matryoshka ordering packs the most information into the first 128 dims, which roughly matches the 11-24 components MEG can reach. The target's own geometry would then do what the proposed PCA-truncation arm does by hand [I].
   - Mechanism (b), one space: audio and text land in the same space, so a word-audio target and a sentence-text target share axes. The MEG embedding then serves one geometry instead of two competing ones [I].
   - Risks:
     - It is trained for retrieval on pooled clips; a 0.5 s word clip is out of distribution.
     - Speech-content fidelity is unproven (MSEB is about sound).
     - So start at sentence level, plus a word-in-sentence text target.
     - The causal audio encoder is a mild plus, since the brain is causal too [I].
   - Proxy: beat t5's official-val 14 components and 0.154 top-5 at 128d.
4. **Qwen3-0.6B mid-layer contextual word states** (Apache).
   - Mechanism: a context-free T5 word-type lexicon cannot be the right target for the in-context MEG response. Caucheteux & King and Antonello show mid-layer causal LMs map MEG/fMRI best, with gains that grow with size.
   - gpt2-large-ctx already reached 24 components vs t5's 14 (held-out), the most of any word-level teacher.
   - **Blocked by the same owner gate as gpt2-large.** The owner must lift that gate first; this ranks it, it does not authorise it.

### How to rank them cheaply with the existing proxy (`megspeech.evaluation.studies.teachers.alignment`)
- **What it computes:** ridge-CCA (ridge 0.1 × mean eigenvalue) between the trained encoder's 512-d `Tap.WORD` embedding and each teacher's per-word targets.
  - Fit on train words, read on val.
  - Counts canonical components above a 200-shuffle null, per component and family-wise.
  - Reports top-1/5/20 val correlations, target anisotropy, and the teacher's own 50-word softmax-probe BAcc@10.
- **Adding a candidate:** precompute its per-word targets into `processed/speech/<name>` (the `Directory` teacher) and pass `--teachers`. No training needed. Rank by **family-wise components, then top-5 correlation**, on **official-val**.
- **Harness caveats** (fix these before trusting a ranking):
  1. **Bias toward the teacher the encoder was trained with.** The encoder may have been trained against one of these teachers, which would favour it. Use a base encoder with no teacher loss, or add a brain-score-style ridge from teacher to the encoder input.
  2. **Audio targets exist only for TRAIN runs.** Audio teachers are therefore read in-sample. Precompute candidates for the official validation runs too, so all teachers share one denominator.
  3. **Dimension changes the count.** Compare at matched dims (PCA or Matryoshka to 128), or the count depends on dimensionality.
  4. **CCA reach shows predictable, not useful.** Also report the teacher-probe BAcc@10 restricted to the significant canonical directions. That is the direct readout of 50-word content in the reachable subspace [I].
