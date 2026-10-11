# Frontier delta: what is new since the 10-08 sweep, aimed at our three weak points (2026-10-11)

**Question.** Since [2026-10-08_frontier_sweep.md](2026-10-08_frontier_sweep.md), what new published or public material
bears on our three measured weak points? Which of it could move public BAcc@10 by >= 1.5 pt (Deep .689, Broad .626;
leaders .775 / .732)?
- (a) Per-window MEG evidence is weak: the word head cannot separate `the` from `a`, and the fit-val gap is
  trial-specific MEG memorisation.
- (b) Span-final words lose ~9 pt.
- (c) Our Deep-Broad gap is ~6 pt; the leaders' is ~4.

**Scope.**
- Web search and reading only, 2026-10-11. About 20 sources opened. No compute was run.
- Numbers are as each source reports them. Abstract-only reads are marked [abstract]; a claim I could not confirm is
  marked [unverified].
- This is a **delta**. Every source cited in the 10-04 / 10-06 / 10-07 / 10-08 / 10-09 docs is excluded, unless it is
  re-read here for a fact those docs did not extract.
- Already closed and not re-recommended:
  - HDND pyramid, smooth top-k, conformer, MEG-XL frozen/LoRA, trial-averaging aug, band-stop/specaug.
  - wav2vec2 / GPT-2 / semantic aux, BERT MLM fusion.
  - TTA and cross-subject test pooling (both forbidden).

**Clock.** The libribrain.com participate page says: "Submissions open 15 July 2026 and close **15 October 2026
(AoE)**". That leaves about 4 days. Any lever that costs more than about a day of wall-clock is priced out of this
competition, whatever its odds.

**Headline.**
- **Still no competitor write-up or code.** The leaderboard is public; the write-ups will come after the deadline.
- **The one new hard fact is on the leaderboard:**
  - "Stéphane d'Ascoli" sits 9th on Deep at **.7309**.
  - We are 27th of 86 at .6888.
  - The author of the d'Ascoli 2025 recipe scores 4.2 pt above us. Our "faithful replica" (7gf.51) is the same model
    class.
  - That gap is the cheapest diagnostic on the table (#1).
- **New literature since 10-08 adds no method for (a) or (b).**
- **For (c), it adds two cross-subject recipes (CPSD / SICMD).** Their components map onto things we have, except
  CorrCA over same-stimulus subjects.
- **Two 2026 benchmarks put a number on our SSL / foundation-model nulls:** general pretraining "only marginally"
  helps.

## Ranked levers

| # | Lever | Mechanism | Evidence (number, dataset) | Expected on our task | Cost | Licence | Our status |
|---|---|---|---|---|---|---|---|
| 1 | **Diff our d'Ascoli replica against the organisers' official port** | A published recipe, run by its own author, reaches .7309 on Deep public. Our replica of the same architecture family sits below that. So either his run uses something we lack, or our replica deviates. Candidates for "something we lack": Meta-internal 723-participant pretraining, the T5-large mid-layer target with SigLIP, or a larger context. The organisers ship a LibriBrain100 port: `neural-processing-lab/dascoli-word-decoding`, "adapted for the 2026 PNPL LibriBrain100 competition". Its stated recipe is "SimpleConvTimeAgg plus Transformer", t5-large targets, a "SigLIP-style contrastive loss", and retrieval scoring. | [libribrain.com 2026 leaderboard](https://libribrain.com/editions/2026/leaderboard/): d'Ascoli .7309 (rank 9). Top 10 spans .7298 to .7747. "Brain & AI" .7747 and "Munich Logic Lab" .7651 lead. The repo README reports **no** competition score. What d'Ascoli's entry actually runs is [unverified]. | **Instrument.** If the official port, run as-is on our ruler, beats our replica by more than 1.5 pt, that difference is a replication bug we can merge. If it scores below ours, the .73 rests on non-public assets, and the lever is closed. Odds of a mergeable >=1.5 pt: ~20%. Our 7gf.51 replica already reached .6408, and our public .689 is only 4 pt short. | 1 training run of the repo (~3-6 h GPU) plus an eval on our ruler. Fits the 4-day clock. | Repo licence [unverified]. The upstream `facebookresearch/brainmagick` is **CC-BY-NC 4.0**, which is not OSI, and is archived (2024-03-13). Reading the port for a recipe diff is fine; vendoring its code is not. | Never diffed. 7gf.51 ("faithful replica") was built from the paper, not from this repo. |
| 2 | **Per-track LM weight in the span beam already in flight (lvj)** | Neural-constrained beam: per slot, take the top-k neural candidates and score neural logprob + α·LM next-word logprob. **The best α is larger when the neural evidence is weaker.** | Maghsoudi, Kankanala, Shamma, Ganapathy, [arXiv 2609.31997](https://arxiv.org/html/2609.31997). 13 subjects, two ~27 s poems, closed vocabulary. Beam B=3, k=10, GPT-2 with SmolLM as a check. Listened top-1 14.17 / top-5 46.96; imagined 7.05 / 31.14. The paper reports α* qualitatively: "higher for imagined". I could not extract a no-LM row. | Refines 10-08 #1; it is not a new lever. Prediction: fit α per track on VAL, because Broad (weaker MEG) should take a larger α than Deep. Toy-scale evidence (two poems), so the size is not transferable. **0 to +1.5 pt Broad, ~0 to +1 Deep.** It does nothing for span-final words: the beam is causal, and the last slot gets no right context. | Within lvj: one α sweep per track on VAL (CPU). | GPT-2 MIT; SmolLM/SmolLM2 Apache-2.0. | lvj open, owner-gated. |
| 3 | **CorrCA on same-stimulus source subjects, then a stimulus-locked subject map (Broad)** | Two stages: (1) pretrain on source subjects, with CorrCA keeping only the components that correlate across subjects hearing the same audio; (2) "personal specialisation", a short fine-tune on the target. PESA, a positional-encoding spatial attention, remaps sensors into a reference space. The loss is CLIP against wav2vec2. | CPSD, Zhang et al., [arXiv 2608.22420](https://arxiv.org/html/2608.22420), Armeni 2022 top-10 (no vocabulary size given): intra-only base 17.4; +PESA 25.7; +specialisation 50.3; both 61.3. SICMD, Zhang & Chen, [arXiv 2609.30832](https://arxiv.org/html/2609.30832), SMN4Lang (12 subjects, MEG+fMRI) leave-one-subject-out top-10: CPSD 71.9 → 82.7 with CorrCA plus fMRI fusion. The fMRI half is not available to us. | Most of the ablation is something we already have: specialisation = our sub-0 pretrain + per-subject fine-tune. PESA = spatial attention + subject layer, null for us (7gf.55). The **only untried piece is CorrCA** over subjects 1-32 on shared stimulus. It needs the stimulus-overlap check that 10-08 #5 already asks for. Same mechanism class as ct5 (cross-subject same-stimulus contrastive aux, commit f35e23a), which is already in the tree. **Broad 0 to +1.5 pt, Deep 0.** | ~0.5 day closed-form CorrCA plus a fine-tune, after the overlap check. | No code released (either paper). | ct5 landed. CorrCA not tried. |
| 4 | **Global + per-subject low-rank adapters, or a frozen linear map + a small residual adapter, for 10-40 min subjects** | Split the fine-tune update into a shared adapter trained across all subjects plus a per-subject adapter. Or (MindAdapter) keep a frozen linear alignment and fit only a light nonlinear residual on "a few shared stimuli". Either way, a few-minute subject fits few parameters and so cannot memorise trials. | Stacked LoRA, [arXiv 2607.03094](https://arxiv.org/abs/2607.03094) [abstract]: motor-imagery EEG, "best accuracy in the large majority" of combinations. No number extracted. Paper CC-BY-NC-ND. MindAdapter, [arXiv 2605.24679](https://arxiv.org/abs/2605.24679) [abstract]: fMRI to image, NSD, KDD 2026. | ~0 to +1 pt Broad. Our subject layer is already the linear half, and neither source is speech MEG. Worth it only if the Broad fine-tune shows the trial memorisation from weak point (a) at the 10-minute budget. | ~0.5 day. | No code found. | Subject layer in tree. No adapter split. |
| 5 | **Negative evidence: general-purpose brain foundation models** | — | [arXiv 2607.27268](https://arxiv.org/abs/2607.27268) [abstract]: LaBraM and EEGMamba on overt and imagined speech, "no consistent advantage over a 16K-parameter CNN". NeuralBench, [arXiv 2605.08495](https://arxiv.org/abs/2605.08495) [abstract], Banville, d'Ascoli et al. (Meta): "current foundation models only marginally outperform task-specific models". | Supports closing the SSL / foundation-model axis (7gf.109.4, 7gf.57, MEG-XL). **Do not spend the remaining days here.** | — | 2607.27268 CC-BY-NC-SA. | Closed. Now externally corroborated. |
| 6 | Weak point (a): single-trial SNR for function words | Searched for methods that raise single-trial evidence for articles and determiners, or other short unstressed words, in MEG/EEG. | **Nothing found.** The only adjacent idea is phonetic-feature (manner / voicing) targets for rare phonemes, which is already in the 10-06 literature (2604.04129 / MEGConformer family). | No literature lever. Our own 7gf.106 result stands: the per-window `the`/`a` margin is ~0.01, and timing alone carries all but 2-3 pt of article recall. Unlike a MEG trick, the a/an rule from the next word's onset (7gf.106, n=34) is a **text-side** fix. | — | — | Open as our own finding only. |
| 7 | Weak point (b): span-final words | Searched for right-context or lookahead handling in brain-to-word decoders. | **Nothing found.** Every LM-fusion paper found (2609.31997, 2505.13446) is causal and gives the last slot no right context. | No literature lever. The deficit is the context head's alone (7gf.106: last −0.72 context, ~0 word head). So any fix is on the context side, e.g. training-time span cuts that match the test's position distribution. That is our design question, not prior art. | — | — | Open. |

## Reading

- **The leaderboard is the most informative new source.** A 2025 method's author sits at .731, and the top 10 is
  bunched between .730 and .775. A d'Ascoli-class model, run well, is therefore worth ~4 pt more than ours on Deep.
  - Our 7gf.51 replica was rebuilt from the paper, not from the organisers' port.
  - Diffing the two (#1) is the one item here that is both cheap and sized at the bar.
  - It is an instrument first: it either finds a replication bug or shows that the .73 rests on assets we cannot use.
- **For (c), new evidence reduces to one untried piece: CorrCA (#3).** It is gated on the same stimulus-overlap check
  as 10-08 #5. Per CPSD's ablation, the large published cross-subject gain is pretrain-then-specialise (17 → 50),
  which we already do.
- **For (a) and (b), the 2025-2026 literature has nothing.** Both stay as our own findings: the a/an next-word rule,
  and span-final context. The LM beam (#2) helps neither: it is causal, and per 7gf.106 the article decision is
  already carried by context.
- **Pricing against the 15 October close.**
  - #1 (hours, ~20%) and #2 (CPU sweep inside lvj, once owner-cleared) fit the clock.
  - #3 and #4 are about half a day each, with odds below the bar.
  - #5-#7 are not runs.

## Sources

- <https://libribrain.com/editions/2026/leaderboard/> (Deep top 30; Broad table not rendered in the fetch)
- <https://libribrain.com/editions/2026/participate/> (deadline)
- <https://github.com/neural-processing-lab> (repo list, update dates) ·
  <https://github.com/neural-processing-lab/dascoli-word-decoding>
- <https://github.com/facebookresearch/brainmagick> (CC-BY-NC 4.0, archived)
- <https://arxiv.org/html/2609.31997>
- <https://arxiv.org/html/2608.22420>
- <https://arxiv.org/html/2609.30832>
- <https://arxiv.org/abs/2607.03094> [abstract]
- <https://arxiv.org/abs/2605.24679> [abstract]
- <https://arxiv.org/abs/2607.27268> [abstract]
- <https://arxiv.org/abs/2605.08495> [abstract]
- Opened and not relevant: <https://arxiv.org/abs/2606.19081> (fMRI, linear contrastive decoders) ·
  <https://arxiv.org/abs/2506.03214> (multilingual semantic-space decoding, no numbers in abstract) ·
  <https://libribrain.com/editions/2025/workshop-papers/> (2025 phoneme-track PDFs, not read)
