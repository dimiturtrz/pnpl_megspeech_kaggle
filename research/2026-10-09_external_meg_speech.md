# External MEG speech-listening data not yet pooled (2026-10-09)

**Question.** Pooling external English MEG into the Broad stem is the one lever that moved a single model by more than
noise: all-w15 .594 against the w1.0 mean of .577. Which public MEG datasets of naturalistic spoken-language
listening could we still add?

**Already pooled or converted:**
- LibriBrain subject 0.
- MEG-MASC (`megspeech/data/raw/sources/meg_masc.py`).
- Appleseed ds007870 (`appleseed.py`).
- Kymata-SOTO (`kymata_soto.py` + `kymata_soto_sync.py`).
- MEG-SCANS ds006468 (`meg_scans.py`, German, converted under megspeech-u2m.3).

**Scope.**
- Desk research only. Nothing was downloaded.
- Sizes come from the OpenNeuro metadata API (`snapshot.size`) or the data papers.
- A cell marked *unverified* was not confirmed in a primary source.
- Existing issues: megspeech-u2m (parent), u2m.2 (Armeni), u2m.4 (SMN4Lang), u2m.5 (MOUS).

## Prior that gates every non-English set
- **MEG-SCANS result.** Pooling German into the stem gave a null result: allde-ac03-s0 val .6311 vs all-ac03-s0 .6271,
  +.004, one seed (bq45, u2m.3 notes).
- **Why foreign words add less.** Foreign words are out-of-vocabulary, so they are IGNORE for the word and context
  heads. They train the encoder only, through the acoustic/speech aux.
- **What carried the pooling gain.** The gain at .594 came from *English, labelled* pooling.
- **Pricing rule.** A non-English set is priced from the German +.004, not from the English +.017. The one open
  question is dose. MEG-SCANS is ~16 h; the sets below are 4-6x larger.

## Datasets

| | Le Petit Prince MEG (ds007523) | SMN4Lang (ds004078) | Armeni 2022 | MOUS (Schoffelen 2019) | Broderick 2018 |
|---|---|---|---|---|---|
| licence | CC0 / PDDL | CC0 | Donders DUA RU-DI-HD-1.0 | Donders DUA RU-DI-HD-1.0 | (EEG; not checked) |
| OSI/CC-compatible | **yes** | **yes** | **no** | **no** | n/a |
| subjects x listening | 58 x ~1.6 h = **~94 h** | 12 x 6 h = **~72 h** | 3 x ~10 h = **~30 h** | 102 auditory subjects, sentences + word lists; total hours *unverified* | out of scope |
| sensors | MEGIN TRIUX 306 (102 mag + 204 grad), 1000 Hz, raw + cal/crosstalk files | MEGIN TRIUX 306, 1000 Hz, raw `.fif` per run (+ tSSS/ICA derivative) | CTF 275 axial gradiometers, 1200 Hz | CTF 275 axial gradiometers, 1200 Hz | EEG |
| vs LibriBrain MEGIN 306 | same family: no field mapping | same family: no field mapping | cross-scanner: field mapping as for MASC/Appleseed | cross-scanner: field mapping | n/a |
| language | French | Mandarin | **English** | Dutch | English |
| stimulus | audiobook, 9 parts x ~10m50s | 60 stories | 10 Sherlock Holmes stories | isolated sentences (9-15 words) + scrambled lists | |
| word onsets ship? | **yes**: per-run `events.tsv` (`onset duration trial_type stimulus`, `Word` rows, ~1850 in run 1) | **yes**: `derivatives/annotations/time_align`, Kaldi GMM-HMM char + word on/offsets in story audio | **yes**: Penn Forced Aligner, TextGrid per story | **partial**: only first and target word onsets, set manually in Praat | |
| size / access | 478 GB snapshot (meg + mri), open | 678 GB snapshot (meg + mri + dwi); **MEG raw already on disk, 304 GB** in `raw/smn4lang` | DUA: owner login + click-through; size *unverified* | DUA: owner login + click-through; size *unverified* | |
| ingest effort | **S**: `meg_scans.py` clone (SSS with the shipped cal/crosstalk, word events from `events.tsv`, French prefix -> IGNORE). No audio clock fit: the onsets are already on the MEG clock. Head-position (cHPI) presence *unverified*. | **S-M**: Neuromag path as `scans.py`; alignments are relative to story audio, so a run-start offset from the triggers is needed (per-run `events.tsv` not inspected). Mandarin -> IGNORE. | **M + owner step**: CTF-275 reader + field map (the `masc.py` pattern), TextGrid via `textgrid.py`, plus excluding stories that overlap LibriBrain's Sherlock val/local test | **M + owner step**: as Armeni, plus the sentence/list task structure; few word labels | -- |

## Licence lines, quoted
- **Le Petit Prince (ds007523).**
  - `dataset_description.json`: `"License": "CC0"`.
  - The repo `LICENSE` file: "Public Domain Dedication and License (PDDL) ... allow you to freely share, modify, and
    use this work for any purpose and without any restrictions."
  - [OpenNeuro ds007523](https://openneuro.org/datasets/ds007523/versions/1.0.1),
    [GitHub mirror](https://github.com/OpenNeuroDatasets/ds007523).
  - Authors Bel, Bonnaire, Pallier, King (2025).
- **SMN4Lang (ds004078).**
  - `dataset_description.json`: `"License": "CC0"`.
  - Data paper: "Creative Commons Attribution 4.0 International License".
  - [OpenNeuro ds004078](https://github.com/OpenNeuroDatasets/ds004078);
    [Wang et al. 2022, Sci Data](https://www.nature.com/articles/s41597-022-01708-5)
    ([PMC9525723](https://pmc.ncbi.nlm.nih.gov/articles/PMC9525723/)).
- **Armeni 2022.**
  - Data paper: "Data use agreement for identifiable human data (version RU-DI-HD-1.0)". The paper's own CC-BY 4.0
    covers the article, not the data.
  - [Armeni et al. 2022, Sci Data](https://www.nature.com/articles/s41597-022-01382-7)
    ([PMC9177538](https://pmc.ncbi.nlm.nih.gov/articles/PMC9177538/));
    [Donders collection DSC_3011085.05_995](https://data.ru.nl/collections/di/dccn/DSC_3011085.05_995).
- **MOUS.**
  - Same DUA. Access needs "registration of the researcher" and a click-through after login.
  - [Schoffelen et al. 2019, Sci Data](https://www.nature.com/articles/s41597-019-0020-y)
    ([Europe PMC full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6472396/fullTextXML)).
- **RU-DI-HD-1.0 clause.** "I will not redistribute or share the data with others, including individuals in my
  research group, unless they have independently applied and been granted access to this data."
  - [DUA page](https://data.ru.nl/doc/dua/RU-DI-HD-1.0.html). The page returned 404 to the fetcher, so this quote is
    from the search-index copy.
  - Whether a trained checkpoint counts as "the data" for the competition's checkpoint-verification clause (rules.md
    "runable model checkpoints") is an owner reading of the DUA. It is not settled here.
- **Competition rules.** rules.md: "Not addressed: external data, pretrained models, ...". The licence bar above is
  ours, not the organisers'.

Tooling note: the [pnpl library](https://github.com/DJayalath/pnpl) README lists loaders for `Armeni2022` and
`Schoffelen2019` (Radboud WebDAV, credentials required) and `Pallier2025` (OpenNeuro ds007523). Its event parsing is a
reference for the converters. I did not check whether our pinned `pnpl>=0.2.0` ships those loaders.

## Ranking: expected value / effort

1. **Le Petit Prince MEG (ds007523).**
   - Cheapest valid add: open, CC0, same scanner family, word onsets already on the MEG clock, and an `meg_scans.py`-sized
     converter.
   - 58 subjects is the most subject diversity on offer for the cross-subject Broad stem.
   - Value is capped by the German +.004 prior. It is worth one run only as the **dose instrument**: about 6x
     MEG-SCANS' hours tells us whether non-English pooling scales or is simply dead.
   - Cost: a 478 GB fetch, which I have not started and which needs owner sign-off.
2. **SMN4Lang (ds004078).**
   - The 304 GB of MEG raw is already on disk, so it has no download cost.
   - Same scanner family; alignments ship.
   - Ranked below LPP: 12 subjects, a language further from English, and a run-offset step the alignments need.
   - It answers the same dose question. Run it only if LPP is not fetched, or after LPP shows a dose effect.
3. **Armeni 2022.**
   - The only remaining **English** set, and the deepest per subject (~10 h each). It is the kind of data that produced
     the +.017, so its expected value per hour is the highest here.
   - It fails our OSI/CC bar (RU-DI-HD-1.0). It needs the owner's DUA signature and a ruling on checkpoint sharing.
     Its Sherlock stories must be checked for text overlap with LibriBrain val/local test.
   - Owner-gated, not agent-actionable (u2m.2).
4. **MOUS.**
   - DUA-gated, Dutch, CTF.
   - Isolated sentences, not narrative.
   - Only first and target word onsets ship.
   - The highest effort for the lowest label yield. Stays parked (u2m.5).
5. **Broderick.** EEG, out of scope for an MEG stem.

**No 2024-2026 MEG listening release beyond these turned up.** MEG-SCANS (2025) and Appleseed (2026) are already in
the tree. The search covered OpenNeuro's MEG listing and Sci Data, and was not exhaustive.

## What this does NOT establish
- **No measured gain for any set here.** The ranking prices value from two of our own results: English pooling +.017,
  German +.004 (single seed).
- **LPP head-position and SMN4Lang run timing.** Neither was inspected. Both affect the converter, not the ranking.
- **Armeni and MOUS sizes, and MOUS listening hours.** Not confirmed from a primary source.
