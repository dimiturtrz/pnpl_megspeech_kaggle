# Speech-source auxiliary on Deep: M vs XF vs H vs D vs mhub (2026-10-09)

Sense-making of our own results. Sources: `bd show megspeech-7gf.132` / `7gf.147` notes and the `stack-ema-cd25-xsp10sem05-*` run logs (`val tau 0.5` and `local test tau 0.5` lines, final fine-tune logs only).

## Scope and denominators

- Metric: macro at tau 0.5. VAL = subject-0 10% stratified split (n=2512, selects epoch). Local test = pnpl validation + test pooled (n=2133, reports). Neither is the final test.
- Unit = one fine-tune run, not one independent stem. Several fine-tunes share a speech-aux pretrain (e.g. `s2-ft-s2/s5/s7/s9`, `hub-s3-ft-s3/s13`), so n overstates independence; seed sd below mixes pretrain seed and fine-tune seed.
- Excluded: aux-only pretrain logs (tau 0, no context head: val .16-.31), `frz`/`lr10` ablations, `adapt12` Broad adapts.
- Kind key: M = masc27w (MASC multi-subject stem warm); XF = xsp10sem05 aux pretrain then fine-tune; H = same with HuBERT aux (`hub-`); D = data2vec aux (`d2v-`); mhub = masc27-warm + HuBERT aux.

## Per kind (tau 0.5)

| kind | n | mean val | mean local test | seed sd val / LT | public (submitted) |
|---|---|---|---|---|---|
| M masc27w | 11 | .6599 | .6818 | .0033 / .0151 | masc7lf16 (M s1-7 + 16 cd8lf16 members) .6678 |
| XF | 28 | .6427 | .6924 | .0056 / .012 | in blend .68883 (with H) |
| H hub | 8 | .6418 | .6980 | .0053 / .0101 | in blend .68883 (with XF) |
| D d2v | 4 | .6417 | .6970 | .0091 / .0137 | not submitted |
| mhub | 1 | .6520 | .7003 | n/a | not submitted |

- All rows parsed from the `stack-ema-cd25-*` logs. M = `masc27w-s1..s11` (`-rd10`, `mi*` variants excluded); all 24 plain seeds s1..s24 give .6600 / .6798.
- No public number exists per individual aux kind. The .68883 figure comes from the 12:16 note as a blend of XF and H members; it cannot be split by kind.

## Reading

- Val and local test rank the kinds in opposite order. M leads val by about +.017 over aux kinds, yet trails them on local test by .011-.019. XF, H, D and mhub are indistinguishable on either axis: val spread .0011, LT spread .006, against seed sd .01.
- M's val advantage is not trivially comparable: M uses a different stem (warm from MASC), and VAL selects the epoch, so M and aux kinds differ in best-epoch bias and fine-tune winner's curse by unknown amounts. A cross-protocol gap of .017 is above the .015 bar but conditional on that.
- Public ordering is the only held-out-ish ordering we have: M ensemble .6678; aux blend .68883 is +.021 higher, but it mixes kinds and member counts, so it is not a per-kind result.
- Everything within aux kinds is directional (under .015, single seed per cell). D vs H vs XF: no claim. mhub is n=1: .6520/.7003 is inside the XF/H range, so no evidence the masc warm start adds to HuBERT aux.
- Local test is not used to select, but XF's 28 runs were launched over days while watching it; treat LT ties as ties.

## Not ruled out

- Per-kind public gap: never measured; each kind needs its own blend of equal member counts to attribute .68883.
- mhub: one seed only; mhub-s2 has no result. A real mhub test needs 3+ seeds.
- d2v s5/s6 never run (skipped by claim to free the lane); D has n=4 over 4 distinct pretrains.
- Aux weight (speech 1.0, semantic .5) tried at one setting for all kinds; other weights untested.
- Broad: HuBERT aux has never been trained into a bgmix Broad stem. The Deep-member adapt (xsp s4-ft-s5, ruler .5054) was the wrong stem, so it says nothing about aux on Broad. Queued as bq63 (`ema-bgmix05-w15-hub-s{0,1}`).
- Window mismatch (1.0 s aux vs 1.5 s) not tested for aux kinds.
- M vs aux on a common protocol (same val rule, same epoch selection) is untested; the +.017 val gap could partly be protocol.
