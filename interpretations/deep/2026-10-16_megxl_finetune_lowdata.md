# Deep: MEG-XL fine-tuned end to end against the CNN, at 5% of the training data

megspeech-6hy · 2026-10-16 · both arms `train --origin ema-cd25-xsp10sem05-ac05x-s1 --data.train_share 0.05
--optim.acoustic 0 --optim.heads 1`, s1; arm B adds `--context.encoder spanned --data.segment 9 --optim.batch 8
--optim.encoder_lr 0.1 --optim.epochs 8 --no-optim.compiled`

## Question

MEG-XL (pnpl, 26M-param criss-cross transformer over BioCodec tokens) was only tried frozen. Its paper's claim is
for the low-data regime. Fine-tune it end to end under our heads, at 5% of the training data, and compare with the
CNN trained the same way.

Pre-registered, on VAL macro bacc@10 at tau .5: B ≥ A + .015 means pretraining is a lever at low data; within ±.015 a
tie; B < A − .015 means the CNN wins even in the paper's regime. A was amended to the three-seed mean once seed 2 showed
a .03 spread.

## Result

VAL macro is the mean over domains of the context head's bacc@10 at tau .5 in the best epoch's report; neural is the
training log's VAL minus VAL with the MEG shuffled across words, at the last epoch. Local test is macro bacc@10 at the run's VAL-selected tau (.5, except CNN s3 at 0). Public is the Kaggle Deep public leaderboard, submitted 14:03 as a pair.

| | VAL macro | MOCHATIMIT | Sherlock1 | TheMoth | neural | local test | public |
|---|---|---|---|---|---|---|---|
| A: CNN s1, 40 epochs (~4 min) | .434 | .486 | .456 | .360 | +.003 | .451 | — |
| A: CNN s2 | .465 | .560 | .477 | .357 | +.004 | .481 | .412 |
| A: CNN s3 | .462 | .529 | .462 | .395 | +.013 | .448 | — |
| A mean (seed SD ~.017) | .453 | | | | | .460 | — |
| B: MEG-XL spanned, encoder lr .1, 8 epochs (~63 min) s1 | .473 | .507 | .510 | .402 | +.030 | PENDING | .544 |
| B s2 | stopped at epoch 2 (deadline) | | | | | | — |
| C: B with encoder lr .3 (instrument) s1, stopped after epoch 6 | .427 (ep 6) | | | | +.050 | — | — |

## Reading

On VAL, B s1 clears the lever bar (.473 against .468) on one seed: a pass, not yet a margin. The clearer VAL-side
signal is neural: +.030 for MEG-XL against at most +.013 for any CNN seed. At 5% of the data, VAL is mostly the context
head's language prior, which both arms share, so VAL macro compresses the difference between encoders.

Public, recordings neither arm trained on, separates them: MEG-XL .544, CNN s2 .412, +.13 on the same rows. The CNN
falls .07 from its local test to public; MEG-XL rises. A low-data CNN fits the training recordings; the pretrained
encoder carries to new ones. This is the paper's low-data claim, seen where our VAL could not show it.

Not ruled out: one MEG-XL seed and one CNN seed on public; whether the edge survives at full data (the full-data CNN
loses only ~.03 to public); other encoder learning rates (C, .3, reads more MEG: neural +.06 at epoch 5).

In the final ensemble: `d-ac05xfam16` plus this 5%-data MEG-XL at weight 1/17 scored VAL .6482 (.6474 without), local test
.7223 (.7204), public .69233 (.69483). A member scoring .544 on public alone, at that weight, neither helps nor hurts.
A MEG-XL trained on all the data is the test this did not reach.
