# Phase 2 — prior recalibration: measured, and the plan's premise is WRONG

## Two errors in the plan I need to correct before spending training time

### Error 1: the plan targets a model that is not serving

The plan says to regenerate `.tmp/attention-ranking-*.jsonl` at the 3.96%
serving prior and retrain the attention reranker. But:

```
spelling.attentionMode    "OFF"
linguistic.attentionMode  "OFF"
```

**The attention reranker is not on the serving path.** The 81.8%-vs-3.96%
mismatch is real, but it belongs to a model that currently contributes nothing.

The model actually gating the lane is `recallReranker` — a 15-feature logistic
regression (`recall-pairwise-v1`, `src/data/recall-reranker.json`) consulted at
`linguistic-rules.mjs:786`, whose output `modelProbability` faces the
`realWordTypoMinProbability >= 0.95` gate. Its training prior is **30.9%**
positive (7,125 / 23,049 rows) against the same 3.96% serving prior. So a prior
mismatch does exist here too — 11x rather than 110x.

### Error 2: logit adjustment on a fixed model cannot change anything

The plan's step 4 is *"logit adjustment `logit - log(prior_train/prior_serve)`"*.
For a **fixed** model this subtracts a constant from every score. It is a
monotone transform — algebraically identical to moving the 0.95 threshold. It
**cannot** reorder candidates, so it cannot convert an FP into a TP. It slides
along the existing curve; it cannot lift it.

The trainer also **already applies balanced class weights**
(`class_w_pos = n/(2*pos)`, `train_recall_reranker.py:157-159`), so the
headline prior correction is in effect during fitting.

## The measurement that settles it

Sweeping `realWordTypoMinProbability` traces every operating point the current
model can reach. Recalibration can only land somewhere **on** this curve:

| minProb | R | P | F0.5 | TP | FP | FA/165 |
|---|---|---|---|---|---|---|
| 0.999 | 0.181 | 0.743 | 0.458 | 200 | 69 | 0 |
| 0.995 | 0.207 | 0.761 | 0.496 | 229 | 72 | 0 |
| 0.99 | 0.220 | **0.752** | 0.507 | 243 | 80 | 0 |
| **0.95 (shipped)** | **0.252** | **0.679** | **0.507** | 279 | 132 | **0** |
| 0.90 | 0.259 | 0.580 | 0.465 | 287 | 208 | 0 |
| 0.85 | 0.264 | 0.522 | 0.437 | 292 | 267 | 0 |
| 0.80 | 0.268 | 0.471 | 0.409 | 296 | 333 | 0 |
| 0.75 | 0.272 | 0.431 | 0.386 | 301 | 398 | 1 |

**Phase 2's gate is R >= 0.38 with P >= 0.70. No point on this curve comes
close.** Max recall anywhere (even at P=0.43) is 0.272. To reach R=0.38 the
model would need ~420 TPs; it produces 301 at its most permissive setting.

The curve is also **brutally steep**: from 0.95 to 0.90, buying 8 extra TPs
costs 76 extra FPs — a 9.5:1 loss rate. Precision-first forbids moving down it.

Note the shipped 0.95 is *not* the F0.5 optimum — 0.99 ties it at 0.507 with
P=0.752 instead of 0.679. That is a genuine, free precision gain available by
config alone, and the only real win recalibration offers.

## What this means

Recall is not blocked by miscalibration. It is blocked by **the candidate never
becoming rankable in the first place** — 301 TPs is the ceiling of what reaches
the ranker at any threshold. The binding constraints are upstream:

- trigram coverage 55.6% (Phase 5 — the LM binary is already built and only
  needs `src/lm/binary-backend.mjs`)
- the `ACCENTED_SAME_KEY` lane, 612/1085 dev labels, currently unreachable
  (Phase 6)
- prefilters dropping 161 + 153 gold errors (Phase 7)

This also matches the Phase 1 finding: the `ngay -> ngày` false alarm outranks
three true positives (0.638 vs 0.596/0.549/0.539) and **all four sit at
`tri=[0/0]`** — zero trigram evidence. The ranker is not miscalibrated there;
it is blind. More training data at a corrected prior does not add evidence that
the LM does not have.

## Recommendation

**Do not spend the ~20x-KEEP-rows retrain.** It optimizes a model that is OFF,
via a transform that is provably monotone, toward a gate the underlying
candidate supply cannot reach. Redirect that effort to **Phase 5** (LM density),
which is the measured upstream bottleneck and already has its expensive artifact
built.

Keep from Phase 2 the one cheap, real gain: consider `realWordTypoMinProbability
0.95 -> 0.99` for P 0.679 -> 0.752 at equal F0.5.
