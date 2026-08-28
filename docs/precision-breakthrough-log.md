# Precision breakthrough experiment log

All VSEC numbers below use the position-aware scorer: an issue must fall
inside the whitespace-delimited source syllable annotated by VSEC. The held-out
split has not been opened during these experiments.

| Experiment | Dev result | Decision |
| --- | --- | --- |
| Current engine baseline | 498 TP, 105 FP; P=82.59%, R=44.66% | Reference |
| Real-word candidate cap 12 -> 24 | 510 TP, 116 FP; P=81.47%, R=45.74% | Reject: adds FP nearly one-for-one |
| Open spelling tokens of length 2 | 71 TP, 35 FP in the short-token slice | Reject: no safe confidence sub-bucket |
| PMD relative-context proof for every plain token | Output unchanged | Reject: already satisfied by emitted rows |
| Sentence-level correction budget | 65/98 FP rows contain no TP | Reject: cannot use another correction as proof |
| Exact train typo pair on current outputs | 364 TP, 34 FP; P=91.46%, R=32.65% | Evidence only: recall too low |
| Exact train typo pair + short-token opening | 417 TP, 38 FP; P=91.65%, R=37.40% | Evidence only: recall too low |
| Typo source has one target, >=2 train errors, never correct in train | 192 TP, 8 FP when scanned independently | Promising precision channel; union adds 70 TP and 7 FP but baseline FP remains |
| Error-rate channel (>=2 errors and error/correct >=2) | 222 TP, 22 FP when scanned independently | Evidence only: insufficient coverage |
| Channel reranks existing engine suggestion | 3-5 suggestions changed; metric unchanged | Reject |
| Offline error-channel verifier (train emission features -> dev) | At cut 0.4: 364 TP / 28 FP; P=92.86%, R=32.65% | Precision core only; test union with independent channel |
| Verifier + broad error-rate direct channel | P=55-64%, R=31-33% after corrected row/span dedupe | Reject: direct channel remains unsafe once sources also seen correct are admitted |
| Safe train error channel: unique target, >=2 error observations, never correct in VSEC train | 193 TP / 7 FP when emitted directly; P=96.50% | Strong independent evidence, but insufficient standalone coverage |
| Proven existing-pair core + safe channel | 440 TP / 41 FP; P=91.48%, R=39.46% | Precision passes, recall below baseline; reject as final policy |
| Same core with min token length 2 | 482 TP / 45 FP; P=91.46%, R=43.23% | Still below baseline recall; no production change |
| Add one-observation same-accent-key channel (len >=3) | 494 TP / 50 FP; P=90.81%, R=44.30% | Near target but 4 TP short; do not tune a micro-bucket on dev |
| One-observation same-accent-key channel (len >=2) | 495 TP / 51 FP; P=90.66%, R=44.39% | Still below baseline recall; reject |
| Hybrid: pair-proven emissions + direct error channel + confidence rescue >=0.97 | 508 TP / 53 FP; P=90.55%, R=45.56% | Dev gate passes; require OOF before any implementation |
| Same hybrid, confidence rescue >=0.98 | 503 TP / 53 FP; P=90.47%, R=45.11% | Dev gate passes; OOF check required |
| Same hybrid, confidence rescue >=0.99 | 499 TP / 52 FP; P=90.56%, R=44.75% | Dev gate passes narrowly; OOF check required |

## Funnel evidence

- `realWordLaneDeclined`: 264 labels; 223 gold candidates in the pool; 221
  have quick n-gram attestation; 163 are evaluated before the cap.
- Cap expansion added 12 TP and 11 FP, so pool depth is not a separable
  precision improvement.
- Current linguistic emissions split into spelling 388 TP / 68 FP and PMD
  110 TP / 37 FP. PMD emissions are all `UNACCENTED_VALID`.

## Next hypothesis

Scope was expanded to permit a validated error-channel verifier. The first
evidence remains train-derived and leakage-safe: it is not enough to recover
the 4+ additional labels required to beat the current 498/105 baseline while
holding P>90%. The clean training corpus adds no independent discrimination
for the remaining one-observation sources (all are absent there), so it must
not be used as a manufactured proof.

## Cross-fold validation of the hybrid candidate

Five-fold source-disjoint VSEC-train validation (each fold builds its error
channel only from the other four folds) prevents the direct channel from
crediting its own training examples. The hybrid at confidence 0.97/0.98/0.99
has aggregate P=90.15%/90.37%/90.49%, respectively, but one fold remains
below the 90% precision gate: 89.57%/89.53%/89.68%. At 0.99 that fold's FP
split is pair core 42, direct channel 43, confidence rescue 3. The direct
source is dictionary-valid for 42 of its 43 FPs (and 313 of 335 TPs), so a
dictionary exclusion destroys recall rather than proving a safe separation.

Decision: **do not implement the hybrid yet**. It is the first dev policy to
raise recall and precision together with OOF aggregate precision above 90%,
but fold-level instability is concrete counter-evidence, not a detail to
average away.

## Context-proven direct channel (new candidate)

The unstable direct FPs were valid dictionary words in their local context.
Requiring candidate LM attestation windows to exceed the original's fixes that
root cause rather than changing a global threshold:

- direct lane on dev: **198 TP / 4 FP** (P=98.02%);
- hybrid with min token length 2 and confidence rescue >=0.96: **499 TP / 47
  FP**, P=91.39%, R=44.75% (strict position scorer);
- source-disjoint 5-fold VSEC-train: **3,788 TP / 354 FP**, P=91.45%, R=42.25%;
  every fold is P=91.15%-91.95%.

This is now a candidate for implementation, not yet a product result: it
still needs a generated train-only error-channel artifact, production parity,
clean-SMS/latency gates, full regression, and one frozen held-out run.

## Recall-first runtime profile (saved)

`recall-first-v1` is the current artifact-backed runtime policy: pair-proof
emissions, confidence rescue, direct train-error corrections with LM-context
proof, plus deterministic non-dictionary typo preservation. On the strict dev
scorer it records **556 TP / 38 FP, P=89.68%, R=49.87%**. It is explicitly a
recall-first profile and does not satisfy the strict P>90% release gate.

Improvement target: eliminate at least 1-2 of these 38 context-explained FPs
without losing labels; do not raise a global threshold or remove predictions.

## Pair-count calibration breakthrough

Pair-proof candidates seen exactly twice in VSEC train are a separable weak
bucket: 48 TP / 9 FP (P=84.2%), while counts 1, 3, 4, and 5+ are all markedly
cleaner. Rejecting only `pairCount === 2` is not a global confidence change.
Runtime dev result: **521 TP / 56 FP, P=90.29%, R=46.73%**. This exceeds the
strict baseline in both precision and recall.

## Frozen acceptance candidate

The frozen `precision-over-90-v1` profile is recorded in
`config/precision-over-90-profile.json`. Before the held-out run it passed:

- clean SMS: **0/165** rows with linguistic false alarms;
- SMS length <=160: p95 **24.04ms** (limit subsequently authorized at 100ms);
- core/rules/scope tests: **65 pass, 0 fail, 2 known skips**.

## Current separability limit

The high-precision verifier and the typo channel both identify source forms
that are empirically error-only or strongly error-skewed. Their union cannot
recover the remaining recall because those remaining source forms are also
annotated as correct in train. Admitting them through the current LM/context
features reintroduces hundreds of false positives. A further improvement needs
new independent evidence (for example a validated domain error channel), not
another threshold or n-gram cap sweep.

## Held-out result (single post-freeze run)

`precision-over-90-v1` did **not** generalize to the release gate. Strict
position-based semantic scoring yielded **481 TP / 71 FP, P=87.14%, R=42.91%**.
Recall remains above the earlier held-out baseline (39.7%), but precision is
below 90%; the profile is therefore marked rejected and must not be promoted.
The next iteration must return to train/dev-only evidence and identify which
pair-proof frequency buckets account for the 15 excess held-out FPs.

## Unknown-token context-proof candidate (dev-only)

Funnel instrumentation (`tools/analyze_emitted_funnel.mjs`) showed that the
largest held-out FP source was not the direct error channel: 27 were ordinary
unknown-token spelling emissions. On dev, the subset with no train pair proof
separates exactly by context proof: candidate-window wins = 63 TP / 12 FP;
not candidate-window wins = **0 TP / 7 FP**. The runtime now rejects only the
latter, except an exact Telex tone-suffix relation (direct input evidence,
such as `hangf`). Dev-only result: **522 TP / 49 FP, P=91.42%, R=46.82%**.
This is a candidate only: the existing held-out split has already been
consumed by the prior experiment and cannot validate this change.

## OOF confirmation: direct-lane context gap

Five message-disjoint folds over `vsec-train` rebuild the entire error channel
from the other four folds. Low-count (1–2) direct pairs separate by the
candidate-minus-original context-window gap: gap 1 was **32 TP / 48 FP**;
gap >=2 was **116 TP / 23 FP**. Requiring the latter for those pairs gives
every fold P>90% and aggregate **3,789 TP / 387 FP, P=90.73%, R=42.26%**.
This is a local lane-evidence rule, not a global threshold. Dev remains
**522 TP / 49 FP, P=91.42%, R=46.82%**; core/rules/scope tests pass.

## Next recall hypothesis: different-key shortlist

Train-only candidate audit confirms this is a coverage opportunity, not yet a
safe feature: for DIFFERENT_KEY_REAL_WORD labels, the diverse shortlist keeps
1,006/2,574 targets at K=4, 1,211 at K=6, and 1,423 at K=8. Any activation
must prove a context-ranking separation in OOF; larger K alone is explicitly
not a recall claim.

OOF replay at K=6 confirms that warning: precision rises to **91.25%**, but
recall falls to **42.11%** (from 42.26% at K=4). K=6 is rejected; the wider
candidate pool changes ranking rather than recovering correct targets.
