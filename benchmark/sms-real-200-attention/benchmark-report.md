# Tiny Attention Transformer — SMS-REAL-200

Run commit: b996b58b1311f90df915a729b331118c55c2e490; dataset records: **200**; Node: v24.13.0.

This is a direct test-set evaluation. No training, fine-tuning, artifact regeneration, vocabulary/LM/dictionary changes, candidate-generation changes, shortlist changes, or threshold tuning were performed. ACTIVE overrides only the serving mode to EXPERIMENTAL_ACTIVE; frozen threshold values remain unchanged.

## Current artifact

| Field | Value |
| --- | --- |
| Architecture | A — Pre-LN Transformer self-attention encoder with shared option scorer |
| Blocks | 1 |
| Hidden dim | 48 |
| Heads / head dim | 2 / 24 |
| FFN dim | 96 |
| Max context | 32 |
| Embedding vocab | 787 |
| Attention word-vocab entries | 779 |
| Shortlist K | 8 |
| Binary size | 288688 bytes |
| Binary SHA-256 | 44c44a3ea756c87f11ea896d53aedc075a09de7298e590e3748812a792a5bd39 |
| Metadata SHA-256 | 32fb4a50c9e4ac1cd4030fb373d51b74bf9918882ecc9b2cbb1a49f75763ad7f |

## Compare table

| Metric | OFF | SHADOW | ACTIVE | ACTIVE delta vs OFF |
| --- | ---: | ---: | ---: | ---: |
| Precision | 0.994253 | 0.994253 | 0.878049 | -0.116204 |
| Recall | 0.659048 | 0.659048 | 0.274286 | -0.384762 |
| F1 | 0.792669 | 0.792669 | 0.417997 | -0.374672 |
| E2E correction recall | 0.638095 | 0.638095 | 0.253333 | -0.384762 |
| Clean FP messages | 1 | 1 | 4 | 3 |
| Protected violations | 0 | 0 | 0 | 0 |
| Candidate pool oracle | N/A | 0.994286 | 0.994286 | N/A |
| Shortlist@8 oracle | N/A | 0.994286 | 0.994286 | N/A |
| Transformer top1 given gold@8 | N/A | 0.668582 | 0.668582 | N/A |
| p50 latency | 8.6654 | 59.5481 | 33.63 | 24.9646 |
| p95 latency | 19.9357 | 103.6385 | 78.9082 | 58.9725 |
| p99 latency | 24.5686 | 120.9467 | 100.2123 | 75.6437 |
| attention calls/msg | 0 | 9.62 | 9.645 | 9.645 |

## Category breakdown

| Category | Arm | GT errors | TP | FP | FN | Precision | Recall | F1 | Corrected | E2E recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CLEAN | OFF | 0 | 0 | 1 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% |
| ALL_UNACCENTED | OFF | 375 | 255 | 0 | 120 | 100.00% | 68.00% | 80.95% | 248 | 66.13% |
| SOME_UNACCENTED | OFF | 60 | 39 | 0 | 21 | 100.00% | 65.00% | 78.79% | 38 | 63.33% |
| WRONG_DIACRITIC | OFF | 25 | 2 | 1 | 23 | 66.67% | 8.00% | 14.29% | 2 | 8.00% |
| TYPO_TELEX | OFF | 25 | 21 | 0 | 4 | 100.00% | 84.00% | 91.30% | 20 | 80.00% |
| PROTECTED_ENTITY | OFF | 20 | 16 | 0 | 4 | 100.00% | 80.00% | 88.89% | 16 | 80.00% |
| CODE_SWITCH | OFF | 10 | 9 | 0 | 1 | 100.00% | 90.00% | 94.74% | 9 | 90.00% |
| ADVERSARIAL_MINIMAL_PAIR | OFF | 10 | 4 | 0 | 6 | 100.00% | 40.00% | 57.14% | 2 | 20.00% |
| CLEAN | SHADOW | 0 | 0 | 1 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% |
| ALL_UNACCENTED | SHADOW | 375 | 255 | 0 | 120 | 100.00% | 68.00% | 80.95% | 248 | 66.13% |
| SOME_UNACCENTED | SHADOW | 60 | 39 | 0 | 21 | 100.00% | 65.00% | 78.79% | 38 | 63.33% |
| WRONG_DIACRITIC | SHADOW | 25 | 2 | 1 | 23 | 66.67% | 8.00% | 14.29% | 2 | 8.00% |
| TYPO_TELEX | SHADOW | 25 | 21 | 0 | 4 | 100.00% | 84.00% | 91.30% | 20 | 80.00% |
| PROTECTED_ENTITY | SHADOW | 20 | 16 | 0 | 4 | 100.00% | 80.00% | 88.89% | 16 | 80.00% |
| CODE_SWITCH | SHADOW | 10 | 9 | 0 | 1 | 100.00% | 90.00% | 94.74% | 9 | 90.00% |
| ADVERSARIAL_MINIMAL_PAIR | SHADOW | 10 | 4 | 0 | 6 | 100.00% | 40.00% | 57.14% | 2 | 20.00% |
| CLEAN | ACTIVE | 0 | 0 | 4 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% |
| ALL_UNACCENTED | ACTIVE | 375 | 75 | 1 | 300 | 98.68% | 20.00% | 33.26% | 67 | 17.87% |
| SOME_UNACCENTED | ACTIVE | 60 | 22 | 10 | 38 | 68.75% | 36.67% | 47.83% | 21 | 35.00% |
| WRONG_DIACRITIC | ACTIVE | 25 | 5 | 3 | 20 | 62.50% | 20.00% | 30.30% | 5 | 20.00% |
| TYPO_TELEX | ACTIVE | 25 | 18 | 2 | 7 | 90.00% | 72.00% | 80.00% | 17 | 68.00% |
| PROTECTED_ENTITY | ACTIVE | 20 | 12 | 0 | 8 | 100.00% | 60.00% | 75.00% | 12 | 60.00% |
| CODE_SWITCH | ACTIVE | 10 | 8 | 0 | 2 | 100.00% | 80.00% | 88.89% | 8 | 80.00% |
| ADVERSARIAL_MINIMAL_PAIR | ACTIVE | 10 | 4 | 0 | 6 | 100.00% | 40.00% | 57.14% | 3 | 30.00% |

## Required answers

### 1. Does Tiny Attention improve SMS-REAL-200?

ACTIVE recall is 27.43% vs OFF 65.90% (38.48% absolute delta); F1 is 41.80% vs 79.27%; end-to-end correction recall is 25.33% vs 63.81% (38.48%). Precision changes from 99.43% to 87.80% (11.62%). Verdict: **REGRESSION**.

### 2. Which category drives recall change?

ACTIVE has no overall recall improvement. The largest positive category delta is **WRONG_DIACRITIC** at +12.00%; the largest drop is **ALL_UNACCENTED** at -48.00%. Full per-category numbers are in category-breakdown.csv.

### 3. Precision or clean-FP impact

ACTIVE has 20 total lexical FP vs OFF 2; precision is 87.80% vs 99.43%. CLEAN FP is 4/40 messages (10.00%) and 4 issues, versus OFF 1/40 and 1 issues.

### 4. Protected entities

ACTIVE protected-span violations: **0/29 (0.00%)**; OFF: 0/29. Target 0 is met.

### 5. Primary pipeline bottleneck

ACTIVE attribution counts:

| Stage | Errors | Share of GT errors |
| --- | ---: | ---: |
| PREFILTERED | 1 | 0.19% |
| CANDIDATE_MISS | 2 | 0.38% |
| SHORTLIST_MISS | 0 | 0.00% |
| MODEL_KEEP_ERROR | 13 | 2.48% |
| MODEL_WRONG_CANDIDATE | 160 | 30.48% |
| CORRECT_RANK_BUT_GATE_REJECT | 223 | 42.48% |
| CORRECT_EMIT | 126 | 24.00% |

The largest observed bucket is **threshold gate** (223 errors). This is the primary diagnosis from the trace, not an assumption about attention quality.

### 6. Why is attention latency high?

ACTIVE measures 9.645 attention calls/message, 15.875 ms attention time/message, and 1.646 ms/call. Engine p95 delta vs OFF is 58.972 ms. The dominant factor is **call count/message**; see latency-report.json for p50/p95/p99 and call distributions.

### 7. Highest-ROI next step

**calibrate the gate on a separate dev/calibration set.** Do not use SMS-REAL-200 to choose a new threshold; it remains the test set.

## SHADOW parity

shadow_output_mismatches = 0 across all final issue fields required by the task (ruleId, span, value, suggestions, severity). SHADOW preserves OFF output.

## Decision metrics

| Metric | SHADOW | ACTIVE |
| --- | ---: | ---: |
| Candidate pool oracle recall | 99.43% | 99.43% |
| Shortlist@8 oracle recall | 99.43% | 99.43% |
| Transformer top-1 given gold@8 | 66.86% | 66.86% |
| Transformer KEEP error rate | 2.49% | 2.49% |

## Runtime

| Arm | Engine p50 | Engine p95 | Engine p99 | Cold start | RSS after load | Calls/message | Attention ms/call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OFF | 8.665 ms | 19.936 ms | 24.569 ms | 7642.716 ms | 942 MiB | 0.000 | 0.000 ms |
| SHADOW | 59.548 ms | 103.638 ms | 120.947 ms | 7692.513 ms | 942 MiB | 9.620 | 2.112 ms |
| ACTIVE | 33.630 ms | 78.908 ms | 100.212 ms | 7178.463 ms | 942 MiB | 9.645 | 1.646 ms |

## Minimal-pair analysis

See adversarial-analysis.md for the required per-case INPUT, EXPECTED, generated pool, K=8 shortlist, KEEP/candidate logits/probabilities, selected word, final emission, stage, and verdict.

## FINAL VERDICT

Attention quality:
REGRESSION

Primary bottleneck:
threshold gate

Current model worth optimizing further:
NO

Reason:
The SMS-REAL-200 evidence does not justify treating the current attention model as a strong production improvement.

Highest-ROI next task:
calibrate the gate on a separate dev/calibration set

Do NOT do yet:
Train/fine-tune or tune thresholds on SMS-REAL-200; use a separate dev/calibration set for any future calibration work.
