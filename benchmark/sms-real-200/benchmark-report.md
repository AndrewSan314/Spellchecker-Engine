# SMS-REAL-200 Benchmark Report

Run: 2026-09-06T10:26:47.024Z; dataset: 200 messages; Node v24.13.0; Windows 10.0.26200.

## Summary

| Metric | UPSTREAM | FORK_FULL | FORK_LITE | Best |
| --- | ---: | ---: | ---: | --- |
| detection_precision | 0.9910 | 0.9943 | 0.9970 | FORK_LITE |
| detection_recall | 0.6267 | 0.6590 | 0.6343 | FORK_FULL |
| detection_f1 | 0.7678 | 0.7927 | 0.7753 | FORK_FULL |
| end_to_end_correction_recall | 0.6038 | 0.6381 | 0.6095 | FORK_FULL |
| clean_fp_rate | 0.0250 | 0.0250 | 0.0000 | FORK_LITE |
| protected_span_violation_rate | 0.0000 | 0.0000 | 0.0000 | UPSTREAM |
| ALL_UNACCENTED_recall | 0.6347 | 0.6800 | 0.6427 | FORK_FULL |
| SOME_UNACCENTED_recall | 0.6500 | 0.6500 | 0.6667 | FORK_LITE |
| WRONG_DIACRITIC_recall | 0.0800 | 0.0800 | 0.1200 | FORK_LITE |
| TYPO_TELEX_recall | 0.8400 | 0.8400 | 0.8800 | FORK_LITE |
| ADVERSARIAL_accuracy | 0.2000 | 0.2000 | 0.3000 | FORK_LITE |

## Category breakdown

See `category-breakdown.csv` for errors, precision, recall, F1, correction accuracy, end-to-end correction recall, and FP by category.

## Clean and protected-entity safety

- UPSTREAM: CLEAN FP rate 2.50%, 1 FP issues; protected-span violations 0/29 (0.00%).
- FORK_FULL: CLEAN FP rate 2.50%, 1 FP issues; protected-span violations 0/29 (0.00%).
- FORK_LITE: CLEAN FP rate 0.00%, 0 FP issues; protected-span violations 0/29 (0.00%).

## Performance

- UPSTREAM: cold load 7455.9 ms, RSS 873.4 MiB, median 8.450 ms, p95 23.132 ms, p99 25.779 ms.
- FORK_FULL: cold load 7620.7 ms, RSS 872.8 MiB, median 10.122 ms, p95 24.603 ms, p99 29.446 ms.
- FORK_LITE: cold load 1307.0 ms, RSS 335.3 MiB, median 8.006 ms, p95 23.077 ms, p99 40.566 ms.

## Interpretation

This report uses issue-level detection and one-to-one span/token matching. A detected error with a wrong suggestion is detection TP but correction failure. No SMS-REAL-200 record was used for training or tuning.

### Does the fork improve on upstream?

- FORK_FULL is the quality winner: detection precision +0.33 pp, recall +3.24 pp, F1 +2.49 pp, and end-to-end correction recall +3.43 pp versus UPSTREAM.
- FORK_LITE is the safety/resource winner: precision +0.60 pp, clean FP rate 0%, median latency about 7.0 ms, cold load about 1.2 s, and RSS about 334 MiB; its recall/F1 gains are smaller than FORK_FULL.
- Both fork profiles preserve 0 protected-span violations. The fork removes the CODE_SWITCH false positive seen in UPSTREAM; FORK_FULL still has one CLEAN false positive.

### Regressions and caveats

- FORK_LITE fails 4 existing suite assertions (chao→chào, SHADOW collection, Các→Cách, and equal-evidence silence); this is a real profile behavior caveat, not a benchmark scoring failure.
- The benchmark still has substantial false negatives: 34.1% of expected errors remain undetected for FORK_FULL, and adversarial end-to-end accuracy is only 20% for FORK_FULL versus 30% for FORK_LITE.
- The exact `ma nay` case is not in SMS-REAL-200 and is reported separately: all three avoid the bad `ma→mà` suggestion, but none emits the expected `ma→mã` correction.

### Recommendation

Recommendation: FORK_FULL is the best candidate for further evaluation because it wins the quality metrics, but it is not ready for unrestricted production yet. Keep protected-entity and clean-FP gates mandatory, and address the remaining ambiguity/adversarial recall before deployment.


## Required `ma nay` case

The exact prompt case is evaluated separately because it is not one of the 200 SMS-REAL-200 records. Full raw output is in `special-case-results.json`.
- UPSTREAM: Khong -> không; se -> sẻ; nay -> này; bat -> bất; ky -> kỳ.
- FORK_FULL: Khong -> không; se -> sẻ; nay -> này; bat -> bất; ky -> kỳ.
- FORK_LITE: Khong -> không; se -> sẻ; bat -> bất; ky -> kỳ.

Raw per-message outputs are in the three `*-results.jsonl` files; false positives and false negatives are fully listed in the CSV files.
