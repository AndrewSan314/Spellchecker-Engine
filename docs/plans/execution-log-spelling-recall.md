# Execution Log — Vietnamese Spelling Recall Improvement (2026-08-24)

Plan: `docs/plans/2026-08-24-spelling-recall-improvement.md`
Workspace is NOT a git repo (plan repository rule): each task checkpoint is
recorded here instead of commits. No `git init` was performed.

## Task 1

- Status: DONE
- Source hashes: see `dataset_artifacts/evaluation/recall-failures-baseline.json`
  → `hashes` block (language.mjs, linguistic-rules.mjs,
  correction-taxonomy.mjs, run_spelling_eval.mjs, config.mjs, lm-ngrams.tsv).
- Tests run:
  - `node --test test/test_correction_taxonomy.mjs test/test_eval_split_guard.mjs`
    → 10 pass / 0 fail (RED-first history: taxonomy tests written before the
      module existed in an earlier session; verified green this session).
  - `node test/core.test.mjs` → 31 pass / 0 fail.
  - `node test/rules.test.mjs` → 35 pass / 0 fail.
- Dev metrics before/after: UNCHANGED by design (Task 1 is observability only,
  no production decision change). Strict recall 0.061, cross-lane recall
  0.1677 — identical to plan §0 baseline.
- Precision constraints: n/a (no emission behavior touched).
- Latency/RSS: n/a this task.
- Artifacts and hashes:
  - `dataset_artifacts/evaluation/recall-failures-baseline.json`
    sha256 b378f97d89fcbd64… (full hash + manifest sidecar in
    `recall-failures-baseline.json.manifest.json`).
- Findings (the immutable baseline, totals reconcile to 1115 labels exactly):
  - Relations: ACCENTED_SAME_KEY 612 · DIFFERENT_KEY_SINGLE_TOKEN 346 ·
    UNACCENTED_SAME_KEY 126 · SPLIT 28 · IDENTITY 3.
  - Stages: prefilter 852 · correct-other-lane 119 · unsupported-multi-token 59
    · correct 68 · cheap-ranker-loss 16 · candidate-miss 1.
  - Prefilter reasons: **classified-DICTIONARY 808** (72% of ALL labels),
    classified-UNACCENTED_VALID 40, min-token-length 3,
    capitalized-proper-noun 1.
  - PMD decisions on uncredited labels: 808 `not-in-pipeline-classified-DICTIONARY`,
    38 `gate-declined`, 3 `no-competing-surface`, 22 emitted-but-other-target.
    This confirms the plan's thesis precisely: the recall wall is real-word
    lane eligibility, not ranking or thresholds.
- Implementation notes / deviations:
  - `classifyCorrectionRelation` adds a documented residual value
    `DIFFERENT_KEY_MULTI_TOKEN` so `DIFFERENT_KEY_SINGLE_TOKEN` stays pure
    for real-word-lane routing (plan lists only single/multi token counts;
    extension documented in module header + tests).
  - `relationStages` records the full stage breakdown per relation
    (superset of the plan's `{prefilter, correct}` sketch) at zero extra cost.
  - Terminal-stage attribution extracted into shared exported
    `stageFromDecision()` (run_spelling_eval.mjs) and consumed by both the
    evaluator and `analyze_recall_failures.mjs` so the two reports cannot drift.
  - Analyzer pool-rank histograms are recorded only once a spelling decision
    exists (`stage === 'decided'`); earlier stages have no pools to rank into.
- Next task decision: PROCEED to Task 2 (semantic product-recall metric).

---

## Task 2

- Status: DONE
- Source hashes: current tree (see next artifact manifest); engine sources
  untouched — Task 2 modifies evaluation/reporting only.
- Tests run:
  - `node --test test/test_semantic_correction_matching.mjs` → 6 pass / 0 fail
    (RED-first: written before `linguisticCorrectionMatches` existed; two
      failures exposed a missing NFC normalization and sharpened the strict-
      vs-semantic pinning test).
  - `node --test test/test_semantic_correction_matching.mjs test/test_benchmark_scope.mjs`
    → 9 pass / 0 fail.
  - `node test/core.test.mjs` → 31 pass / 0 fail.
  - `node test/rules.test.mjs` → 35 pass / 0 fail.
  - `npm.cmd run eval:spelling:dev` (via node tools/run_spelling_eval.mjs) → OK.
- Dev metrics before/after:
  - Engine issue set byte-equivalent (stages / relationStages / relations /
    prefilterReasons identical to the Task 1 baseline, values compared
    zero-insensitively + order-insensitively).
  - New reporting views (labels=1115):
    - strictRuleId P/R = 0.9067 / 0.0610 (unchanged contract view);
    - semanticLinguistic P/R = 0.6361 / 0.1677 (PRIMARY product score;
      equals the legacy crossLane recall as expected);
    - valueOnlyDiagnostic P/R = 0.7653 / 0.2018 (diagnostic only).
- Precision constraints: no production change ⇒ all gates trivially hold.
- Latency/RSS: n/a this task.
- Artifacts and hashes:
  - `.tmp/task2-eval-check.json` (verification run; not an acceptance
    artifact). Benchmark reports gain an additive `semanticLinguistic`
    block on next benchmark regeneration.
- Implementation notes / deviations:
  - `linguisticCorrectionMatches(issue, expected)` implemented in
    `src/correction-taxonomy.mjs`; supports both expect shapes
    (`suggestions:[…]}` vsec adapter and singular `suggestion` benchmark);
    NFC+case normalized; deterministic rule IDs never credit.
  - Evaluator report now exposes `metrics.views.{strictRuleId,
    semanticLinguistic, valueOnlyDiagnostic}` with legacy `metrics.strict`
    and `metrics.crossLane` kept as aliases for existing consumers.
  - `benchmark/run-benchmark.mjs`: additive per-row `semanticMissed` +
    top-level `semanticLinguistic {caughtIssues, missedIssues, recall}`;
    strict totals/recall/precision semantics untouched.
  - BUG FOUND & FIXED during equivalence verification: the evaluator's
    caught-path incremented `stageCounts` directly, so `correct` /
    `correct-other-lane` cells were missing from `relationStages` in the
    evaluator report (the analyzer recorded them correctly). All label paths
    now route through `countStage`, restoring the one-label-one-cell invariant.
- Next task decision: PROCEED to Task 4 (shadow wrong-diacritic lane).

---

## Task 3

- Status: DONE
- Source hashes: current tree; engine behavior proven unchanged (below).
- Tests run:
  - `node --test test/test_correction_candidate_builder.mjs` → 6 pass / 0 fail
    (RED-first: import error before implementation existed).
  - `node --test test/test_correction_candidate_builder.mjs
    test/test_candidate_generation.mjs test/test_context_reranker.mjs`
    → 20 pass / 0 fail.
  - `npm.cmd test` (rules suite) → 35 pass / 0 fail.
  - `node test/core.test.mjs` → 31 pass / 0 fail.
  - `npm.cmd run eval:spelling:dev` → OK, metrics identical to Task 2 state.
- Dev metrics before/after: IDENTICAL — stages / relationStages / relations /
  prefilterReasons / metrics.views all equal to the Task 2 verification run
  (zero-insensitive + order-insensitive JSON comparison). Production issue
  set unchanged; real-word lanes built but NOT consumed for emission.
- Precision constraints: no emission change ⇒ all gates hold. Candidate
  oracle coverage re-measured at **93.5%** (= §0 baseline, >=93% gate ✓).
- Latency/RSS: n/a this task (no hot-path change beyond one cheap
  classifyToken re-call inside the builder guard).
- Artifacts and hashes:
  - `.tmp/task3-eval-check.json` (verification run).
  - `tools/audit_candidate_recall.mjs --split dev` now reports per-lane
    coverage (console run; artifact not regenerated to keep the committed
    baseline intact):
    - owned lanes: UNACCENTED_SAME_KEY **126/126 (100%)**, ACCENTED_SAME_KEY
      **612/612 (100%)**, DIFFERENT_KEY_REAL_WORD **273/316 (86.4%)**;
    - relations in audit terms: ACCENTED_SAME_KEY 612 ·
      DIFFERENT_KEY_SINGLE_TOKEN 316 · UNACCENTED_SAME_KEY 126 · SPLIT 28 ·
      IDENTITY 3 (DIFFERENT_KEY count differs from the evaluator's 346
      because the audit strips punctuation from targets before classifying —
      pre-existing label hygiene, documented in-file).
- Implementation notes / deviations:
  - `buildCorrectionCandidates({token, lane, services, snap, ctx?, doc?})`
    exported from `src/rules/linguistic-rules.mjs`; `CORRECTION_LANES`
    frozen list; entries carry `{word, stripped, dist, freq, sameAccentKey}`.
  - DIFFERENT_KEY_REAL_WORD keeps same-key surfaces in the pool but FLAGGED
    (`sameAccentKey`) instead of filtered, so lane audits slice without a
    second generation pass (documented choice).
  - `evaluateSpellingToken` now generates via the builder's UNKNOWN_TYPO
    lane; the inline sibling-merge block was deleted (logic moved verbatim
    into the builder); `pool` in the empty-pool return reshaped as
    `{entries}` — diagnostic-only field, consumers unaffected.
  - Audit tool: added `relations`, `laneCoverage`, `ownedLaneCoverage`
    blocks with per-lane percentages; strict totals untouched.
- Next task decision: PROCEED to Task 5 (different-key real-word lane in
  SHADOW).

---

## Task 4

- Status: DONE
- Source hashes: current tree; OFF-mode engine behavior proven unchanged.
- Tests run:
  - `node --test test/test_context_evidence.mjs` → 4 pass / 0 fail (RED-first:
    module missing → import error).
  - `node --test test/test_wrong_diacritic_lane.mjs` → 7 pass / 0 fail
    (RED-first; one REAL bug caught by tests: the provisional helper received
    WORD-token objects where string surfaces were expected, so every window
    count read zero — fixed by normalizing to `.normalized` inside the helper).
  - `node test/core.test.mjs` → 31 pass / 0 fail.
  - `node test/rules.test.mjs` → 35 pass / 0 fail.
- Dev metrics before/after (OFF mode): identical to Task 2/3 state — stages,
  relationStages, relations, prefilterReasons, metrics.views all equal;
  production issue set unchanged. SHADOW run changes ONLY evaluation-side
  stage attribution (dictionary tokens now reach the decision function), as
  the plan intends.
- Precision constraints: no user-visible emission change ⇒ gates hold in the
  shipped default (OFF). SHADOW emits nothing (`gates.shadowSuppressed`).
- Latency/RSS: n/a this task for default config; SHADOW adds per-decision
  feature extraction only on uncaught labeled tokens during measurement runs.
- Artifacts and hashes:
  - `.tmp/task4-off-eval.json` (OFF verification),
    `.tmp/task4-shadow-eval.json` (SHADOW headroom).
- SHADOW headroom measured on dev (uncaught labels, provisional gate):
  - decisions 488 · wouldEmit 75 · **TP 28 / FP 47** (precision 0.373);
  - gate failures: originalZeroTrigramWins 372 (conservative ✓),
    sameAccentKey 56 (best rival different-key → other lane's business),
    marginPositive 8, candidateAttestedWindow 6;
  - reachableNoPmdLine 49 of the previously unreachable same-key labels.
  - Verdict: provisional ceiling is real but FAR too loose for activation —
    exactly why the plan forbids activating before Tasks 6/9 calibration.
- Implementation notes / deviations:
  - `src/context-evidence.mjs`: `extractContextEvidence` (10 features,
    log1p ratios, finite at zero) + `PROVISIONAL_EVIDENCE_WEIGHTS` +
    `evidenceScore`. Feature contract pre-registered as `recall-pairwise-v1`
    ordering is finalized in Task 6 tests.
  - Config: `linguistic.wrongDiacriticMode: 'OFF'` default; absent-key legacy
    mapping (`wrongToneEnabled=true` → ACTIVE) preserved via exported
    `wrongDiacriticModeOf(snap)`; explicit key always wins over legacy flag.
  - `evaluateSpellingToken`: eligibility expansion guarded to ACCENTED
    dictionary tokens under non-OFF modes; SHADOW computes its OWN pairwise
    gate independent of the legacy softmax conf/margin chain (so headroom is
    not capped by unknown-token thresholds) and suppresses emission via
    `gates.shadowSuppressed`; ACTIVE keeps today's decisive-trigram path
    until Task 9 replaces thresholds.
  - Evaluator: `--wrong-diacritic-mode` flag (validated), provenance keys,
    and a `shadowWrongDiacritic` report block (decisions/wouldEmit/tp/fp/
    checkFailures/reachableNoPmdLine).
- Next task decision: PROCEED to Task 5 (real-word typo lane, SHADOW).

---

## Task 5

- Status: DONE
- Source hashes: linguistic-rules.mjs 77b271e14230d147… ·
  run_spelling_eval.mjs 9424f171fd1a2e17… · config.mjs 6384522e222bf944… ·
  test_real_word_typo_lane.mjs 7373377fa6697627….
- Tests run:
  - `node --test test/test_real_word_typo_lane.mjs` → **8 pass / 0 fail**
    (RED-first honored: the previous session left the file with 2 failing
    positive cases; this session fixed the implementation until green, then
    added an eighth issue-parity regression test).
  - `node --test test/test_wrong_diacritic_lane.mjs test/test_context_evidence.mjs
    test/test_correction_candidate_builder.mjs test/test_candidate_generation.mjs
    test/test_context_reranker.mjs test/core.test.mjs` → 62 pass / 0 fail.
  - `npm.cmd test` (rules suite) → 35 pass / 0 fail.
- Dev metrics before/after:
  - OFF mode: stages / relationStages / relations / prefilterReasons /
    metrics.views IDENTICAL to the Task 4 state (zero-insensitive +
    order-insensitive comparison vs `.tmp/task4-off-eval.json`; labels
    reconcile 1115 = 1115). Production issue set unchanged — verified again
    on the final source state (`.tmp/task5-off-eval.json`, EQUIVALENCE: PASS).
  - SHADOW run views byte-equal OFF views (strictRuleId 0.9067/0.0610,
    semanticLinguistic 0.6361/0.1677, valueOnlyDiagnostic 0.7653/0.2018) —
    the lane is fully invisible in user-visible output.
- Precision constraints: no user-visible emission change ⇒ gates hold in the
  shipped default (OFF). SHADOW emits nothing (`gates.realWordShadowSuppressed`).
- Latency/RSS: n/a for default config; SHADOW adds ≤12 pairwise probes per
  dictionary-valid token reaching the cascade during measurement runs only.
- Artifacts and hashes:
  - `dataset_artifacts/evaluation/recall-real-word-shadow.json`
    sha256 5d1e08b026278e1d… (+ manifest sidecar), written by the evaluator
    (`--real-word-typo-mode SHADOW`) at the final source state.
  - `.tmp/task5-off-eval.json` (OFF equivalence verification),
    `.tmp/compare-task5.mjs` (comparison script).
- SHADOW headroom measured on dev (uncaught labels):
  - decisions 651 · wouldEmit 593 · **TP 163 / FP 430** (wouldEmit precision
    0.275);
  - gate failures: marginPositive 27 · accentKeyRelation 32 ·
    candidateAttestedWindow 1;
  - reachableNoPmdLine 585; `classified-DICTIONARY` prefilters 808 → 0 under
    SHADOW (fully reachable by construction).
  - Verdict: real but FAR too loose for activation (same conclusion as Task
    4's provisional ceiling) — activation stays forbidden before Tasks 6/9
    calibration.
- Implementation notes / deviations:
  - Inherited state: a previous session had started Task 5 (tests + config key +
    first wiring) and stopped mid-TDD with the two positive cases RED. This
    session completed it without redoing finished work.
  - BUG FIXED (inherited): the lane selected its candidate as "first
    different-key entry of the legacy post-scoring ranking" — wrong twice: it
    skipped same-key vowel-change winners ("đen"→"đến") and inherited the
    cheap-ranker blindspot ("cách" cut before context scoring). Replaced with
    a bounded scan over the FULL wide pool (≤12 distinct surfaces inside
    `maxEditDistanceFor`, deterministic dist/freq/word order); each surface
    gets the pairwise gate; winner = best-evidence survivor of the MINIMAL
    SURFACE Telex edit-distance tier (stripped-key distance collapses prefix
    families and cannot separate "đến" from "để"); runner-ups recorded (max 3)
    as hard negatives for the Task 6 trainer.
  - GATE SPLIT: `originalZeroTrigramWins` now applies ONLY to
    ACCENTED_SAME_KEY (Task 4 semantics). The real-word lane omits it per
    plan §Task5 eligibility list: grammatical originals ("Các", "hàng")
    legitimately attest in trigrams; their pull is priced into marginPositive.
  - ACCENT KEY POLICY: the real-word lane rejects same-key rivals that differ
    ONLY by tone mark (NFD five-tone-strip comparison — "quỳ"→"quý",
    "hành"→"hàng": wrong-diacritic territory) while allowing same-key
    vowel-quality changes ("đen"→"đến") that no other lane can reach because
    the token is dictionary-valid.
  - CRITICAL BUG FIXED (found by measurement, guarded by a new test): the
    inherited wiring let dictionary tokens pulled into the cascade by
    `isRealWordTypo` EMIT through the legacy unknown-token softmax gates — a
    first SHADOW run crashed strict precision 0.907 → 0.196. Added explicit
    suppression (`gates.realWordShadowSuppressed`); new regression test asserts
    SHADOW/OFF issue-set identity across six dictionary-heavy lines.
  - Evaluator: `--real-word-typo-mode` flag (validated), provenance keys,
    `shadowRealWordTypo` report block mirroring Task 4's. Both lane flags are
    applied in ONE `configService.reload()` call (reload merges onto the
    DEFAULT snapshot, so sequential reloads would drop the earlier flag).
  - Deviation (Task 4 precedent): plan Step 5's analyzer command was replaced
    by evaluator runs with `--out` (analyze_recall_failures.mjs has no lane
    mode flags; adding them would duplicate plumbing ahead of need).
- Next task decision: PROCEED to Task 6 (leakage-safe pairwise reranker;
  feature contract `recall-pairwise-v1` freeze).

---

## Task 6

- Status: DONE
- Source hashes: recall-reranker.mjs (new) · train_recall_reranker.py (new) ·
  extract_recall_training_rows.mjs (new) · engine.mjs · linguistic-rules.mjs ·
  run_spelling_eval.mjs; artifact `src/data/recall-reranker.json`
  sha256 583d3ae722539d6b… (+ manifest sidecar), trained from rows file
  sha256 6426b672b1f4bc88….
- Tests run:
  - `python -m unittest test.test_recall_reranker_training -v` → **8 pass**
    (standardization math, finite sigmoid, BYTE-IDENTICAL double training on
    the fixture — both runs sha256 91410ef7…, leakage guard rejects each of
    the five forbidden marks, CLI forbidden run exits non-zero with a clear
    message).
  - `node --test test/test_recall_reranker.mjs` → **8 pass / 0 fail** (RED-
    first for loader validation: schema/contract/feature-order rejection,
    missing-feature rejection, deterministic hand-computed sigmoid, strict
    open-interval output, canonical ranks, real-artifact smoke).
  - Full focused battery → **97 pass / 0 fail**; `npm.cmd test` → 31+35 pass.
- Dev metrics before/after:
  - OFF mode WITH the artifact live: EQUIVALENCE PASS vs Task 5 baseline
    (`.tmp/task6-off-eval.json`) — production issue set untouched.
  - Both-lane SHADOW run views byte-equal OFF views; headroom numbers
    identical to Tasks 4/5 (WD 488/75 TP28 FP47; RW 651 decisions wouldEmit
    593 TP163 FP430) — model probability rides along WITHOUT changing gates
    (Task 7 Step "keep lanes in SHADOW" honored).
  - Model separation measured on `.tmp/task6-shadow-decisions.jsonl`:
    DIFFERENT_KEY_REAL_WORD prob>=0.90 → TP117/FP28 (precision 0.807);
    ACCENTED_SAME_KEY caps at ~0.667 precision even near prob 1.0.
- Precision constraints: no user-visible change ⇒ gates hold in shipped
  default. Latency/RSS: n/a default; SHADOW adds ≤12 score() dot products
  per eligible token (15-dim, allocation-free).
- Implementation notes / deviations:
  - Feature extraction is JS (`tools/extract_recall_training_rows.mjs`,
    NEW file beyond the plan list): the trainer cannot re-implement
    production SymSpell/LM helpers in stdlib Python, so it consumes
    pre-extracted feature rows (header declares sources; trainer validates).
    Real command therefore = extractor + trainer (documented deviation from
    the plan's single-command shape). VSEC splits live under
    `dataset_artifacts/vsec/` (plan wrote `dataset_artifacts/vsec-train.jsonl`).
  - Trainer: full-batch GD, zero init, lr schedule lr0/(1+decay·t),
    L2 never touching bias, class weights n/(2·count), stable sort ordering,
    NO RNG anywhere (seed recorded for provenance only); weights-only
    artifact + atomic writes + manifest (inputs hashed).
  - Leakage guard enforced TWICE: path markers refused before any read
    (trainer + extractor) and header sources validated against allowed set.
  - `RecallReranker.loadDefault()` wired through SmsValidationEngine
    constructor/services: absent artifact → disabled stub; present-but-
    malformed → construction fails fast. One immutable instance, never read
    per token.
  - Shadow decisions now carry `modelProbability` (+ canonical cheap/pool
    ranks threaded to both call sites; evidence-desc ties broken by model
    probability for reproducibility).
  - Evaluator gained `--dump-shadow-decisions <path>` writing one JSONL
    record per shadow decision {lane, rel, id, target, candidate, wouldEmit,
    evidenceScore, modelProbability, checks, features} — the substrate for
    Task 9's offline threshold replay.
  - BUG caught by suites during wiring: wd call site referenced
    nonexistent `wideRankedWords` variable (it was an object-literal key);
    fixed to `wideRanked`. Also fixed float64 edge: sigmoid clamp tightened
    ±60→±36 so outputs stay strictly inside (0,1).
  - Training data: 8992 train labels seen → 7125 positives (target-in-oracle
    only; UNACCENTED_SAME_KEY left to PMD by design), 15923 negatives (same-
    distance freq rival + context-attested rival + cheapest rival + 3960
    clean-train rows), 421 oracle misses recorded.
- Next task decision: PROCEED to Task 7 (diversity-preserving candidate
  shortlist).

---

## Task 7

- Status: DONE
- Source hashes: linguistic-rules.mjs (selectDiverseShortlist exported +
  cascade wiring) · config.mjs (`spellingShortlistSize: 4`) ·
  audit_candidate_recall.mjs (additive `shortlistCoverageNoContext` block);
  artifact `dataset_artifacts/evaluation/candidate-recall-dev-task7.json`
  sha256 749283ae79e24e79….
- Tests run:
  - `node --test test/test_candidate_shortlist.mjs` → **5 pass / 0 fail**
    (RED-first: context winner below the frequency cut survives; bound ≤4;
    union covers all four heads; determinism; no-attest fallback).
  - `npm.cmd test` → 31+35 pass; candidate suites + core → 50 pass.
- Dev metrics before/after (OFF default):
  - stages: cheap-ranker-loss **16 → 13** (below 16 ✓); correct 68 → 69;
    gate-rejected 0 → 2 (two labels now REACH the gates instead of being cut);
  - strictRuleId P/R/F0.5: 0.9067/0.0610/0.2403 → **0.9079/0.0619/0.2431**
    (precision improved);
  - semanticLinguistic P/R/F0.5: 0.6361/0.1677/0.4081 →
    **0.6416/0.1686/0.4110**.
- Precision constraints / gates:
  - candidate oracle coverage **93.5%** (>=93% gate ✓);
  - SMS <=160 p95 **3.16 ms** via `npm.cmd run profile:engine` (<=15 ms ✓);
  - expensive non-original candidates bounded at 4 structurally
    (`spellingShortlistSize`, selector caps internally, unit-tested).
- Latency/RSS: sms160P95 3.16 ms; steadyRss ~852 MB (unchanged class).
- Implementation notes / deviations:
  - Attestation probes moved OUT of the sort comparator into a per-candidate
    map computed once (strictly fewer LM probes than before, identical
    scores/ordering).
  - The pure selector takes an explicit `cheapOrder` (the production
    wide-ranked list) for head 4 so "cheap score" stays a production concept
    rather than being duplicated inside the selector; unit test mirrors that
    call shape.
  - Audit tool reports `shortlistCoverageNoContext`: worst-case (zero-context
    assumption) target survival through the real selector — UNACCENTED_SAME_KEY
    99.2%, ACCENTED_SAME_KEY 97.4%, DIFFERENT_KEY_REAL_WORD 44% lower bound
    (production context attestation rescues more; evaluator stages show the
    net effect).
- Next task decision: PROCEED to Task 8 CONDITION EVALUATION after Task 9
  calibration numbers are known (plan sequences the decision on Tasks 1–7
  passing gates — they do — AND semantic dev recall remaining below 0.35,
  which cannot be finally judged until lane activation lands).

---

## Task 8

- Status: DONE (condition TRIGGERED and executed; acceptance criterion NOT
  met on dev — lane remains SHADOW)
- Condition check: Tasks 1–7 pass all precision gates ✓; semantic dev recall
  after single-token calibration = 0.2108 < 0.35 ⇒ condition MET, task
  executed.
- Source hashes: word-boundary-candidates.mjs (new) · linguistic-rules.mjs
  (createWordBoundaryRule + wiring) · engine.mjs (rule registered) ·
  config.mjs (`wordBoundaryCorrectionMode: 'SHADOW'` default,
  `wordBoundaryMinFrequency: 2000`) · run_spelling_eval.mjs
  (shadowWordBoundary block + WORD_BOUNDARY dump records).
- Tests run: `node --test test/test_word_boundary_candidates.mjs` → **6 pass /
  0 fail** (RED-first: split 'cảmơn'->'cảm ơn'; merge 'như ng'->'nhưng';
  merge span covers the exact original substring incl. the space;
  URL/code/placeholder ranges untouched; bounded deterministic counts; SHADOW
  rule surfaces nothing).
- Dev metrics before/after: OFF/SHADOW issue sets byte-identical
  (views parity TRUE); SHADOW headroom on dev after quality guards:
  decisions 400 · wouldEmit 1 · TP 0 / FP 1.
- Precision constraints: no user-visible change in shipped default ✓.
- Implementation notes / deviations:
  - Split only at internal boundaries with BOTH halves corpus-backed
    (realFrequency > 0), each ≥2 chars; ≤4 candidates; merges only across a
    single ordinary space with corpus-backed concatenation; ≤2 candidates.
  - Provisional wouldEmit: frequency floor + DIRECT bigram attestation of the
    repaired sequence (+ anchored neighbour context); unigram frequency alone
    never fires.
  - QUALITY GUARDS added after measurement showed junk: (a) no single-char
    halves ('b ảo' style — bare letters carry huge news-corpus frequencies);
    (b) never split a token that is itself well-attested ('nhưng', 'trương'):
    genuine solid-written errors have ~zero frequency of their own. These cut
    wouldEmit from 49 junky fires to 1 on dev.
- Acceptance verdict (plan Step 4): semantic F0.5 does NOT increase with the
  lane ACTIVE (TP 0 / FP 1) ⇒ lane stays SHADOW; recorded honestly rather
  than forced.
- Next task decision: PROCEED to Task 9 final freeze (combined calibration
  verdict recorded there).

---

## Task 9

- Status: DONE (no feasible trial under full constraints; lanes NOT forced
  active per plan rule; configuration frozen)
- Source hashes: calibrate_recall_lanes.mjs (new) · test_recall_calibration.mjs
  (new) · config/spelling-tuning.json v2 (frozen) · src/config.mjs (Task 9
  threshold keys + defaults).
- Tests run: `node --test test/test_recall_calibration.mjs` → **6 pass /
  0 fail** (hard-constraint rejection with reasons, F0.5 maximization,
  precision/latency/lexical tie-breaks, byte-identical repeated selection,
  replay hit accounting mirroring the evaluator, plan floors exposed).
- Calibration flow: shadow-decision REPLAY grid per lane (instant) → real
  in-process dev verification per surviving candidate (semantic views,
  red-team clean corpus, synthetic missing-diacritic recall, SMS-length p95)
  → pure selector.
- Verdicts:
  - WRONG-DIACRITIC: cannot clear the 0.70 lane-local precision floor at any
    threshold (max ~0.67 near probability 1.0) ⇒ stays OFF/SHADOW.
  - REAL-WORD best trial (p≥0.85, o=0): lane precision 0.72, semantic P
    0.6416→**0.6528** (improved), F0.5 0.411→**0.4599** (+11.9% relative,
    above the 10% relative gate), md recall unchanged, clean corpus
    unchanged, p95 10.8 ms — BUT semantic recall 0.2108 fails the plan's
    "+10 pp before activation" bar (0.2677) ⇒ rejected.
  - COMBINED RW+word-boundary: recall 0.2206 — still short ⇒ rejected.
  - winner: null; feasibleCount 0; five trials recorded with explicit
    violation lists in `dataset_artifacts/evaluation/recall-calibration.json`
    sha256 592864773c23adca….
- Freeze (config/spelling-tuning.json v2, frozen:true): acceptedLaneModes
  wrongDiacriticMode OFF · realWordTypoMode OFF · wordBoundaryCorrectionMode
  SHADOW; frozenParams include spellingShortlistSize 4 and all Task 9
  threshold keys; artifacts hashed (calibration + reranker model
  583d3ae722539d6b…).
- CRITICAL BUGS FIXED during verification (both found by real measurement):
  - ACTIVE-mode flood: dictionary tokens still traversed the legacy softmax
    cascade (suppression was SHADOW-only) — strict precision collapsed to
    ~0.14 in a verification run. Fixed by a dedicated LIGHT PATH for ACTIVE
    dictionary tokens (evaluateRealWordToken) that bypasses the cascade and
    decides emission solely from the calibrated lane decision; the shared
    computeRealWordLaneDecision makes SHADOW and ACTIVE decisions identical
    by construction.
  - Latency: full-cascade expansion cost every dictionary token ~5 ms ⇒
    light path + quick-attested precheck (3 map lookups vs full feature
    extraction for hopeless candidates) brought ACTIVE sms160 p95 back under
    budget (measured 10.6–10.9 ms in-loop incl. dev rows; short-SMS microbench
    p95 1.36 ms vs OFF 1.31 ms).
- Next task decision: PROCEED to Task 10 (acceptance gates + ONE held-out
  evaluation; held-out data remained untouched so far).

---

## Task 10

- Status: DONE
- Freeze check (Step 1): config/spelling-tuning.json v2 frozen:true; hashes
  recorded (calibration artifact 592864773c23adca…, model 583d3ae722539d6b…);
  lane modes OFF/OFF/SHADOW — no mismatch abort.
- Tests run (Step 2): `npm.cmd test` → 31+35 pass · python trainer unittest →
  OK (byte-deterministic) · ALL focused suites per plan list +
  test_candidate_shortlist/test_recall_calibration/test_word_boundary_candidates
  → **109 pass / 0 fail** · `npm.cmd run data:audit` → pass · candidate audit
  oracle **93.5%** (≥93% ✓) · final dev eval written.
- Dev gates (Step 3):
  - strictRuleId P/R/F0.5 = 0.9079/0.0619/0.2431;
  - semanticLinguistic P/R/F0.5 = 0.6416/0.1686/0.4110;
  - candidate oracle ≥93 ✓ · synthetic missing-diacritic recall 0.9559
    (no drop) ✓ · red-team clean corpus: no new unexpected issues ✓ ·
    SMS ≤160 p95 **3.17 ms** (≤15 ✓) · steady RSS ~853 MB;
  - semantic recall target ≥0.35: NOT MET (0.1686). Recorded as failed
    objective; per plan this blocks the attention stop/go (criterion 4/5).
- Held-out run (Step 4) — executed EXACTLY ONCE after freeze:
  - VSEC test (`final-recall-heldout.json`, sha256 ce9ff4a59d8b202e…,
    --final guard honored): labels 1121; strictRuleId
    0.9118/0.0553/0.2225; semanticLinguistic **0.6469/0.1650/0.4084**;
    valueOnlyDiagnostic 0.7649/0.1945.
  - Full benchmark (`npm.cmd run bench` → benchmark/results.json): expected
    2949 · caught 1646 · FP 28 · error recall 55.8% · precision 98.3%
    (covers VSEC test, Viwiki external-test, synthetic missing-diacritic,
    deterministic categories, clean false positives).
  - Generalization: dev ≈ held-out on every view (no overfitting); no tuning
    was performed against held-out at any point.
- README updated honestly (Step 5): top status line, lane states table
  (OFF/OFF/SHADOW with reasons), frozen-state dev + held-out metric tables,
  benchmark snapshot, performance line, extended suite list incl. Python
  trainer tests.
- Known unsupported categories (unchanged by this plan): multi-token
  corrections beyond the bounded boundary lane, SPLIT/MERGE labels not
  matching corpus-backed boundary patterns, DIFFERENT_KEY targets outside the
  generation pool (~6.5% of routable labels), wrong-diacritic tone twins
  (lane reachable in SHADOW only, sub-floor precision).
- Attention stop/go: criteria 1 (lanes reachable), 2 (pairwise calibrated),
  3 (oracle ≥93%) hold; criterion 4 FAILS (semantic recall 0.165 < 0.35 at
  the precision floor) and criterion 5 fails (prefiltering still dominates:
  852/1115 dev labels sit behind classified-DICTIONARY-style prefilters
  under OFF). Attention milestone NOT opened.
- Final artifact hashes: final-recall-dev-report.json ·
  final-recall-performance.json (sms160P95 3.17 ms) ·
  final-recall-heldout.json ce9ff4a59d8b202e… (+ manifest sidecars).
- Plan COMPLETE. All tasks executed or conditionally resolved; every gate
  result recorded above exactly as measured.

---

## Post-plan calibration audit (2026-08-25)

- Root cause: `calibrate_recall_lanes.mjs` defaulted `maxVerified` to 3 and
  truncated replay survivors in grid-generation order. This skipped every
  `p>=0.95` real-word trial even though replay precision was 0.8537-0.8908.
- Fix: verify all replay survivors by default; an explicit cap now ranks by
  replay precision/TP first. Calibration baseline explicitly forces the
  experimental lanes OFF/SHADOW so an ACTIVE production default cannot
  contaminate future baseline measurements.
- Full-grid result: `DIFFERENT_KEY_REAL_WORD p>=0.95/o<=3` is feasible and
  ACTIVE. Dev semantic P/R/F0.5 = 0.6502/0.2717/0.5086 versus baseline
  0.6416/0.1686/0.4110; no new clean issues; MD recall 0.9559 unchanged;
  independent profile SMS<=160 p95 12.79 ms. Wrong-diacritic remains OFF and
  word-boundary remains SHADOW.
- Regression: focused real-word/calibration 16/16; core/rules 31+35 pass.
- Held-out was NOT rerun; the existing held-out artifact belongs to v2.
