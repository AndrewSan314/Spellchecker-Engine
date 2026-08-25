# Vietnamese Spelling Recall Improvement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Increase Vietnamese spelling-correction recall by routing real-word and wrong-diacritic errors into evidence-aware correction lanes while preserving the current precision, deterministic-rule behavior, latency, and train/dev/test isolation.

**Architecture:** Keep the existing rule engine, SymSpell candidate pool and trigram LM. First make evaluation classify every miss by linguistic relation and production prefilter reason; then add two opt-in real-word lanes (same-key wrong diacritic, different-key contextual typo) behind shadow modes. Rank candidates with a small pairwise feature model trained only on VSEC train plus clean-train negatives, calibrate thresholds on dev, and activate a lane only when its precision/latency constraints pass.

**Tech Stack:** Node.js ES modules, built-in `node:test`, Python 3 standard library, immutable JSON model/config artifacts, existing VSEC grouped splits, clean-source splits, synthetic-diacritic dev corpus, trigram LM and benchmark tooling.

---

## 0. Evidence and decision

Current dev evidence (`dataset_artifacts/evaluation/spelling-eval-dev.json`):

| Item | Current value |
|---|---:|
| VSEC dev rows / labels | 934 / 1,115 |
| Candidate oracle coverage | 93.5% |
| Strict spelling P / R / F0.5 | 0.907 / 0.061 / 0.240 |
| Cross-lane P / R / F0.5 | 0.731 / 0.168 / 0.437 |
| Correct in spelling lane | 68 |
| Correct in another linguistic lane | 119 |
| Prefiltered | 852 |
| Candidate miss | 1 |
| Cheap-ranker loss | 16 |
| Unsupported multi-token | 59 |

The temporary PMD diagnostic found 571 uncredited same-accent-key labels with no PMD decision at all. The main recall wall is therefore lane eligibility/prefiltering, not n-gram order, candidate breadth or global confidence thresholds.

### Required strategy

1. Measure and split the 852 prefilters before changing emission behavior.
2. Route unaccented same-key corrections to missing-diacritic.
3. Route accented same-key corrections to a wrong-diacritic spelling lane.
4. Route different-key dictionary-word errors to a strongly gated real-word spelling lane.
5. Use pairwise candidate-vs-original evidence; never judge a real word from candidate probability alone.
6. Train only on VSEC train and clean-train negatives; tune only on dev.
7. Do not open VSEC test or Viwiki external-test until configuration is frozen.

### Non-goals

- Do not rebuild the LM in this plan.
- Do not increase n-gram order above trigram.
- Do not introduce attention/Transformer code yet.
- Do not lower the existing global thresholds as the first intervention.
- Do not emit two issues for the same token.
- Do not mutate raw datasets or hand-edit generated benchmark rows.

### Global acceptance gates

All accepted changes must satisfy:

- Core tests: 31/31 pass.
- Rule tests: 35/35 pass.
- Existing focused spelling tests all pass.
- Candidate oracle coverage remains >=93% (hard floor 92%).
- Synthetic missing-diacritic dev recall drops by <=0.5 percentage points.
- Cross-lane dev precision >=0.70.
- Cross-lane dev recall improves by at least 10 absolute points over 0.1677 before activation; target >=0.35 after all classical phases.
- Cross-lane F0.5 improves by at least 10% relative over 0.4371.
- No new issue on the existing red-team clean corpus.
- SMS <=160 character p95 <=15 ms with `node --expose-gc`.
- External test remains unopened until the final frozen run.

### Repository rule

This directory is not currently a Git repository. Do not run `git init`. The commit commands below are conditional checkpoints for a Git-enabled worktree; otherwise record completion in `docs/plans/execution-log-spelling-recall.md`.

## Phase 1: Make the recall wall observable

### Task 1: Add correction-relation and prefilter-reason taxonomy

**Files:**

- Create: `src/correction-taxonomy.mjs`
- Create: `test/test_correction_taxonomy.mjs`
- Modify: `tools/run_spelling_eval.mjs:129-242`
- Create: `tools/analyze_recall_failures.mjs`
- Create: `docs/plans/execution-log-spelling-recall.md`

**Step 1: Write taxonomy tests first**

Implement tests for these pure cases:

```js
assert.equal(classifyCorrectionRelation('khach', 'khách'), 'UNACCENTED_SAME_KEY');
assert.equal(classifyCorrectionRelation('quỳ', 'quý'), 'ACCENTED_SAME_KEY');
assert.equal(classifyCorrectionRelation('đế', 'đến'), 'DIFFERENT_KEY_SINGLE_TOKEN');
assert.equal(classifyCorrectionRelation('cảmơn', 'cảm ơn'), 'SPLIT');
assert.equal(classifyCorrectionRelation('cảm ơn', 'cảmơn'), 'MERGE');
assert.equal(classifyCorrectionRelation('abc', 'abc'), 'IDENTITY');
```

The implementation must normalize NFC and case but must not strip punctuation beyond the existing benchmark normalization policy.

**Step 2: Run the test and confirm RED**

```powershell
node --test test/test_correction_taxonomy.mjs
```

Expected: FAIL because `src/correction-taxonomy.mjs` does not exist.

**Step 3: Implement the pure taxonomy**

Export:

```js
export function classifyCorrectionRelation(input, target) {
  // IDENTITY | UNACCENTED_SAME_KEY | ACCENTED_SAME_KEY |
  // DIFFERENT_KEY_SINGLE_TOKEN | SPLIT | MERGE
}
```

Use existing `accentKey` and `hasVietnameseAccent`; do not duplicate Vietnamese normalization logic.

**Step 4: Add structured production prefilter reasons**

`evaluateSpellingToken` already returns `reason`. Extend the evaluator report to aggregate:

```json
{
  "prefilterReasons": {
    "classified-DICTIONARY": 0,
    "classified-UNACCENTED_VALID": 0,
    "classified-WHITELIST": 0,
    "classified-ABBREVIATION": 0,
    "min-token-length": 0,
    "all-caps": 0,
    "capitalized-proper-noun": 0,
    "no-context-anchor": 0
  },
  "relationStages": {
    "ACCENTED_SAME_KEY": { "prefilter": 0, "correct": 0 },
    "DIFFERENT_KEY_SINGLE_TOKEN": { "prefilter": 0, "correct": 0 }
  }
}
```

Do not change production decisions in this task.

**Step 5: Replace the temporary PMD diagnostic**

`tools/analyze_recall_failures.mjs --split dev` must call the same production decision functions and write an atomic report plus SHA-256 manifest under `dataset_artifacts/evaluation/`. It must reject `test` and `external-test` without `--final` using the existing split guard.

Required report fields:

- relation counts;
- stage counts per relation;
- prefilter reason counts;
- PMD eligibility/decision reason;
- target rank in wide pool, cheap pool and expensive ranker;
- at most 20 examples per bucket;
- code/config/LM hashes.

**Step 6: Run and record the immutable baseline**

```powershell
node --test test/test_correction_taxonomy.mjs test/test_eval_split_guard.mjs
node tools/analyze_recall_failures.mjs --split dev `
  --out dataset_artifacts/evaluation/recall-failures-baseline.json
```

Expected: totals reconcile exactly to 1,115 labels; no label is counted in two terminal stages.

**Step 7: Checkpoint**

```bash
git add src/correction-taxonomy.mjs test/test_correction_taxonomy.mjs tools/run_spelling_eval.mjs tools/analyze_recall_failures.mjs docs/plans/execution-log-spelling-recall.md
git commit -m "test: classify spelling recall failures by lane"
```

### Task 2: Separate product recall from benchmark rule-ID attribution

**Files:**

- Modify: `src/correction-taxonomy.mjs`
- Modify: `tools/run_spelling_eval.mjs:150-242`
- Modify: `benchmark/run-benchmark.mjs:32-80`
- Test: `test/test_semantic_correction_matching.mjs`

**Reason:** VSEC labels every correction as `POSSIBLE_SPELLING_ERROR`, while the product intentionally emits unaccented same-key corrections as `POSSIBLE_MISSING_DIACRITIC`. This makes strict rule-ID recall useful for contract compatibility but unsuitable as the primary product-recall metric.

**Step 1: Write failing semantic-match tests**

Cover:

- `khach -> khách` is credited when PMD emits `khách`;
- `quỳ -> quý` is credited when spelling emits `quý`;
- wrong value or wrong suggestion is not credited;
- deterministic issues never count as linguistic corrections;
- strict matching remains unchanged.

**Step 2: Run RED test**

```powershell
node --test test/test_semantic_correction_matching.mjs
```

**Step 3: Add an explicit semantic matcher**

Export a pure helper such as:

```js
export function linguisticCorrectionMatches(issue, expected) {
  if (!['POSSIBLE_MISSING_DIACRITIC', 'POSSIBLE_SPELLING_ERROR']
    .includes(issue.ruleId)) return false;
  return normalizedValueMatches(issue, expected)
    && normalizedSuggestionMatches(issue, expected);
}
```

Keep `benchmarkIssueMatches` strict. Do not silently change the existing benchmark contract. Add a separately named `semanticLinguistic` metric block to dev evaluation and optional benchmark reports.

**Step 4: Add relation-aware reporting**

Report three views side by side:

1. `strictRuleId`: backwards-compatible contract score;
2. `semanticLinguistic`: primary product correction score;
3. `valueOnlyDiagnostic`: diagnostic only, never an acceptance metric.

**Step 5: Verify without changing engine output**

```powershell
node --test test/test_semantic_correction_matching.mjs test/test_benchmark_scope.mjs
npm.cmd run eval:spelling:dev
```

Expected: engine issue set byte-identical before/after; only evaluation fields change.

**Step 6: Checkpoint**

```bash
git add src/correction-taxonomy.mjs tools/run_spelling_eval.mjs benchmark/run-benchmark.mjs test/test_semantic_correction_matching.mjs
git commit -m "test: report semantic linguistic correction recall"
```

## Phase 2: Open the blocked correction lanes safely

### Task 3: Extract candidate construction from emission policy

**Files:**

- Modify: `src/rules/linguistic-rules.mjs:461-684`
- Create: `test/test_correction_candidate_builder.mjs`
- Modify: `tools/audit_candidate_recall.mjs`

**Step 1: Write failing tests for a pure candidate builder**

The new exported helper must accept a token plus lane and return candidates without deciding whether to warn:

```js
const result = buildCorrectionCandidates({
  token, lane: 'ACCENTED_SAME_KEY', services, snap,
});
assert.ok(result.entries.some((c) => c.word === 'quý'));
assert.equal(result.entries.some((c) => c.word === token.normalized), false);
```

Test these lanes independently:

- `UNKNOWN_TYPO`: current SymSpell behavior is preserved;
- `UNACCENTED_SAME_KEY`: accent-family entries only;
- `ACCENTED_SAME_KEY`: all other same-key surfaces, regardless of dictionary validity;
- `DIFFERENT_KEY_REAL_WORD`: SymSpell alternatives for a valid dictionary word;
- protected/whitelisted/abbreviation tokens return an ineligible policy before construction.

**Step 2: Run RED test**

```powershell
node --test test/test_correction_candidate_builder.mjs
```

**Step 3: Implement the extraction with no output change**

Introduce data-only structures:

```js
{
  lane,
  eligible,
  reason,
  entries: [{ word, stripped, dist, freq, sameAccentKey }],
  original: token.normalized
}
```

`evaluateSpellingToken` must consume this helper for the existing unknown-token lane. Keep real-word lanes disabled, so production issues remain identical.

**Step 4: Extend candidate audit per lane**

Report oracle coverage separately for same-key accented, unaccented same-key, different-key real-word and unknown typo. Do not combine them into one number only.

**Step 5: Verify behavior preservation**

```powershell
node --test test/test_correction_candidate_builder.mjs `
  test/test_candidate_generation.mjs test/test_context_reranker.mjs
npm.cmd test
npm.cmd run eval:spelling:dev
```

Expected: exact same issue set and metrics as Task 1 baseline.

**Step 6: Checkpoint**

```bash
git add src/rules/linguistic-rules.mjs tools/audit_candidate_recall.mjs test/test_correction_candidate_builder.mjs
git commit -m "refactor: separate correction candidates from emission policy"
```

### Task 4: Add an accented same-key wrong-diacritic lane in SHADOW

**Files:**

- Create: `src/context-evidence.mjs`
- Create: `test/test_context_evidence.mjs`
- Create: `test/test_wrong_diacritic_lane.mjs`
- Modify: `src/rules/linguistic-rules.mjs:469-684`
- Modify: `src/config.mjs:25-76`
- Modify: `tools/run_spelling_eval.mjs`

**Step 1: Write failing feature-extraction tests**

For a tiny hand-built LM, verify candidate-vs-original features exactly:

```js
{
  unigramLogRatio,
  leftBigramLogRatio,
  rightBigramLogRatio,
  centeredTrigramLogRatio,
  forwardTrigramLogRatio,
  backwardTrigramLogRatio,
  candidateAttestedWindows,
  originalAttestedWindows,
  sameAccentKey,
  editDistance
}
```

All count ratios must use `log1p(candidateCount) - log1p(originalCount)` and remain finite at zero. Context tokens are fixed observed/beam surfaces; never divide a centered trigram by a non-contiguous bigram.

**Step 2: Write lane tests before implementation**

Cover:

- accented dictionary token can reach `ACCENTED_SAME_KEY` candidates;
- candidate with direct contextual wins can become `wouldEmit=true` in SHADOW;
- original with equal/better evidence stays silent;
- `bảo hành chính hãng` must never suggest `hàng`;
- whitelist, abbreviation, protected, all-caps and mid-sentence proper noun remain blocked;
- no duplicate PMD/PSE issue for one span.

**Step 3: Run RED tests**

```powershell
node --test test/test_context_evidence.mjs test/test_wrong_diacritic_lane.mjs
```

**Step 4: Add an explicit feature mode**

Replace the ambiguous boolean `wrongToneEnabled` with:

```js
wrongDiacriticMode: 'SHADOW' // OFF | SHADOW | ACTIVE
```

Migration behavior: absent key is `OFF`; `SHADOW` calculates and reports a decision but `createPossibleSpellingErrorRule` returns no user-visible issue. Evaluation tooling must still score `wouldEmit`.

**Step 5: Implement a conservative provisional gate**

Before the learned calibrator exists, `wouldEmit` may be true only when:

- best candidate has the same accent key;
- candidate has at least one directly attested bigram/trigram window;
- original has zero directly attested trigram wins in the same windows;
- pairwise score margin is positive;
- candidate frequency passes the existing floor.

Do not activate this provisional gate. Its purpose is to collect shadow features and establish a ceiling.

**Step 6: Measure SHADOW headroom on dev and clean negatives**

```powershell
npm.cmd run eval:spelling:dev
node tools/analyze_recall_failures.mjs --split dev `
  --out dataset_artifacts/evaluation/recall-wrong-diacritic-shadow.json
```

Record:

- eligible labels;
- oracle coverage;
- shadow TP/FP;
- feature distributions for TP vs FP;
- how many of the 571 `noPmdLine` misses become reachable.

**Step 7: Checkpoint**

```bash
git add src/context-evidence.mjs src/rules/linguistic-rules.mjs src/config.mjs tools/run_spelling_eval.mjs test/test_context_evidence.mjs test/test_wrong_diacritic_lane.mjs
git commit -m "feat: add shadow wrong-diacritic correction lane"
```

### Task 5: Add a different-key real-word typo lane in SHADOW

**Files:**

- Create: `test/test_real_word_typo_lane.mjs`
- Modify: `src/rules/linguistic-rules.mjs`
- Modify: `src/config.mjs`
- Modify: `tools/run_spelling_eval.mjs`

**Step 1: Write failing real-word tests**

Use tiny deterministic LMs for examples shaped like:

- `đế -> đến` when `đến` is contextually attested;
- `Các -> Cách` at sentence start when candidate wins decisively;
- valid real word remains unchanged without direct context proof;
- foreign/product/proper-name guards remain unchanged;
- distance-two candidate is not allowed for tokens shorter than the existing `maxEditDistanceFor` policy.

**Step 2: Run RED test**

```powershell
node --test test/test_real_word_typo_lane.mjs
```

**Step 3: Add separate mode and tighter eligibility**

```js
realWordTypoMode: 'SHADOW' // OFF | SHADOW | ACTIVE
```

A dictionary token may enter this lane only when:

- it is not protected, whitelisted or abbreviation;
- capitalization guard passes;
- at least one neighbor is anchorable;
- candidate edit distance obeys `maxEditDistanceFor`;
- candidate is a corpus-backed dictionary entry;
- candidate has direct bigram/trigram context evidence, not unigram frequency alone.

Do not reuse the unknown-token emission threshold. Real-word errors need a separate pairwise threshold because the original is itself plausible.

**Step 4: Preserve original and hard negatives**

Always score the original beside candidates. Add the highest-frequency same-distance rival and the best context-attested rival as hard negatives so the later trainer sees realistic mistakes.

**Step 5: Measure in SHADOW only**

```powershell
node --test test/test_real_word_typo_lane.mjs
npm.cmd run eval:spelling:dev
node tools/analyze_recall_failures.mjs --split dev `
  --out dataset_artifacts/evaluation/recall-real-word-shadow.json
```

Expected: `classified-DICTIONARY` prefilters become reachable; production issue set remains unchanged.

**Step 6: Checkpoint**

```bash
git add src/rules/linguistic-rules.mjs src/config.mjs tools/run_spelling_eval.mjs test/test_real_word_typo_lane.mjs
git commit -m "feat: add shadow real-word typo lane"
```

## Phase 3: Learn a small precision-safe pairwise decision

### Task 6: Train and load a lightweight pairwise recall reranker

**Files:**

- Create: `tools/train_recall_reranker.py`
- Create: `src/recall-reranker.mjs`
- Create: `src/data/recall-reranker.json` (generated)
- Create: `src/data/recall-reranker.manifest.json` (generated)
- Create: `test/test_recall_reranker.mjs`
- Create: `test/test_recall_reranker_training.py`
- Modify: `src/engine.mjs:104-155`
- Modify: `src/rules/linguistic-rules.mjs`

**Why a small model:** Global threshold sweeps were neutral because 76% of labels never reached the scorer. Once the real-word lanes are open, a pairwise candidate-vs-original model can learn how much direct context evidence is enough without adding an attention runtime or replacing deterministic rules.

**Step 1: Freeze the feature contract in tests**

Use this ordered feature vector, versioned as `recall-pairwise-v1`:

```text
1  bias
2  editDistance
3  sameAccentKey
4  candidateMinusOriginalLogFrequency
5  leftBigramLogRatio
6  rightBigramLogRatio
7  centeredTrigramLogRatio
8  forwardTrigramLogRatio
9  backwardTrigramLogRatio
10 candidateAttestedWindows
11 originalAttestedWindows
12 tokenLength
13 originalIsDictionary
14 candidateCheapRank
15 candidatePoolRank
```

Tests must verify feature order, finite output, missing feature rejection, model-version rejection and deterministic dot-product probability.

**Step 2: Build leakage-safe training examples**

Positive pairwise rows come only from `dataset_artifacts/vsec-train.jsonl`:

- candidate = labeled target;
- original = noisy token;
- include only single-token labels whose target exists in the production oracle;
- extract features using production candidate/context helpers.

Hard negatives:

- original token against the target;
- top frequency rival at the same edit distance;
- top context-attested wrong rival;
- unchanged tokens from `dataset_artifacts/clean-source/clean-train.txt`.

Never read `vsec-dev.jsonl`, `vsec-test.jsonl`, Viwiki or benchmark JSON in the trainer. Fail closed if any forbidden path appears in resolved inputs.

**Step 3: Implement deterministic Python training**

Use Python standard library only:

- standardize continuous features and export means/scales;
- fixed input ordering and fixed seed;
- logistic loss with L2 regularization;
- fixed iteration count and learning-rate schedule;
- class weights recorded in the artifact;
- export JSON atomically plus a manifest of source hashes, counts and command.

The artifact must contain weights only, not corpus text or raw examples.

**Step 4: Test determinism and leakage guard**

```powershell
python -m unittest test.test_recall_reranker_training -v
python tools/train_recall_reranker.py --fixture test/fixtures/recall-reranker `
  --out .tmp/recall-reranker-a.json
python tools/train_recall_reranker.py --fixture test/fixtures/recall-reranker `
  --out .tmp/recall-reranker-b.json
```

Expected: the two model files are byte-identical; a forbidden test path fails non-zero.

**Step 5: Implement immutable JS inference**

`RecallReranker.load()` validates schema/version/feature order and fails fast on a present-but-malformed artifact. `score(features)` returns a calibrated sigmoid probability without allocations proportional to vocabulary size.

Inject one immutable instance through `SmsValidationEngine`; never read the artifact per token.

**Step 6: Train the real artifact**

```powershell
python tools/train_recall_reranker.py `
  --vsec-train dataset_artifacts/vsec-train.jsonl `
  --clean-train dataset_artifacts/clean-source/clean-train.txt `
  --out src/data/recall-reranker.json
node --test test/test_recall_reranker.mjs
```

**Step 7: Keep both lanes in SHADOW**

Record model probability and proposed candidate in dev reports but do not emit user-visible issues yet.

**Step 8: Checkpoint**

```bash
git add tools/train_recall_reranker.py src/recall-reranker.mjs src/data/recall-reranker.json src/data/recall-reranker.manifest.json src/engine.mjs src/rules/linguistic-rules.mjs test/test_recall_reranker.mjs test/test_recall_reranker_training.py
git commit -m "feat: add leakage-safe pairwise correction reranker"
```

### Task 7: Protect oracle candidates from cheap-ranker pruning

**Files:**

- Modify: `src/rules/linguistic-rules.mjs:568-612`
- Modify: `tools/audit_candidate_recall.mjs`
- Create: `test/test_candidate_shortlist.mjs`

**Step 1: Write a failing hard-negative fixture**

Create a pool where the correct candidate is rank 4 by cheap frequency but is the only candidate with direct context evidence. Assert it survives the expensive shortlist without increasing the bound beyond four non-original candidates.

**Step 2: Implement a diversity-preserving shortlist**

Select the union, deduplicated in deterministic order, of:

1. best edit-distance candidate;
2. best direct-context-attested candidate;
3. best same-accent-key candidate when applicable;
4. highest remaining cheap score.

Do not peek at the label/target. The selection uses production features only.

**Step 3: Verify the narrow cascade**

```powershell
node --test test/test_candidate_shortlist.mjs test/test_candidate_generation.mjs
node tools/audit_candidate_recall.mjs --split dev
npm.cmd run profile:engine
```

Expected: cheap-ranker-loss is below 16, expensive non-original candidates <=4, candidate oracle >=93%, SMS p95 <=15 ms.

**Step 4: Checkpoint**

```bash
git add src/rules/linguistic-rules.mjs tools/audit_candidate_recall.mjs test/test_candidate_shortlist.mjs
git commit -m "perf: preserve contextual candidates in narrow shortlist"
```

## Phase 4: Optional word-boundary recovery

### Task 8: Add bounded split/merge candidates only if single-token target is insufficient

**Condition:** Execute this task only if Tasks 1–7 pass all precision gates but semantic dev recall remains below 0.35. Unsupported multi-token labels are only 59/1,115, so this work must not delay the higher-value real-word lanes.

**Files:**

- Create: `src/word-boundary-candidates.mjs`
- Create: `test/test_word_boundary_candidates.mjs`
- Modify: `src/rules/linguistic-rules.mjs`
- Modify: `src/engine.mjs` conflict handling if required
- Modify: `tools/run_spelling_eval.mjs`

**Step 1: Write offset and guard tests first**

Cover:

- one token can suggest two corpus-backed words (`cảmơn -> cảm ơn`);
- two adjacent words can suggest one corpus-backed word;
- issue span covers the exact original substring, including inter-token whitespace for merge;
- URL/code/placeholder ranges are never split/merged;
- no overlap with deterministic whitespace issues unless conflict priority explicitly keeps both;
- candidate count is bounded and deterministic.

**Step 2: Run RED test**

```powershell
node --test test/test_word_boundary_candidates.mjs
```

**Step 3: Implement bounded generation**

- Split only at internal character boundaries producing two corpus-backed words.
- Merge only an adjacent word pair separated by ordinary whitespace.
- Keep at most four split and two merge candidates.
- Rank using edit cost, word frequencies and existing left/right n-gram context.
- Default `wordBoundaryCorrectionMode: 'SHADOW'`.

Do not implement arbitrary sequence-to-sequence rewriting.

**Step 4: Measure and activate only on a net win**

```powershell
node --test test/test_word_boundary_candidates.mjs
npm.cmd run eval:spelling:dev
npm.cmd run profile:engine
```

Accept only if semantic F0.5 increases, cross-lane precision stays >=0.70 and deterministic whitespace behavior is identical.

**Step 5: Checkpoint**

```bash
git add src/word-boundary-candidates.mjs src/rules/linguistic-rules.mjs src/engine.mjs tools/run_spelling_eval.mjs test/test_word_boundary_candidates.mjs
git commit -m "feat: add bounded word-boundary correction candidates"
```

## Phase 5: Calibrate, activate and freeze

### Task 9: Calibrate each recall lane on dev and activate independently

**Files:**

- Create: `tools/calibrate_recall_lanes.mjs`
- Create: `test/test_recall_calibration.mjs`
- Modify: `config/spelling-tuning.json`
- Modify: `src/config.mjs` only to apply the accepted frozen values
- Modify: `docs/plans/execution-log-spelling-recall.md`

**Step 1: Write deterministic selection tests**

Given fixture trials, the selector must:

1. reject any trial violating a hard constraint;
2. maximize semantic F0.5;
3. break ties by higher precision, then lower latency, then lexical config order;
4. produce byte-identical output on repeated runs.

**Step 2: Define the dev-only search**

Search one lane at a time:

```text
wrongDiacriticMode: SHADOW -> ACTIVE candidate
realWordTypoMode: SHADOW -> ACTIVE candidate
pairwise probability threshold per lane
minimum candidate-attested windows
maximum original-attested windows
minimum pairwise score margin
```

Do not combine threshold search with LM rebuilding. Do not search global PMD/spelling thresholds unless the lane-specific search has passed first.

**Step 3: Evaluate clean constraints per trial**

Each trial must measure:

- semantic correction P/R/F0.5 on VSEC dev;
- relation-specific recall;
- fully-labelled VSEC dev false positives;
- clean-source dev issue rate;
- red-team clean issue count;
- synthetic missing-diacritic recall;
- SMS <=160 p95.

**Step 4: Apply lanes incrementally**

Activation order:

1. `ACCENTED_SAME_KEY` wrong-diacritic lane;
2. `DIFFERENT_KEY_REAL_WORD` lane;
3. optional word-boundary lane.

After each activation rerun all constraints. If a later lane fails, retain the earlier accepted lane rather than reverting the whole improvement.

**Step 5: Run calibration**

```powershell
node --test test/test_recall_calibration.mjs
node tools/calibrate_recall_lanes.mjs `
  --split dev `
  --out dataset_artifacts/evaluation/recall-calibration.json
```

Expected acceptance target after classical phases:

- semantic recall >=0.35;
- semantic precision >=0.70;
- semantic F0.5 >=0.4808 (10% relative improvement);
- all global gates remain green.

If no feasible trial exists, leave that lane in SHADOW and record the failed constraint. Never force an active winner.

**Step 6: Freeze configuration**

Update `config/spelling-tuning.json` with:

- `frozen: true`;
- model and code hashes;
- accepted lane modes and thresholds;
- dev metrics and constraints;
- calibration artifact hash.

**Step 7: Checkpoint**

```bash
git add tools/calibrate_recall_lanes.mjs test/test_recall_calibration.mjs config/spelling-tuning.json src/config.mjs docs/plans/execution-log-spelling-recall.md
git commit -m "feat: calibrate and freeze recall-safe spelling lanes"
```

### Task 10: Run acceptance gates and one final held-out evaluation

**Files:**

- Create: `dataset_artifacts/evaluation/final-recall-dev-report.json`
- Create: `dataset_artifacts/evaluation/final-recall-performance.json`
- Create once: `dataset_artifacts/evaluation/final-recall-heldout.json`
- Modify: `README.md`
- Modify: `docs/plans/execution-log-spelling-recall.md`

**Step 1: Confirm the freeze before any held-out access**

Verify `config/spelling-tuning.json` has `frozen: true` and hashes match current source/model artifacts. Abort on mismatch.

**Step 2: Run all development gates**

```powershell
npm.cmd test
node --test test/test_viwiki_converter.mjs `
  test/test_language_model_loader.mjs `
  test/test_language_scoring.mjs `
  test/test_candidate_generation.mjs `
  test/test_context_reranker.mjs `
  test/test_symspell_index.mjs `
  test/test_eval_split_guard.mjs `
  test/test_tuning_determinism.mjs `
  test/test_correction_taxonomy.mjs `
  test/test_semantic_correction_matching.mjs `
  test/test_correction_candidate_builder.mjs `
  test/test_context_evidence.mjs `
  test/test_wrong_diacritic_lane.mjs `
  test/test_real_word_typo_lane.mjs `
  test/test_recall_reranker.mjs `
  test/test_candidate_shortlist.mjs `
  test/test_recall_calibration.mjs
python -m unittest test.test_recall_reranker_training -v
npm.cmd run data:audit
node tools/audit_candidate_recall.mjs --split dev
npm.cmd run eval:spelling:dev
npm.cmd run profile:engine
```

If Task 8 was executed, include `test/test_word_boundary_candidates.mjs`.

**Step 3: Compare against every acceptance gate**

Do not proceed if any hard gate fails. Fixes after this point must use train/dev only and require a new freeze; held-out data must remain unopened.

**Step 4: Run held-out exactly once**

```powershell
npm.cmd run eval:spelling:test
npm.cmd run bench
```

The report must clearly separate VSEC test, Viwiki external-test, synthetic missing-diacritic, deterministic categories and clean false positives. Record source/config/model/LM hashes.

If held-out performance is below target, record it as a failed generalization result. Do not tune against held-out examples and rerun.

**Step 5: Update README honestly**

Document:

- which lanes are ACTIVE/SHADOW/OFF;
- dev and held-out semantic/strict metrics;
- overall precision and deterministic regressions;
- p50/p95/p99, startup and RSS;
- known unsupported categories;
- exact artifact hashes.

**Step 6: Final checkpoint**

```bash
git add README.md docs/plans/execution-log-spelling-recall.md dataset_artifacts/evaluation/final-recall-*.json
git commit -m "docs: record final spelling recall evaluation"
```

## Stop/go decision for attention

Do not add attention inside this plan. Open a separate attention-reranker milestone only if all are true:

1. same-key and different-key lanes are reachable;
2. pairwise classical reranker has been calibrated correctly;
3. candidate oracle remains >=93%;
4. semantic recall is still below 0.35 at precision >=0.70;
5. failure analysis shows `trigram-ranker-loss` or `wrong-suggestion` dominates rather than prefiltering;
6. a deploy budget for model bytes, cold start and p95 has been agreed.

The attention model, if later approved, should rerank at most four frozen candidates and must never generate arbitrary text.

## Required execution report

After every task, append to `docs/plans/execution-log-spelling-recall.md`:

```markdown
## Task N
- Status: DONE | BLOCKED | REJECTED
- Source hashes:
- Tests run:
- Dev metrics before/after:
- Precision constraints:
- Latency/RSS:
- Artifacts and hashes:
- Deviations:
- Next task decision:
```

Do not report a task DONE merely because tests pass. A recall task is DONE only when its stated metric and safety gates pass.

## Handoff prompt

Give the coding agent this instruction:

> Read `docs/plans/2026-08-24-spelling-recall-improvement.md` completely. Use `superpowers:executing-plans` and execute one task at a time with TDD. Use Semble before locating code. Do not initialize Git, do not edit raw/test datasets, do not open held-out or external-test data before Task 10, and do not activate a recall lane unless every precision and latency gate passes. Record each checkpoint in `docs/plans/execution-log-spelling-recall.md`.
