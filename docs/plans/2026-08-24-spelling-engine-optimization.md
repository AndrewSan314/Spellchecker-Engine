# Vietnamese Spelling Engine Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve Vietnamese spelling-correction accuracy, latency, startup time, and memory usage without reintroducing train/test leakage or reducing deterministic-rule reliability.

**Architecture:** Keep trigram as the highest-order statistical model, but separate candidate generation, cheap candidate filtering, expensive context reranking, and storage. Correct evaluation and probability defects first; then improve candidate recall and ranking; finally replace string-keyed JavaScript `Map` storage with a versioned compact word-ID backend. Tune only on train/dev and run external tests once after configuration is frozen.

**Tech Stack:** Node.js ES modules, Python 3, built-in `node:test`, Python `unittest`, deterministic n-gram artifacts, typed arrays/binary files, existing VSEC/Viwiki/synthetic datasets.

---

## 0. Execution rules and current baseline

### Required workflow

1. Read `AGENTS.md` and applicable skills before editing.
2. Use Semble first for every new code-location or behavior search.
3. Use GitNexus/CodeGraph impact checks if an index exists.
4. Use TDD: failing test, minimal implementation, passing test, then next change.
5. Do not modify raw parquet/JSON downloads.
6. Do not use test or external benchmark labels for tuning.
7. Do not read or merge the historical leaked LM into a new artifact.
8. Every generated artifact must be written atomically and have a manifest/hash.
9. The current workspace is not a Git repository. Do not run `git init`. Commit steps below apply only when executing in a Git-enabled copy/worktree; otherwise record each checkpoint in the execution log.

### Baseline artifacts

- Active LM: `src/data/lm-ngrams.tsv`
- Active LM manifest: `src/data/lm-ngrams.manifest.json`
- Active LM SHA-256: `791f4525653489ef98dae1ed23e238ab4aac4af90df2f8a4378e09ae78560ce6`
- LM size: `79,874,419` bytes
- Runtime entries: `U=200,000`, `B=1,000,000`, `T=2,760,041`
- Weighted token count: `1,325,600,394`
- Full unigram coverage: 17 news shards
- Higher-order coverage: 4 deterministic news shards plus Wikipedia, clean domain train, and VSEC train
- Held-out exact intersections: zero for clean-dev, clean-test, synthetic, Viwiki external, VSEC dev, and VSEC test

### Measured baseline

| Metric | Current |
|---|---:|
| Overall recall | 65.85% |
| Overall precision | 89.87% |
| Sentence-perfect | 38.85% |
| Missing-diacritic recall | 94.47% |
| VSEC spelling recall | 29.35% |
| Viwiki detection recall | 25.45% |
| Overall p95 | 25.39 ms |
| p95, 0-80 characters | 10.29 ms |
| p95, 81-160 characters | 21.38 ms |
| p95, 161-300 characters | 31.70 ms |
| Isolated LM load | ~3.23 s / ~687 MB RSS |
| Full engine startup | ~4.25 s / ~793 MB RSS |

### Profiling evidence

- `scoreCandidateOverSurfaces`: 264,345 calls and about 13.8 seconds of a 20-second instrumented benchmark.
- `_pBiRaw`: 12.5 million calls.
- Bigram lookup hit rate on 100 VSEC rows: 14.56%.
- Trigram lookup hit rate on 100 VSEC rows: 1.26%.
- Current candidate oracle coverage on VSEC: about 80.9%.
- Simulated candidate key coverage: top-12 keys 92.0%; top-20 keys 93.94%.
- SymSpell index: 680,258 delete variants and 993,432 variant-to-key edges.
- Length-aware delete depth simulation reduces variants 17.5% and edges 34.4%.
- String `Map` memory after streaming load: U ~92 MB RSS, U+B ~267 MB, U+B+T ~577 MB.

### Global definition of done

- All existing core/rule/data/LM tests pass.
- Viwiki benchmark retains labeled suggestions and tests correction, not only detection.
- No silent LM fallback on invalid/corrupt artifacts.
- Centered trigram scoring has a mathematically valid formulation.
- VSEC dev candidate oracle coverage is at least 92%.
- VSEC dev spelling recall and precision improve without reducing missing-diacritic recall by more than 0.5 percentage points.
- Final configuration is selected using train/dev only.
- Final VSEC test and Viwiki external reports are generated once after freeze.
- SMS-length (`<=160` chars) p95 is at most 15 ms on the same machine.
- Full engine startup is at most 2 seconds and steady RSS is at most 400 MB; if this target cannot be met, document the measured blocker before changing the target.
- LM artifact remains at most 100 MB and is loadable without fallback.
- Raw datasets remain byte-identical.

---

## Phase 1: Correctness and evaluation gates

### Task 1: Preserve Viwiki correction suggestions

**Files:**

- Modify: `tools/convert_viwiki_spelling.mjs:37-50`
- Test: `test/test_viwiki_converter.mjs`
- Regenerate: `benchmark/corpus-viwiki-spelling.json`

**Problem:** `validMistake()` returns only `{startOffset, value}`. `addMistake()` later reads `mistake.suggest`, so every suggestion is lost. The current Viwiki metric measures detection only.

**Step 1: Write the failing test**

Export or test a pure helper with a small inline JSONL fixture containing one sentence and one mistake:

```js
test('converter preserves all normalized suggestions', () => {
  const row = convertDocumentFixture({
    text: 'Kính chao quý khách.',
    mistakes: [{ start_offset: 5, text: 'chao', suggest: ['chào'] }],
  });
  assert.deepEqual(row.expect[0].suggestions, ['chào']);
  assert.equal(row.expect[0].suggestion, 'chào');
});
```

Also add tests for:

- multiple mistakes in one sentence;
- multiple suggestions for one mistake;
- identity suggestion such as `ràng -> ràng` being flagged/skipped according to an explicit policy;
- offset preservation after sentence trimming.

**Step 2: Run the failing test**

Run:

```powershell
node --test test/test_viwiki_converter.mjs
```

Expected: FAIL because `suggestions` is empty.

**Step 3: Implement the minimal fix**

Make `validMistake()` retain a defensive copy of suggestions:

```js
return {
  startOffset,
  value,
  suggestions: Array.isArray(mistake.suggest)
    ? mistake.suggest.filter((value) => typeof value === 'string' && value.trim())
    : [],
};
```

Update `addMistake()` to read the retained `suggestions` property. Do not read benchmark output as training data.

**Step 4: Regenerate and verify**

Run:

```powershell
node tools/convert_viwiki_spelling.mjs
node --test test/test_viwiki_converter.mjs
npm.cmd run data:audit
```

Expected:

- 150 external rows remain external-only;
- at least one non-empty expected suggestion exists;
- `corpus-train ∩ Viwiki external = 0`;
- data audit passes.

**Step 5: Commit/checkpoint**

```bash
git add tools/convert_viwiki_spelling.mjs test/test_viwiki_converter.mjs benchmark/corpus-viwiki-spelling.json
git commit -m "fix: preserve Viwiki spelling suggestions"
```

### Task 2: Make LM loading fail fast and observable

**Files:**

- Modify: `src/language.mjs:92-109`
- Test: `test/test_language_model_loader.mjs`

**Problem:** `NGramLanguageModel.load()` catches every error and silently falls back to `corpus-train.txt`. This previously hid an invalid 650 MB artifact and produced incomparable benchmarks.

**Step 1: Write failing tests**

Cover three cases:

1. Missing artifact may use the documented raw fallback.
2. Malformed artifact must throw an error containing the artifact path and line/reason.
3. Oversized/unreadable artifact must not silently load raw corpus.

Use dependency injection or a loader option instead of renaming production files.

**Step 2: Run tests and confirm failure**

```powershell
node --test test/test_language_model_loader.mjs
```

Expected: malformed artifact currently falls back instead of throwing.

**Step 3: Implement explicit error handling**

- Fall back only for `ENOENT` when fallback is explicitly enabled.
- Validate header, token count, kind, tab count, integer count, unique key, and section ordering.
- Wrap parse errors as `LanguageModelArtifactError`.
- Log/return selected backend and artifact hash in diagnostics.

Do not use a blanket `catch {}`.

**Step 4: Verify**

```powershell
node --test test/test_language_model_loader.mjs
npm.cmd test
python tools/audit_lm.py
```

**Step 5: Commit/checkpoint**

```bash
git add src/language.mjs test/test_language_model_loader.mjs
git commit -m "fix: fail fast on invalid language model artifacts"
```

### Task 3: Correct centered trigram scoring

**Files:**

- Modify: `src/language.mjs:255-309`
- Test: `test/test_language_scoring.mjs`
- Possibly modify: `src/config.mjs`

**Problem:** centered scoring currently divides `count(prev, candidate, next)` by `count(prev, next)`. The denominator is a non-contiguous neighbor bigram and is not the conditioning context of the trigram.

**Step 1: Write failing mathematical tests**

Build a tiny synthetic LM where counts are fully known. Assert that:

- increasing `count(prev,candidate,next)` increases candidate score;
- unrelated `count(prev,next)` does not decrease centered candidate evidence;
- left-to-right factorization ranks an attested candidate above an unattested candidate;
- scores are finite when counts are zero.

**Step 2: Replace the invalid centered term**

Use a valid factorization, preferably:

```text
log P(candidate | prev) + log P(next | prev, candidate)
```

The implementation should reuse `_pBiRaw(candidate, prev)` and `_pTriRaw(next, candidate, prev)`. Avoid double-counting if equivalent bigram terms already exist; put weights in config rather than new hard-coded constants.

Do not introduce Kneser-Ney in this task. This task changes one formula only.

**Step 3: Run focused and regression tests**

```powershell
node --test test/test_language_scoring.mjs
npm.cmd test
npm.cmd run data:audit
```

**Step 4: Record dev-only metrics**

Run the dev evaluation created in Task 8. Do not open VSEC test/Viwiki external results while tuning this change.

**Step 5: Commit/checkpoint**

```bash
git add src/language.mjs src/config.mjs test/test_language_scoring.mjs
git commit -m "fix: use valid centered trigram factorization"
```

---

## Phase 2: Candidate recall and runtime cascade

### Task 4: Normalize Telex before candidate lookup

**Files:**

- Modify: `src/rules/linguistic-rules.mjs:497-510`
- Modify if necessary: `src/language.mjs:415-470`
- Test: `test/test_candidate_generation.mjs`

**Problem:** Telex normalization is currently applied only when scoring candidates, after SymSpell lookup. Inputs such as `dawnf` cannot generate `dằn` even though the edit scorer understands Telex.

**Step 1: Write failing cases**

Include at least:

- `dawnf -> dằn`;
- `ddat -> đặt` or another unambiguous Telex base-form case;
- ordinary non-Telex typos remain unchanged;
- protected codes/URLs are not normalized as spelling tokens.

**Step 2: Implement lookup-key normalization**

- Preserve the original token for offsets and messages.
- Derive a Telex-normalized lookup surface before `accentKey()` and delete generation.
- Do not mutate `ValidationContext.content`.

**Step 3: Verify**

```powershell
node --test test/test_candidate_generation.mjs
npm.cmd test
```

**Step 4: Commit/checkpoint**

```bash
git add src/language.mjs src/rules/linguistic-rules.mjs test/test_candidate_generation.mjs
git commit -m "feat: normalize Telex before candidate lookup"
```

### Task 5: Increase candidate oracle coverage without increasing expensive scoring

**Files:**

- Modify: `src/language.mjs:415-470`
- Modify: `src/rules/linguistic-rules.mjs:490-565`
- Create: `tools/audit_candidate_recall.mjs`
- Test: `test/test_candidate_generation.mjs`

**Problem:** SymSpell keeps six stripped keys and three surfaces per key. Correct VSEC targets are dropped before ranking. Expanding every candidate directly into trigram scoring would worsen latency.

**Step 1: Add a candidate-oracle audit**

The script must read VSEC dev only and report:

- total labeled corrections;
- accent-family coverage;
- SymSpell top-6/top-12/top-20 key coverage;
- surface-pruning losses;
- Telex/multi-token/unsupported losses.

It must reject test/external input unless an explicit final-report flag is supplied after configuration freeze.

**Step 2: Write failing coverage tests**

Use fixed synthetic lexicon fixtures proving that:

- an unaccented correct surface is not removed merely because three accented siblings are more frequent;
- top-12 key generation includes a valid transposition candidate;
- returned candidates are deterministic;
- maximum returned candidates remains bounded.

**Step 3: Split generation from expensive ranking**

Return a wider cheap pool, for example:

- up to 12 stripped keys;
- enough surfaces to retain the raw/unaccented surface plus high-frequency accented siblings;
- metadata containing edit distance and frequency.

Do not call trigram scoring yet.

**Step 4: Add a cheap first-stage ranker**

Rank the wide pool using only:

- edit distance;
- Telex-aware distance;
- unigram/lexicon frequency;
- direct accent-family relationship;
- optionally one cheap bigram attestation check.

Keep only top 3-4 candidates for expensive trigram reranking.

**Step 5: Verify coverage and latency**

```powershell
node tools/audit_candidate_recall.mjs --split dev
node --test test/test_candidate_generation.mjs
npm.cmd test
```

Expected:

- VSEC dev oracle coverage >=92%;
- at most four candidates enter expensive context scoring;
- deterministic ordering across repeated runs.

**Step 6: Commit/checkpoint**

```bash
git add src/language.mjs src/rules/linguistic-rules.mjs tools/audit_candidate_recall.mjs test/test_candidate_generation.mjs
git commit -m "perf: add wide candidate generation and narrow reranking cascade"
```

### Task 6: Remove the surface Cartesian-product hotspot

**Files:**

- Modify: `src/language.mjs:255-333`
- Modify: `src/rules/linguistic-rules.mjs:87-103,280-310,530-565`
- Create: `tools/profile_engine.mjs`
- Test: `test/test_context_reranker.mjs`

**Problem:** each candidate may probe up to three `12 x 12` surface combinations, mostly misses. Instrumentation observed 264,345 scoring calls and 12.5 million bigram probability calls.

**Step 1: Add a reproducible profiler**

Report:

- startup/load time;
- RSS/heap after GC;
- calls and time for `resolveInContext`, `pTri`, `_pBiRaw`, and `scoreCandidateOverSurfaces`;
- bigram/trigram hit rates;
- latency by text-length bucket and benchmark category.

The profiler must not mutate benchmark results or production artifacts.

**Step 2: Write behavior-preservation tests**

Create tiny LMs testing:

- beam-first surface wins when attested;
- fallback surfaces are used only when the beam-first surface has no evidence;
- candidate score is identical before/after optimization for the fixture;
- no candidate receives NaN/Infinity.

**Step 3: Implement beam-first short-circuiting**

- Try beam-selected neighbor surfaces first.
- If an attested trigram exists, stop expanding that context.
- Expand sibling surfaces only when direct evidence is absent.
- Avoid building string keys repeatedly inside the same scoring call.

**Step 4: Add request-local numeric/count caches only where measured**

Do not add an unbounded global cache. A preliminary string-key cache improved only about 6%, so keep caching subordinate to reducing the number of probes.

**Step 5: Verify performance**

```powershell
node --test test/test_context_reranker.mjs
node tools/profile_engine.mjs
npm.cmd test
```

Expected intermediate gate:

- at least 3x fewer `scoreCandidateOverSurfaces` calls or at least 3x fewer trigram probes;
- `<=160` character p95 <=15 ms;
- identical deterministic-rule results;
- no dev accuracy regression.

**Step 6: Commit/checkpoint**

```bash
git add src/language.mjs src/rules/linguistic-rules.mjs tools/profile_engine.mjs test/test_context_reranker.mjs
git commit -m "perf: short-circuit sparse context scoring"
```

### Task 7: Make the SymSpell delete index length-aware

**Files:**

- Modify: `src/language.mjs:415-480`
- Test: `test/test_symspell_index.mjs`

**Problem:** depth-2 deletes are indexed for every word even though lookup policy uses edit distance 1 for words of length <=7.

**Step 1: Write equivalence tests**

For a fixture lexicon, compare old and new candidate results for all fixture inputs under the existing `maxEditDistanceFor()` policy.

**Step 2: Implement dynamic index depth**

```js
const depth = maxEditDistanceFor(key.length);
for (const variant of generateDeletes(key, depth)) {
  // index variant
}
```

Keep identity keys and deterministic bucket ordering.

**Step 3: Add index diagnostics**

Expose read-only counts in profiler output:

- key count;
- variant count;
- edge count;
- maximum bucket size;
- build time and RSS delta.

**Step 4: Verify**

```powershell
node --test test/test_symspell_index.mjs
node tools/profile_engine.mjs
npm.cmd test
```

Expected: roughly 30% fewer delete edges without candidate-recall loss.

**Step 5: Commit/checkpoint**

```bash
git add src/language.mjs test/test_symspell_index.mjs tools/profile_engine.mjs
git commit -m "perf: use length-aware SymSpell delete depth"
```

---

## Phase 3: Dev-only evaluation and statistical calibration

### Task 8: Create a strict train/dev/test evaluation workflow

**Files:**

- Create: `tools/generate_vsec_benchmark.mjs` or extend `tools/spelling_benchmark_adapter_vsec.mjs`
- Create: `tools/run_spelling_eval.mjs`
- Create: `config/spelling-tuning.json`
- Test: `test/test_eval_split_guard.mjs`
- Modify: `package.json`

**Step 1: Add commands**

Recommended scripts:

```json
{
  "scripts": {
    "eval:spelling:dev": "node tools/run_spelling_eval.mjs --split dev",
    "eval:spelling:test": "node tools/run_spelling_eval.mjs --split test --final",
    "profile:engine": "node tools/profile_engine.mjs"
  }
}
```

**Step 2: Enforce split guards**

- Default evaluation split is dev.
- Test/external evaluation requires `--final`.
- Tuning scripts must refuse `test` and `external-test`.
- Reports include code/config/artifact hashes.

**Step 3: Report diagnostic stages**

For every labeled correction report:

- correct target absent from generated candidates;
- target removed by cheap ranker;
- target removed by trigram ranker;
- target present but confidence/gate rejected;
- wrong suggestion emitted;
- unsupported multi-token split/merge.

**Step 4: Verify guards**

```powershell
node --test test/test_eval_split_guard.mjs
npm.cmd run eval:spelling:dev
```

Expected: attempting test without `--final` fails clearly.

**Step 5: Commit/checkpoint**

```bash
git add tools/run_spelling_eval.mjs tools/spelling_benchmark_adapter_vsec.mjs config/spelling-tuning.json test/test_eval_split_guard.mjs package.json
git commit -m "test: add split-safe spelling evaluation workflow"
```

### Task 9: Tune existing weights and gates on dev only

**Files:**

- Modify: `src/config.mjs`
- Modify: `config/spelling-tuning.json`
- Create: `tools/tune_spelling.mjs`
- Test: `test/test_tuning_determinism.mjs`

**Parameters to tune:**

- `SCORE_WEIGHTS` currently hard-coded as POC values;
- original prior bonus;
- beam width;
- typo alpha/beta/gamma;
- confidence and margin gates;
- cheap-ranker top-K;
- expensive-reranker top-K;
- smoothing/backoff masses.

**Step 1: Move tunable constants into config**

Keep defaults behavior-compatible before tuning.

**Step 2: Implement deterministic search**

- Use a bounded grid or seeded Bayesian/random search.
- Optimize a declared objective such as spelling-dev F0.5 or constrained recall with a minimum precision.
- Record every trial and config hash.
- Never open test/external labels.

**Step 3: Add constraints**

Reject a candidate configuration when:

- any core/rule regression test fails;
- missing-diacritic dev recall drops more than 0.5 percentage points;
- clean/dev false-positive rate exceeds the agreed bound;
- `<=160` character p95 exceeds 15 ms.

**Step 4: Freeze winning config**

Write a versioned config with provenance and dev metrics. Do not hand-edit thresholds after seeing test results.

**Step 5: Commit/checkpoint**

```bash
git add src/config.mjs config/spelling-tuning.json tools/tune_spelling.mjs test/test_tuning_determinism.mjs
git commit -m "perf: calibrate spelling ranker on VSEC dev"
```

### Task 10: Evaluate Modified Kneser-Ney/backoff as an isolated experiment

**Files:**

- Create: `src/language-kneser-ney.mjs`
- Create: `test/test_kneser_ney.mjs`
- Modify only after acceptance: `src/language.mjs`

**Rules:**

- Do not combine this experiment with candidate-generation changes.
- Use the same frozen candidate lists and dev rows.
- Compare against the corrected interpolation backend.
- Measure accuracy, p95, load time, and memory.

**Acceptance:** adopt only if it improves the chosen dev objective meaningfully without breaking latency/memory gates. Otherwise keep the simpler interpolated trigram.

---

## Phase 4: Compact deployable LM backend

### Task 11: Define a versioned binary LM format

**Files:**

- Create: `docs/lm-binary-format.md`
- Create: `tools/export_lm_binary.py`
- Test: `test/test_lm_binary_format.py`

**Format requirements:**

- magic bytes and version;
- endianness;
- flags and token total;
- vocabulary string table with offsets;
- unigram word IDs and counts;
- bigram `(wordId1, wordId2, count)` records;
- trigram `(wordId1, wordId2, wordId3, count)` records;
- section lengths and checksums;
- manifest SHA-256;
- deterministic byte output.

**Step 1: Write round-trip tests**

Use a tiny TSV fixture and assert exact equivalence of all U/B/T counts after binary encode/decode.

**Step 2: Implement deterministic exporter**

Do not rebuild raw corpora. Convert the active audited TSV into binary and record the source TSV hash.

**Step 3: Validate corruption handling**

Tests must reject wrong magic, unsupported version, truncated sections, invalid word IDs, and checksum mismatch.

**Step 4: Commit/checkpoint**

```bash
git add docs/lm-binary-format.md tools/export_lm_binary.py test/test_lm_binary_format.py
git commit -m "feat: define deterministic binary n-gram format"
```

### Task 12: Add a backend interface and binary reader

**Files:**

- Create: `src/lm/backend.mjs`
- Create: `src/lm/tsv-backend.mjs`
- Create: `src/lm/binary-backend.mjs`
- Modify: `src/language.mjs`
- Test: `test/test_lm_backend_equivalence.mjs`

**Interface:**

```js
class LanguageModelBackend {
  unigramCount(wordIdOrWord) {}
  bigramCount(leftIdOrWord, rightIdOrWord) {}
  trigramCount(left2IdOrWord, left1IdOrWord, wordIdOrWord) {}
  knows(word) {}
  wordId(word) {}
}
```

**Implementation requirements:**

- No concatenated string allocation per lookup.
- Use word IDs and typed arrays.
- Use a bounded-memory lookup structure: sorted typed arrays with measured binary search, open-addressed typed-array hash, or a compact trie.
- Do not use millions of `BigInt` or object entries as a disguised replacement for string `Map`.
- Loader must validate the complete artifact before making it active.
- TSV backend remains available only for compatibility/debugging until equivalence is proven.

**Equivalence tests:**

- all tiny-fixture counts match;
- a deterministic sample of at least 100,000 active-artifact U/B/T keys matches TSV;
- a deterministic sample of missing keys returns zero;
- all engine rule tests produce identical results before performance tuning continues.

**Performance acceptance:**

- LM steady RSS <=250 MB;
- full engine RSS <=400 MB;
- startup <=2 seconds;
- artifact <=100 MB;
- no p95 regression.

**Commit/checkpoint:**

```bash
git add src/lm src/language.mjs test/test_lm_backend_equivalence.mjs
git commit -m "perf: load n-grams through compact word-id backend"
```

### Task 13: Remove transient full-file parsing and silent compatibility paths

**Files:**

- Modify: `src/lm/tsv-backend.mjs`
- Modify: `src/lm/binary-backend.mjs`
- Modify: `src/language.mjs`
- Test: `test/test_language_model_loader.mjs`

**Steps:**

1. Remove `readFileSync(...).split('\n')` from the production path.
2. Load binary sections without generating millions of line strings.
3. Keep explicit diagnostics for backend, version, counts, hash, load time, and memory.
4. Remove raw-corpus fallback from production mode; retain it only behind an explicit development option.
5. Run artifact corruption and rollback tests.

---

## Phase 5: Final validation and optional attention reranker

### Task 14: Freeze configuration and run final tests once

**Files:**

- Create: `dataset_artifacts/evaluation/final-spelling-report.json`
- Create: `dataset_artifacts/evaluation/final-performance-report.json`
- Modify: `README.md`

**Step 1: Run all non-test gates first**

```powershell
npm.cmd test
npm.cmd run data:audit
python -m unittest discover -s test -p 'test_*.py' -v
python tools/audit_lm.py
npm.cmd run eval:spelling:dev
npm.cmd run profile:engine
```

All must pass before opening final test/external reports.

**Step 2: Freeze hashes**

Record:

- source code/config hashes;
- LM and lexicon hashes;
- split manifests;
- command line and environment;
- Node/Python versions.

**Step 3: Run test/external evaluation exactly once**

```powershell
npm.cmd run eval:spelling:test -- --final
node benchmark/run-benchmark.mjs
```

Do not retune based on these results. Any later change requires a new version/milestone and a fresh untouched external set.

**Step 4: Document final metrics and deploy envelope**

Include accuracy per category, candidate-stage failure reasons, length-bucket latency, startup, RSS/heap, artifact sizes, and rollback instructions.

### Task 15: Optional lightweight attention reranker

Start this task only if, after Tasks 1-14:

- candidate oracle coverage >=92%;
- runtime/memory gates pass;
- VSEC dev ranking/gating remains the dominant error source;
- the classical trigram ranker misses the agreed dev target.

**Architecture:** attention is a reranker over the top 4-8 candidates, not a full replacement for deterministic rules or candidate generation.

**Data:** train on VSEC train and clean synthetic train only; select on VSEC dev; never train/tune on VSEC test or Viwiki external.

**Inputs:** local character/subword window, candidate word, edit features, unigram/bigram/trigram features, and protected-token flags.

**Constraints:**

- quantized artifact <=10 MB;
- reranker p95 contribution <=3 ms;
- CPU inference only;
- deterministic fallback to trigram ranker;
- must improve dev objective beyond confidence interval/noise;
- must not weaken deterministic rule guarantees.

If these constraints are not met, do not ship attention.

---

## Required execution report from the implementing agent

For every task, report:

1. Files changed.
2. Failing test added and failure observed.
3. Minimal implementation made.
4. Focused tests and full regression results.
5. Dev-only accuracy delta.
6. Startup/RSS/p50/p95 delta when performance-sensitive.
7. Artifact and config hashes.
8. Any deviation from this plan and why.

Final handoff must explicitly state:

- whether Viwiki now measures correct suggestion quality;
- candidate oracle recall before/after;
- ranking/gating recall before/after;
- missing-diacritic, VSEC dev/test, and Viwiki external metrics separately;
- engine RSS/startup and SMS-length p95;
- whether attention was needed;
- exact rollback artifact/path.

