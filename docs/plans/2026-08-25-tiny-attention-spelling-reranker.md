# Tiny Attention Spelling Reranker Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Also use `superpowers:test-driven-development` for every behavior change. Do not batch tasks.

**Goal:** Build and pretrain a tiny Vietnamese attention encoder, fine-tune it as a listwise spelling reranker over `KEEP_ORIGINAL + <=4` frozen candidates, and permit `EXPERIMENTAL_ACTIVE` only when it beats the agreed classical baseline without violating precision, clean-data, artifact-size, or latency gates.

**Architecture:** Production remains rule-first. Existing guards and candidate generation run before a one-block, two-head Transformer sees the SMS context (maximum 32 document tokens), the original token, at most four candidates, and the existing classical features. Training uses PyTorch, but production inference is pure JavaScript over typed arrays and quantized weights; the model can select only `KEEP_ORIGINAL` or one supplied candidate and can never generate text.

**Tech Stack:** Node.js ESM, JavaScript typed arrays, Python 3, PyTorch (training only), NumPy (artifact export/tests), JSON metadata + int8 binary weights, Node/Python test runners.

---

## 0. Non-negotiable contract

The implementing agent must obey every item below.

1. Use Semble before locating code. Use GitNexus/CodeGraph impact analysis if an index exists.
2. Do not initialize Git. This workspace currently has no standalone Git history for `sms-validation-demo`. Use the execution log and SHA-256 manifests as checkpoints. If the user later provides a real repository, make one atomic commit per task.
3. Never read or run VSEC test, Viwiki external-test, benchmark external categories, or the old held-out artifact. This milestone has no untouched official held-out set.
4. Allowed supervised data: `dataset_artifacts/vsec/vsec-train.jsonl` and `dataset_artifacts/clean-source/clean-train.txt` only.
5. Dev may be opened once only after model choices and thresholds are frozen on the internal calibration split. Dev is for the experimental gate, never for retuning.
6. Attention must never create candidates, alter offsets, bypass protected ranges, bypass whitelist/abbreviation/brand rules, or change deterministic rules.
7. Candidate count is hard-capped at four. Context is hard-capped at 32 document tokens.
8. Production must not import PyTorch, NumPy, ONNX Runtime, native addons, or any new npm package.
9. Modes are exactly `OFF`, `SHADOW`, and `EXPERIMENTAL_ACTIVE`. Do not name this production `ACTIVE` because no fresh external set exists.
10. Missing artifact means disabled fallback. Present-but-invalid artifact fails closed to classical behavior and records a diagnostic; it must never crash validation for normal user input.
11. Do not change the LM corpus, n-gram counts, SymSpell limits, current rule thresholds, or raw datasets in this milestone.
12. Do not activate attention merely because unit tests pass. All acceptance gates in section 14 must pass.

### Agreed comparison baseline

Use the safer classical real-word configuration, not the aggressive current `originalWindows<=3` state:

```json
{
  "realWordTypoMode": "ACTIVE",
  "realWordTypoMinProbability": 0.95,
  "realWordTypoMaxOriginalWindows": 1,
  "wrongDiacriticMode": "OFF",
  "wordBoundaryCorrectionMode": "SHADOW"
}
```

Known dev reference from the fresh 2026-08-25 calibration:

| Metric | Classical reference |
|---|---:|
| semantic precision | 0.6829 |
| semantic recall | 0.2395 |
| semantic F0.5 | 0.4983 |
| incremental lane precision | approximately 0.806 |
| SMS <=160 p95 | remeasure on the same machine before attention |

Do not hard-code the latency reference. Task 0 writes a fresh baseline artifact.

### Model contract

```text
max context tokens       32
max non-original options 4
output options           KEEP_ORIGINAL + candidate[0..3]
hidden size              48
attention blocks         1
attention heads          2
head size                24
FFN size                 96
word vocabulary          8192 including special tokens
char hash buckets        4096
char n-grams             Unicode character 2..4-grams
position embeddings      32
production weights       symmetric int8 + float32 scales/biases
artifact budget          <=10 MiB including metadata and binary weights
```

Token representation:

```text
x_i = wordEmbedding[wordId]
    + mean(charEmbedding[hash(charNgram)])
    + positionEmbedding[position]
    + markerEmbedding[marker]
```

Encoder:

```text
h1 = LayerNorm(x + MultiHeadSelfAttention(x, mask))
h2 = LayerNorm(h1 + FFN_GELU(h1))
context = h2[targetPosition]
```

For each option (`KEEP_ORIGINAL` uses the original token embedding):

```text
z = concat(context, optionEmbedding, context * optionEmbedding,
           projectedClassicalFeatures, optionTypeEmbedding)
score = Linear(GELU(Linear(z)))
probabilities = maskedSoftmax(scores)
```

The JavaScript and PyTorch implementations must share the formulas above exactly.

## 1. Task dependency order

```text
Task 0 baseline
  -> Task 1 evaluation/provenance correctness
  -> Task 2 tokenizer/context contract
  -> Task 3 attention data split + ranking rows
  -> Task 4 vocabulary/pretraining corpus
  -> Task 5 PyTorch model + pretraining
  -> Task 6 listwise fine-tuning
  -> Task 7 quantized export
  -> Task 8 pure-JS numerical parity
  -> Task 9 engine SHADOW integration
  -> Task 10 calibration and full-emission evaluation
  -> Task 11 EXPERIMENTAL_ACTIVE decision
  -> Task 12 documentation and handoff
```

Do not start a task until the previous task's tests, artifacts, hashes, and execution-log entry are complete.

---

### Task 0: Freeze a trustworthy classical baseline

**Files:**

- Create: `tools/run_attention_baseline.mjs`
- Create: `test/test_attention_baseline_scope.mjs`
- Create: `dataset_artifacts/evaluation/attention-classical-baseline.json` (generated)
- Create: `dataset_artifacts/evaluation/attention-classical-baseline.json.manifest.json` (generated)
- Create: `docs/plans/execution-log-attention-reranker.md`

**Step 1: Write the failing scope test**

Test that the baseline runner:

- accepts only `--split dev`;
- rejects `test`, `external-test`, `viwiki`, and `benchmark` path markers before reading;
- forces the exact classical overrides from the agreed baseline;
- reports source/config/model hashes and all semantic metrics;
- reports full engine latency and an attention contribution of zero.

Do not mock file access. Import pure helpers from the runner and test them directly.

**Step 2: Run RED**

```powershell
node --test test/test_attention_baseline_scope.mjs
```

Expected: FAIL because `tools/run_attention_baseline.mjs` does not exist.

**Step 3: Implement the minimal runner**

Reuse `adaptVsecSplit`, `linguisticCorrectionMatches`, `ValidationContext`, and the existing profiler percentile definition. Do not duplicate correction matching logic.

Required artifact fields:

```json
{
  "schema": "attention-classical-baseline-v1",
  "split": "dev",
  "heldOutOpened": false,
  "overrides": {},
  "hashes": {},
  "metrics": {
    "strict": {},
    "semantic": {},
    "incrementalLane": {},
    "cleanUnexpectedIssues": 0,
    "mdRecall": 0,
    "sms160P95Ms": 0
  }
}
```

Incremental lane precision must be computed from the difference between all emitted semantic issues in baseline OFF and classical candidate mode across every token in every message. Never reuse replay precision attached only to labeled spans.

**Step 4: Run GREEN and generate the artifact**

```powershell
node --test test/test_attention_baseline_scope.mjs
node tools/run_attention_baseline.mjs --split dev `
  --out dataset_artifacts/evaluation/attention-classical-baseline.json
```

Expected: test PASS; semantic P/R/F0.5 approximately `0.6829/0.2395/0.4983`. Stop if materially different and investigate before continuing.

**Step 5: Record checkpoint**

Append Task 0 status, commands, metrics, source hashes, artifact SHA-256, latency, RSS, and deviations to `docs/plans/execution-log-attention-reranker.md`.

---

### Task 1: Fix evaluation provenance and ACTIVE-stage attribution first

Attention must not be built on the known broken measurement contract.

**Files:**

- Modify: `tools/run_spelling_eval.mjs` (`stageFromDecision`, shadow dump writer)
- Modify: `tools/calibrate_recall_lanes.mjs` (provenance validation)
- Modify: `src/rules/linguistic-rules.mjs` (`evaluateRealWordToken` diagnostic return)
- Create: `src/evaluation-provenance.mjs`
- Create: `test/test_evaluation_provenance.mjs`
- Create: `test/test_active_stage_attribution.mjs`

**Step 1: Write RED provenance tests**

Required API:

```js
createEvaluationHeader({ schema, split, hashes, config, createdBy })
validateEvaluationHeader(header, expected)
```

Test rejection for a mismatch in each of:

- linguistic rules hash;
- config hash;
- classical reranker hash;
- LM hash;
- tokenizer/vocab/model hash when present;
- feature-contract version;
- split.

The first JSONL line must be a header object with `recordType: "header"`. Decision lines use `recordType: "decision"`.

**Step 2: Write RED stage tests**

Construct real ACTIVE-light-path decisions and assert terminal stages:

```text
target absent from wide pool       candidate-miss
target present, other option wins  wrong-suggestion
target wins but confidence fails   attention/classical-gate-rejected
target wins and emits              correct before downstream matching
```

Do not infer ACTIVE stages from empty `ranked` or `cheapKept` arrays. Add explicit diagnostic fields to the light-path return.

**Step 3: Run RED**

```powershell
node --test test/test_evaluation_provenance.mjs test/test_active_stage_attribution.mjs
```

Expected: FAIL for missing module/explicit stage contract.

**Step 4: Implement minimally**

Add a decision diagnostic shape shared by classical and attention paths:

```js
{
  generatedWords,
  consideredWords,
  selectedWord,
  emitted,
  rejectionReason,
  decisionOwner
}
```

Update `stageFromDecision` to prefer this explicit shape. Retain legacy fallback for unknown-token decisions until their migration is separately tested.

**Step 5: Run GREEN and regressions**

```powershell
node --test test/test_evaluation_provenance.mjs test/test_active_stage_attribution.mjs
node --test test/test_real_word_typo_lane.mjs test/test_recall_calibration.mjs
npm.cmd test
```

Expected: all PASS. Regenerate no official held-out artifacts.

**Step 6: Record checkpoint**

Append hashes and explain that future calibration refuses stale decision dumps.

---

### Task 2: Implement the shared tokenizer and 32-token context selector

**Files:**

- Create: `src/attention-tokenizer.mjs`
- Create: `tools/attention_tokenizer.py`
- Create: `test/fixtures/attention-tokenizer-cases.json`
- Create: `test/test_attention_tokenizer.mjs`
- Create: `test/test_attention_tokenizer.py`

**Step 1: Define special IDs**

Use this immutable order:

```text
0 PAD
1 UNK
2 BOS
3 EOS
4 TARGET
5 PROTECTED
6 PUNCT
7 MASK
8..8191 learned vocabulary
```

Both languages must NFC-normalize and lowercase with Unicode-aware behavior. Never strip Vietnamese accents.

**Step 2: Write RED fixture-driven parity tests**

Fixture cases must cover:

- short SMS uses all tokens;
- 40-token SMS retains target and at most 32 tokens;
- first four document tokens are retained when truncation occurs;
- nearest target context fills remaining capacity deterministically;
- punctuation markers;
- protected URL/phone/placeholder represented as `PROTECTED`, never decomposed;
- NFD/NFC equivalence;
- OOV word with deterministic character 2..4-gram hashes;
- candidate order independent tokenization.

Character hash is FNV-1a over UTF-8 bytes, unsigned 32-bit, modulo 4096. Do not use JS `String` code-unit hashing.

**Step 3: Run RED**

```powershell
node --test test/test_attention_tokenizer.mjs
python -m unittest test.test_attention_tokenizer -v
```

Expected: FAIL because implementations do not exist.

**Step 4: Implement JS first, then Python to the fixture**

Context selection algorithm:

1. Build document tokens from existing `ValidationDocument`; do not re-tokenize production text.
2. If count `<=32`, keep all in original order.
3. Otherwise retain the first four token indices, the target index, then add nearest indices by `(absoluteDistance, index)` until 32 slots are full.
4. Sort retained indices ascending.
5. Mark the selected target position explicitly; never replace the original surface with a candidate.

**Step 5: Run GREEN**

```powershell
node --test test/test_attention_tokenizer.mjs
python -m unittest test.test_attention_tokenizer -v
```

Expected: both PASS and produce byte-identical fixture vectors.

**Step 6: Record checkpoint**

Record tokenizer contract version as `attention-tokenizer-v1`.

---

### Task 3: Create leakage-safe attention ranking rows and internal groups

**Files:**

- Create: `tools/extract_attention_ranking_rows.mjs`
- Create: `src/attention-ranking-schema.mjs`
- Create: `test/test_attention_ranking_rows.mjs`
- Create: `.tmp/attention-ranking-{train,calibration,internal-test}.jsonl` (generated)
- Create: `.tmp/attention-ranking-split-manifest.json` (generated)

**Step 1: Write RED schema and leakage tests**

Every file begins with a provenance header. Every row contains:

```json
{
  "recordType": "ranking-row",
  "id": "stable-id",
  "groupId": "normalized-source-sentence-hash",
  "source": "vsec-train|clean-train",
  "context": {},
  "original": "",
  "candidates": [],
  "classicalFeatures": [],
  "labelIndex": 0,
  "lane": "ACCENTED_SAME_KEY|DIFFERENT_KEY_REAL_WORD|CLEAN_KEEP"
}
```

Rules:

- option index 0 is always `KEEP_ORIGINAL`;
- candidates are production shortlist order, maximum four, deduplicated case-insensitively;
- correct VSEC target is never injected into the shortlist;
- drop a correction row when the target is absent;
- clean rows always label index 0;
- no group may occur in more than one attention split;
- path guard rejects forbidden held-out markers before any read.

**Step 2: Define deterministic split**

Hash `groupId + "attention-v1-20260825"` with SHA-256:

```text
00..79 train
80..89 calibration
90..99 internal-test
```

This is an attention-milestone internal split of allowed TRAIN sources, not an official external test.

**Step 3: Run RED**

```powershell
node --test test/test_attention_ranking_rows.mjs
```

Expected: FAIL because extractor/schema do not exist.

**Step 4: Implement extraction through production helpers**

Reuse:

- `buildCorrectionCandidates`;
- `selectDiverseShortlist`;
- `buildRecallPairwiseFeatures`;
- `classifyCorrectionRelation`;
- `attention-tokenizer.mjs`.

Do not reimplement candidate generation in Python.

**Step 5: Run GREEN and extraction**

```powershell
node --test test/test_attention_ranking_rows.mjs
node tools/extract_attention_ranking_rows.mjs `
  --vsec-train dataset_artifacts/vsec/vsec-train.jsonl `
  --clean-train dataset_artifacts/clean-source/clean-train.txt `
  --out-dir .tmp
```

Expected: PASS; non-empty train/calibration/internal-test; zero group overlap; all option counts in `[2,5]`; manifest contains source hashes and counts per lane/label.

**Step 6: Write pretraining deny list**

The manifest must include normalized sentence hashes for attention calibration and internal-test groups. Task 4 uses these hashes to exclude those exact sentences from self-supervised pretraining.

**Step 7: Record checkpoint**

Record counts, dropped target-absent rows, KEEP/correction balance, and SHA-256 for all outputs.

---

### Task 4: Build a leakage-safe vocabulary and self-supervised pretraining set

**Files**

- Create: `tools/build_attention_pretrain_data.py`
- Create: `test/test_attention_pretrain_data.py`
- Generate, do not commit: `.tmp/attention-vocab.json`
- Generate, do not commit: `.tmp/attention-pretrain-*.npz`
- Generate, do not commit: `.tmp/attention-pretrain-manifest.json`

**Step 1: Write RED tests**

Cover all of these cases:

- only `src/data/corpus-train.txt` and `dataset_artifacts/clean-source/clean-train.txt` are accepted as text sources;
- a path containing `dev`, `test`, `heldout`, `calibration`, or an evaluation artifact is rejected;
- any sentence hash from Task 3's deny list is excluded;
- duplicate normalized sentences are emitted once;
- two runs with the same seed produce identical vocabulary, shards, ordering, and hashes;
- special IDs are fixed: `PAD=0`, `UNK=1`, `MASK=2`, `CLS=3`, `SEP=4`;
- vocabulary has at most 8,192 word entries, excluding hashed character buckets;
- sequences are at most 32 tokens and padding positions have mask zero.

**Step 2: Run RED**

```powershell
python -m unittest test.test_attention_pretrain_data -v
```

Expected: FAIL because the builder does not exist.

**Step 3: Implement the deterministic builder**

- Normalize Unicode to NFC; preserve Vietnamese diacritics and useful case information.
- Replace URLs, emails, phone-like strings, tracking IDs, and markup with stable placeholders.
- Do not silently repair spelling.
- Deduplicate by SHA-256 of normalized text and apply Task 3's deny list first.
- Count word tokens on allowed training text only; sort by descending frequency then Unicode order.
- Create deterministic sharded NPZ files instead of one giant in-memory array.
- Store IDs, masks, and sentence hashes; never store raw corpus lines in the deployable model.
- Expose `--seed`, `--max-rows`, `--shard-size`, and `--out-dir`; default seed is `20260825`.

Reuse Task 2's tokenizer contract. If Python needs a port, first export JavaScript golden fixtures and require exact token/ID parity. JavaScript remains the source of truth.

**Step 4: Run GREEN and build**

```powershell
python -m unittest test.test_attention_pretrain_data -v
python tools/build_attention_pretrain_data.py `
  --corpus src/data/corpus-train.txt `
  --clean dataset_artifacts/clean-source/clean-train.txt `
  --deny-list .tmp/attention-ranking-manifest.json `
  --out-dir .tmp --seed 20260825
```

Expected: PASS; zero denied hashes, vocabulary at most 8,192, and SHA-256 for every shard.

**Step 5: Record checkpoint**

Record source hashes, lines seen/accepted/deduplicated/rejected, vocabulary size, shard count, and peak RSS. Do not copy the raw corpus into `src/data`.

---

### Task 5: Implement and self-pretrain the tiny encoder in PyTorch

**Files**

- Create: `tools/attention_model.py`
- Create: `tools/pretrain_attention_encoder.py`
- Create: `test/test_attention_model.py`
- Create: `test/test_attention_pretraining.py`
- Generate, do not commit: `.tmp/attention-pretrained.pt`
- Generate, do not commit: `.tmp/attention-pretrain-report.json`

**Step 1: Write RED architecture tests**

Freeze and assert: one encoder block; hidden size 48; two 24-dimensional heads; FFN size 96; maximum 32 tokens; 32 learned position embeddings; word plus hashed-character embeddings; pre-layer normalization; residuals after attention and FFN; masked stable softmax; no decoder or generation path.

Also test shapes, padding masks, no NaN/Inf, a parameter-count ceiling, and exact repeatability for the same seed.

**Step 2: Write RED pretraining tests**

Use masked/replaced-token prediction:

- select 15% of non-special, non-padding tokens;
- replace selected tokens 80% with `MASK`, 10% with a random training-vocabulary token, and leave 10% unchanged;
- compute loss only at selected positions;
- never target special or padding positions;
- require a tiny fixture's loss to fall materially within 50 optimizer steps.

**Step 3: Run RED**

```powershell
python -m unittest test.test_attention_model test.test_attention_pretraining -v
```

Expected: FAIL because model/trainer do not exist.

**Step 4: Implement model and trainer**

Use PyTorch only for offline training. Set Python, NumPy, and Torch seeds, deterministic algorithms, and explicit CPU thread count. Reject CUDA-only paths.

Starting configuration: AdamW `lr=3e-4`, weight decay `0.01`, batch size `128` or largest safe size, gradient clipping `1.0`, at most 5 epochs, and early stop after two validation checks without improvement. Make a deterministic 5% pretraining-validation partition using allowed training text only.

The checkpoint contains state, architecture, vocabulary hash, tokenizer version, seed, and source-manifest hash. It contains no raw text.

**Step 5: Run GREEN and pretrain**

```powershell
python -m unittest test.test_attention_model test.test_attention_pretraining -v
python tools/pretrain_attention_encoder.py `
  --data-manifest .tmp/attention-pretrain-manifest.json `
  --vocab .tmp/attention-vocab.json `
  --output .tmp/attention-pretrained.pt `
  --report .tmp/attention-pretrain-report.json `
  --seed 20260825
```

Expected: PASS; validation loss improves over an untrained control; no dev/test/held-out path appears in the report.

**Step 6: Stop condition**

If the tiny fixture cannot learn or real pretraining does not beat the untrained control, stop. Fix masks, tokenizer parity, data, or objective; do not increase model size.

---

### Task 6: Fine-tune a listwise spelling reranker

**Files**

- Create: `tools/train_attention_reranker.py`
- Create: `test/test_attention_finetuning.py`
- Generate, do not commit: `.tmp/attention-finetuned.pt`
- Generate, do not commit: `.tmp/attention-finetune-report.json`

**Step 1: Write RED ranking tests**

Each row has `2..5` choices in this exact order:

```text
[KEEP_ORIGINAL, candidate_1, candidate_2, candidate_3, candidate_4]
```

Assert that gold is unmasked; KEEP is gold for clean rows; correction is gold only when production candidate generation found it; padding gets zero probability; candidate permutations preserve remapped scores; a tiny mixed fixture can overfit; no missing target is injected; and train/calibration/internal-test group hashes are disjoint.

**Step 2: Run RED**

```powershell
python -m unittest test.test_attention_finetuning -v
```

Expected: FAIL because the fine-tuner does not exist.

**Step 3: Implement listwise scoring**

For each choice encode the same 32-token message context plus explicit target and candidate markers. Score all choices with one shared head, then masked-softmax over KEEP and candidates.

```text
L = cross_entropy(choice_logits, gold_index)
  + 0.1 * max(0, 0.2 - gold_logit + highest_wrong_logit)
```

Start with AdamW `lr=1e-4`, batch 64, at most 10 epochs, clipping `1.0`, and early stopping by calibration F0.5 subject to semantic precision at least 68%. Stratify batches across clean KEEP, accented, unaccented, and different-key rows. Cap any lane weight at 3x.

Tune only on Task 3 calibration. Freeze architecture, seed, and thresholds before running internal-test once. Do not open VSEC dev yet.

**Step 4: Run GREEN and train**

```powershell
python -m unittest test.test_attention_finetuning -v
python tools/train_attention_reranker.py `
  --rows .tmp/attention-ranking-rows.jsonl `
  --split-manifest .tmp/attention-ranking-manifest.json `
  --pretrained .tmp/attention-pretrained.pt `
  --output .tmp/attention-finetuned.pt `
  --report .tmp/attention-finetune-report.json `
  --seed 20260825
```

Expected: PASS; report includes precision, recall, F0.5, KEEP accuracy, candidate-choice accuracy, and metrics by lane.

**Step 5: Stop condition**

Stop before export if the model cannot beat the classical row-level ranker on calibration and internal-test, or if gain comes mainly from new clean false positives.

---

### Task 7: Export a deterministic int8 artifact below 10 MiB

**Files**

- Create: `tools/export_attention_model.py`
- Create: `test/test_attention_export.py`
- Create after a successful export: `src/data/attention-reranker.int8.bin`
- Create after a successful export: `src/data/attention-reranker.json`

**Step 1: Write RED format tests**

Freeze binary format v1. It is little-endian and begins with:

```text
magic[8] = TDRANK01
format_version: uint32
tensor_count: uint32
table_offset: uint32
data_offset: uint32
```

Each tensor entry contains stable numeric ID, dtype, rank, up to four dimensions, byte offset/length, quantization mode, and scale-table offset. Require bounds checks and 16-byte alignment.

Assert two exports are byte-identical; corrupt magic/version/length/offset/shape/checksum is rejected; weights and embeddings are int8; linear matrices use symmetric per-output-channel scales; biases, layer norms, and scales are float32; total BIN+JSON is at most `10 * 1024 * 1024`; metadata contains architecture and all input hashes.

**Step 2: Run RED**

```powershell
python -m unittest test.test_attention_export -v
```

Expected: FAIL because exporter/artifacts do not exist.

**Step 3: Implement and export**

Quantize the frozen checkpoint. Do not use ONNX, TensorFlow.js, native addons, `onnxruntime-node`, or runtime downloads. Keep large arrays in BIN, not JSON.

```powershell
python tools/export_attention_model.py `
  --checkpoint .tmp/attention-finetuned.pt `
  --vocab .tmp/attention-vocab.json `
  --output-bin src/data/attention-reranker.int8.bin `
  --output-meta src/data/attention-reranker.json
python -m unittest test.test_attention_export -v
```

Expected: PASS; deterministic artifact is no larger than 10 MiB.

**Step 4: Stop condition**

If quantization drops row-level F0.5 by more than 0.5 percentage points or precision below 68%, fix quantization/export math. Do not ship a float checkpoint.

---

### Task 8: Implement pure-JavaScript typed-array inference with Python parity

**Files**

- Create: `src/attention-reranker.mjs`
- Create: `tools/export_attention_parity_fixture.py`
- Create: `test/fixtures/attention-parity.json`
- Create: `test/test_attention_reranker.mjs`

**Step 1: Export a frozen parity fixture**

Include at least 20 synthetic/public cases covering NFC/NFD Vietnamese, OOV, padding, one/four candidates, all spelling lanes, KEEP, short text, and text longer than 32 tokens. Store IDs, masks, option map, and Python logits/probabilities; do not copy private corpus text.

**Step 2: Write RED JavaScript tests**

Assert loader validation, one load per engine instance, built-in JavaScript plus typed arrays only, logit max error `<=1e-3`, probability max error `<=2e-3`, candidate permutation invariance, padding never wins, no NaN/Inf, deterministic outputs, and malformed/missing artifacts return a classical-fallback status instead of throwing through `validate()`.

**Step 3: Run RED**

```powershell
python tools/export_attention_parity_fixture.py `
  --checkpoint .tmp/attention-finetuned.pt `
  --artifact src/data/attention-reranker.int8.bin `
  --output test/fixtures/attention-parity.json
node --test test/test_attention_reranker.mjs
```

Expected: JavaScript test FAIL because inference does not exist.

**Step 4: Implement the minimal runtime**

Implement only: int8 embedding lookup/scales, float32 residuals, layer norm with Python's epsilon, Q/K/V projections, two-head scaled attention, stable masked softmax, identical GELU approximation, two FFN projections, shared choice head, and final softmax. Parse once into typed-array views and reuse safe scratch buffers.

Do not add a generic tensor framework, dynamic graph, WebAssembly compiler, worker thread, native dependency, or network fetch.

**Step 5: Run GREEN**

```powershell
node --test test/test_attention_reranker.mjs
```

Expected: PASS for parity and corrupt-artifact cases.

**Step 6: Record runtime checkpoint**

Record cold load time, bytes, RSS delta, median and p95 batch-one inference, Node version, CPU, and run count. Task 11 applies the full-engine gate.

---

### Task 9: Integrate attention in SHADOW without weakening rules

**Files**

- Modify: `src/config.mjs`
- Modify: `src/engine.mjs`
- Modify: `src/rules/linguistic-rules.mjs`
- Create: `test/test_attention_integration.mjs`

**Step 1: Write RED integration tests**

Cover these invariants:

- `OFF` is behaviorally identical to the current classical engine;
- `SHADOW` records a decision but never changes emitted issues;
- attention runs only after whitelist/dictionary/abbreviation/brand exceptions and protected-pattern guards;
- it only ranks the frozen production shortlist and can choose KEEP; it never generates a candidate;
- no candidate outside the shortlist can appear in output;
- missing/corrupt/incompatible model falls back to the classical p=0.95/o=1 decision;
- one token produces at most one spelling issue;
- URLs, emails, phone numbers, prices, dates, codes, brandnames, whitelist, dictionary, abbreviations, and explicit exceptions remain protected;
- same-key accented candidates can reach SHADOW scoring even though their current emission lane is disabled;
- repeated calls are deterministic and do not reload the artifact.

**Step 2: Run RED**

```powershell
node --test test/test_attention_integration.mjs
```

Expected: FAIL because engine integration does not exist.

**Step 3: Add explicit modes and config validation**

```js
attentionRerankerMode: 'SHADOW', // OFF | SHADOW | EXPERIMENTAL_ACTIVE
attentionMaxTokens: 32,
attentionMaxCandidates: 4,
attentionMinProbability: null,
attentionMinMargin: null,
```

Invalid values fail at startup. Thresholds stay null until Task 10 freezes them. The safe fallback is classical p=0.95/o=1.

**Step 4: Integrate at one ownership point**

Flow must be:

```text
token + protected-rule result
  -> production candidate generation
  -> production shortlist (max 4)
  -> classical decision
  -> optional attention scoring
  -> mode switch
       OFF: classical
       SHADOW: classical + diagnostics
       EXPERIMENTAL_ACTIVE: gated attention or classical fallback
  -> issue deduplication/emission
```

Expose diagnostic fields only in evaluation/debug output: model/version hash, context indices, shortlist, KEEP/candidate probabilities, margin, chosen option, mode, fallback reason, and elapsed microseconds. Never include full corpus content.

**Step 5: Run GREEN and regression tests**

```powershell
node --test test/test_attention_integration.mjs
node --test test/test_core_engine.mjs test/test_linguistic_rules.mjs test/test_recall_reranker.mjs
```

Expected: PASS; OFF and SHADOW user-visible output hashes equal the classical baseline.

---

### Task 10: Calibrate attention with full-emission accounting

**Files**

- Create: `tools/run_attention_eval.mjs`
- Create: `tools/calibrate_attention_reranker.mjs`
- Create: `test/test_attention_evaluation.mjs`
- Generate: `dataset_artifacts/evaluation/attention-shadow-dev.json`
- Generate: `dataset_artifacts/evaluation/attention-calibration.json`

**Step 1: Write RED evaluator tests**

Use tiny fixtures proving:

- every emitted issue at every token is counted, not only the labeled span;
- an extra issue elsewhere in a positive message is a false positive;
- new issues are computed by stable issue identity relative to classical output;
- attention stage attribution uses real returned fields, not empty `ranked`/`cheapKept` arrays;
- candidate miss, context-selector miss, KEEP choice, wrong-candidate choice, threshold decline, guard block, and fallback are distinct stages;
- provenance contains dataset hash, model hash, config, git/worktree identifier when available, timestamp, Node version, CPU, warmup, run count, and metric definitions;
- evaluation refuses the old v2 held-out path.

**Step 2: Run RED**

```powershell
node --test test/test_attention_evaluation.mjs
```

Expected: FAIL, including regression cases for the old span-only lane-precision bug.

**Step 3: Implement shadow evaluator**

Compare four systems on exactly the same examples:

1. classical OFF baseline;
2. attention SHADOW raw argmax;
3. attention thresholded decision;
4. final user-visible result under the selected mode.

Report semantic precision/recall/F0.5, spelling-only metrics, clean false positives, red-team regressions, metrics by lane/relation, candidate oracle, KEEP accuracy, attention incremental TP/FP, stage-loss counts, latency median/p95/p99, cold-start, RSS, and artifact bytes.

**Step 4: Implement deterministic calibration**

Grid-search only Task 3 calibration rows over:

- minimum selected-choice probability;
- minimum margin over KEEP;
- minimum margin over second-best candidate;
- allowed relation lanes.

Optimize F0.5 subject to semantic precision `>=0.68` and no clean regression. Break ties by higher precision, lower p95, fewer enabled lanes, then lexicographic config JSON. Freeze the winning JSON before VSEC dev is opened.

**Step 5: Run GREEN, calibrate, then one dev evaluation**

```powershell
node --test test/test_attention_evaluation.mjs
node tools/calibrate_attention_reranker.mjs `
  --rows .tmp/attention-ranking-rows.jsonl `
  --split calibration `
  --output dataset_artifacts/evaluation/attention-calibration.json
node tools/run_attention_eval.mjs `
  --config dataset_artifacts/evaluation/attention-calibration.json `
  --split dev `
  --output dataset_artifacts/evaluation/attention-shadow-dev.json
```

Expected: tests PASS; calibration and dev files contain complete provenance. Do not retune after seeing dev. A failure on dev means SHADOW, not another dev-driven search.

---

### Task 11: Gate EXPERIMENTAL_ACTIVE mechanically

**Files**

- Create: `config/attention-tuning.json`
- Create: `tools/select_attention_mode.mjs`
- Create: `test/test_attention_activation_gate.mjs`
- Modify only on PASS: `src/config.mjs`

**Step 1: Write RED gate tests**

Create one fixture per failing gate and prove the selector returns `SHADOW`. Only a fixture satisfying every gate returns `EXPERIMENTAL_ACTIVE`. Missing metrics, NaN, stale hashes, or provenance mismatch are failures.

**Step 2: Run RED**

```powershell
node --test test/test_attention_activation_gate.mjs
```

Expected: FAIL because the selector does not exist.

**Step 3: Implement the pure selector**

All gates are mandatory on frozen VSEC dev versus classical p=0.95/o=1:

| Gate | Required value |
|---|---:|
| Semantic precision | `>= 0.6800` |
| Semantic recall | `> 0.2400` |
| Semantic F0.5 | `> 0.4983` |
| Attention's actual incremental precision | `>= 0.7000` |
| New clean false positives | `0` |
| New protected/red-team regressions | `0` |
| Multidimensional-test drop | `<= 0.5` percentage point |
| Attention-only p95 delta | `<= 3 ms` |
| Full-engine p95 | `<= 15 ms` |
| Cold-start delta | `<= 250 ms` |
| Model RSS delta | `<= 20 MiB` |
| BIN + JSON artifact | `<= 10 MiB` |
| Python/JS parity, core, rules, integration | all PASS |
| Output determinism | identical hashes across repeated runs |

Also require internal-test not to be materially worse than calibration: precision drop at most 2 points and F0.5 drop at most 2 points. This catches overfit before dev is trusted.

**Step 4: Generate frozen config and verdict**

```powershell
node tools/select_attention_mode.mjs `
  --baseline dataset_artifacts/evaluation/attention-classical-baseline.json `
  --calibration dataset_artifacts/evaluation/attention-calibration.json `
  --dev dataset_artifacts/evaluation/attention-shadow-dev.json `
  --output config/attention-tuning.json
node --test test/test_attention_activation_gate.mjs
```

Expected: selector prints every gate as PASS/FAIL and writes the verdict with input hashes.

**Step 5: Activate only if verdict is PASS**

- PASS: set default to `EXPERIMENTAL_ACTIVE` and load exactly the frozen thresholds/model hash.
- FAIL: keep `SHADOW`; do not relax a threshold, remove a lane, or edit evaluation data merely to turn it green.
- At runtime, any hash/config/model/load error automatically uses classical p=0.95/o=1 for that process.

This is experimental activation only. Production promotion still requires a new independent held-out set because the old v2 held-out has already been consumed by earlier development.

---

### Task 12: Final regression, documentation, and rollback handoff

**Files**

- Modify: `README.md`
- Create: `docs/attention-reranker.md`
- Create: `dataset_artifacts/evaluation/attention-final-report.md`
- Create: `dataset_artifacts/evaluation/attention-execution-log.md`

**Step 1: Write the operator documentation**

Document:

- what attention does and does not do;
- offline Python training versus dependency-free JavaScript inference;
- exact input sources and leakage policy;
- model shape, 32-token context policy, and maximum four candidates;
- `OFF`, `SHADOW`, and `EXPERIMENTAL_ACTIVE` behavior;
- all thresholds and model/artifact hashes;
- how to rebuild, test, benchmark, evaluate, and diagnose fallback reasons;
- how UI-managed whitelist/dictionary/abbreviation/brand exceptions still take priority;
- why old held-out results cannot authorize production promotion.

**Step 2: Run the complete required suite**

```powershell
node --test
python -m unittest discover -s test -p test_attention_*.py -v
node tools/profile_engine.mjs
node tools/run_attention_eval.mjs `
  --config config/attention-tuning.json `
  --split dev `
  --output dataset_artifacts/evaluation/attention-final-dev.json
```

Expected: all tests PASS; repeated evaluation produces identical prediction and metric hashes; timing report includes environment and warmup details.

Do not run the previously consumed held-out set.

**Step 3: Produce the final report**

The report must contain one comparison table for:

- classical OFF p=0.95/o=1;
- attention SHADOW raw;
- attention thresholded;
- final selected mode.

Include semantic P/R/F0.5, spelling P/R/F0.5, clean FP, lane metrics, candidate oracle, stage losses, incremental TP/FP/precision, median/p95/p99, cold start, RSS, artifact bytes, model hash, config hash, and PASS/FAIL for each Task 11 gate.

**Step 4: Verify rollback**

Set `attentionRerankerMode='OFF'` in a test override and prove output equals the recorded classical p=0.95/o=1 baseline hash. Rollback must require no model rebuild and no data migration.

**Step 5: Record completion checkpoint**

Since this directory is not currently its own Git repository, do not initialize Git or create fake commits. Append commands, exit codes, hashes, metrics, and changed files to `attention-execution-log.md`. If the project later becomes a real repository, use one focused commit per completed task.

---

## Failure modes the implementation must handle

| Failure | Required behavior |
|---|---|
| Model absent, corrupt, wrong version, or wrong hash | Log diagnostic; use classical fallback |
| Candidate target absent | Count candidate miss; never inject gold |
| Context truncates target | Test must fail; selector must always preserve target |
| Input exceeds 32 tokens | Keep target window plus message-start budget deterministically |
| More than four candidates | Use frozen production diversity shortlist |
| All candidate scores weak | KEEP/classical fallback according to frozen thresholds |
| Protected token or UI exception | Skip attention and preserve rule decision |
| NaN/Inf or invalid probability | Reject attention result; classical fallback |
| Latency, memory, or quality gate fails | Remain in SHADOW |
| Dev improves but internal-test collapses | Remain in SHADOW and report likely overfit |

---

## Rules for the coding agent

1. Execute tasks in order; do not start a dependent task while an earlier stop condition is unresolved.
2. For every logic change: write the named RED test, run it and capture the expected failure, implement minimally, then run GREEN.
3. Reuse production tokenizer, guards, candidate generation, shortlist, and issue identity. Do not create a second spelling pipeline.
4. Never manufacture candidates, copy dev/test into training, reopen old held-out, or tune after seeing dev.
5. Never weaken whitelist, dictionary, abbreviation, brandname, exception, or protected-pattern rules to gain recall.
6. Do not add runtime dependencies. Production inference is pure `.mjs` plus typed arrays and local BIN/JSON.
7. Keep temporary corpora/checkpoints in `.tmp`; only deployable int8 BIN, small metadata, tests, config, docs, and reproducibility reports belong in project paths.
8. If a specified gate fails, record the evidence and keep SHADOW. A truthful failed experiment is an acceptable result.
9. Do not rewrite unrelated files or delete existing user changes.
10. After each task, append this record:

```markdown
### Task N checkpoint
- Status: PASS | STOPPED
- Files changed:
- RED command and observed failure:
- GREEN command and result:
- Data/model/config hashes:
- Metrics and resource use:
- Deviations with reason:
- Next task allowed: yes | no
```

---

## Definition of done

Implementation is done only when Tasks 0-12 are recorded, the whole test suite passes, artifacts are reproducible, JavaScript matches quantized Python, fallback is proven, evaluation counts all emitted issues, and Task 11 has mechanically selected either SHADOW or EXPERIMENTAL_ACTIVE.

`EXPERIMENTAL_ACTIVE` is not the required outcome. Correct measurement, safe fallback, and an honest gate verdict are required outcomes.
