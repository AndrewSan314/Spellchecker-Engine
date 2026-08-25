# Tiny Attention Spelling Reranker Implementation Plan — FIXED

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Also use `superpowers:test-driven-development` for every behavior change. Do not batch tasks.

**Goal:** Build and pretrain a tiny Vietnamese attention encoder, fine-tune it as a listwise spelling reranker over `KEEP_ORIGINAL + K` frozen production candidates, where `K` is selected from `{4,6,8}` using candidate-oracle evidence before model training, and permit `EXPERIMENTAL_ACTIVE` only when the full engine beats the agreed classical baseline by a meaningful margin without violating precision, clean-data, artifact-size, memory, or latency gates.

**Architecture:** Production remains rule-first. Existing deterministic rules, protected-range guards, UI/business allowlists, candidate generation, and shortlist logic run before attention. The base Vietnamese lexicon is a **soft feature**, not a hard skip for real-word errors. A tiny Transformer encodes the selected SMS context **once per target token** (maximum 32 document tokens). `KEEP_ORIGINAL` and the frozen candidate list are then scored with the same shared head using the context vector, option embedding, and existing classical features. The model can only select `KEEP_ORIGINAL` or one supplied candidate and can never generate text.

**Tech Stack:** Node.js ESM, JavaScript typed arrays, Python 3, PyTorch (training only), NumPy (artifact export/tests), JSON metadata + int8 binary weights, Node/Python test runners. Production must remain local and dependency-light.

---

## 0. Non-negotiable contract

The implementing agent must obey every item below.

1. Use Semble before locating code. Use GitNexus/CodeGraph impact analysis if an index exists.
2. Do not initialize Git. This workspace currently has no standalone Git history for `sms-validation-demo`. Use the execution log and SHA-256 manifests as checkpoints. If the user later provides a real repository, make one atomic commit per task.
3. Never read or run VSEC test, Viwiki external-test, benchmark external categories, or the old held-out artifact. This milestone has no untouched official held-out set.
4. Allowed supervised sources are `dataset_artifacts/vsec/vsec-train.jsonl` and `dataset_artifacts/clean-source/clean-train.txt`. Synthetic corruptions and hard negatives may be generated **only from those allowed training sources** and must still pass through production candidate generation; never inject gold candidates.
5. VSEC dev may be opened **once only**, in Task 10, after architecture, model weights, shortlist size, enabled lanes, and thresholds are frozen on the internal training/calibration data. The one Task 10 dev run must evaluate classical and attention on the same messages. Dev is for the experimental gate, never for retuning. Task 12 must not rerun dev.
6. Attention must never create candidates, alter offsets, bypass protected ranges, bypass explicit whitelist/customer dictionary/abbreviation/brand/allowed-term rules, or change deterministic rules.
7. Distinguish these two concepts everywhere:
   - **Hard guards:** protected ranges, explicit whitelist, customer-managed allowed dictionary, brand terms, abbreviation exceptions, and explicit allowed terms. Attention must skip these.
   - **Base Vietnamese lexicon membership:** a soft lexical feature only. A valid dictionary word such as `lên`, `mày`, `các`, or `đế` may still be a contextual real-word error and may reach attention when the real-word lane considers it ambiguous.
8. Candidate count is not hard-coded to four. Task 3 must measure `oracle@4`, `oracle@6`, and `oracle@8` on allowed TRAIN/calibration groups and freeze the smallest `K` whose oracle is within `0.5` percentage point of `oracle@8`. `K` must be one of `{4,6,8}` and never exceed 8. Context is hard-capped at 32 document tokens.
9. Production must not import PyTorch, NumPy, ONNX Runtime, native addons, or any new npm package in this milestone.
10. Modes are exactly `OFF`, `SHADOW`, and `EXPERIMENTAL_ACTIVE`. Do not name this production `ACTIVE` because no fresh external held-out set exists.
11. Missing artifact means disabled fallback. Present-but-invalid artifact fails closed to classical behavior and records a diagnostic; it must never crash validation for normal user input.
12. Do not change the LM corpus, n-gram counts, SymSpell limits, current deterministic rule thresholds, or raw datasets in this milestone. Candidate-shortlist `K` and the attention layer are the only intentional ranking-path changes.
13. Do not activate attention merely because unit tests pass. All mandatory gates in Task 11 must pass.
14. Precision is the priority. `KEEP_ORIGINAL` is a first-class option. If attention is uncertain, selects an invalid option, produces NaN/Inf, or fails a threshold, fall back to KEEP/classical behavior according to the frozen policy.

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

Historical dev reference from the 2026-08-25 calibration is **informational only** and must not trigger another dev read before Task 10:

| Metric | Historical classical reference |
|---|---:|
| semantic precision | 0.6829 |
| semantic recall | 0.2395 |
| semantic F0.5 | 0.4983 |
| incremental lane precision | approximately 0.806 |
| SMS <=160 p95 | remeasure on the same machine before attention |

Task 0 freezes the classical config and runtime baseline without opening dev. Task 10 creates the authoritative same-run dev baseline used by Task 11.

### Candidate-shortlist contract

Before training attention, Task 3 must compute candidate oracle using the exact production candidate generator and diversity shortlist:

```text
oracle@4
oracle@6
oracle@8
```

Freeze:

```text
K = smallest value in {4,6,8}
    such that oracle@K >= oracle@8 - 0.005
```

If even `oracle@8` is materially below the wide-pool oracle, stop and fix candidate retrieval/shortlisting before blaming attention. The attention model cannot recover a target it never receives.

### Model search contract

Do **not** lock the experiment to a single under-sized encoder before measuring it. Train only these three deployable candidates, all under the same data and seed policy:

| ID | Blocks | Hidden | Heads | Head size | FFN |
|---|---:|---:|---:|---:|---:|
| A | 1 | 48 | 2 | 24 | 96 |
| B | 2 | 64 | 4 | 16 | 128 |
| C | 2 | 96 | 4 | 24 | 192 |

Shared limits:

```text
max context tokens       32
max non-original options K selected in Task 3, K <= 8
output options           KEEP_ORIGINAL + candidate[0..K-1]
word vocabulary          <=8192 total IDs including special tokens
char hash buckets        4096
char n-grams             Unicode character 2..4-grams
position embeddings      32
production weights       int8 embeddings/matrices + float32 scales/biases/LN
artifact budget          <=10 MiB including metadata and binary weights
```

Select the smallest model that wins on calibration under the quality constraints. If A fails, B/C may be tried; do not increase beyond C in this milestone. Internal-test remains sealed until the selected architecture and thresholds are frozen in Task 10.

### Shared tokenizer IDs

Use one immutable ID contract in JavaScript and Python:

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

There is no separate `CLS/SEP` numbering. Both training and production use this exact contract.

### Token representation

```text
x_i = wordEmbedding[wordId]
    + mean(charEmbedding[hash(charNgram)])
    + positionEmbedding[position]
    + markerEmbedding[marker]
```

Model-input text is NFC-normalized and Unicode-lowercased for vocabulary efficiency. Original SMS surface text and offsets are never modified. Candidate output preserves the production candidate surface form.

### Encoder contract: Pre-LN

Use **Pre-LayerNorm** consistently in PyTorch and JavaScript:

```text
a  = x + MultiHeadSelfAttention(LayerNorm(x), mask)
h  = a + FFN_GELU(LayerNorm(a))
context = h[targetPosition]
```

For two-block models, feed `h` into the same Pre-LN block again. Do not mix this with the old post-LN formula.

### Single-encode listwise scoring

The 32-token context is encoded **once** per target token. Options do not re-run the Transformer.

For each option (`KEEP_ORIGINAL` uses the original token option embedding):

```text
optionEmbedding = word/char embedding of normalized option surface
z = concat(context,
           optionEmbedding,
           context * optionEmbedding,
           projectedClassicalFeatures,
           optionTypeEmbedding)
score = Linear(GELU(Linear(z)))
probabilities = maskedSoftmax(scores over KEEP + candidates)
```

Classical features may include n-gram score, frequency, edit distance, accent relation, base-lexicon membership, original prior, and production shortlist rank. Hard guards are applied before this stage and are not learnable features.

The JavaScript and PyTorch implementations must share all formulas and tensor layouts exactly.

## 1. Task dependency order

```text
Task 0 classical runtime/config baseline (no dev)
  -> Task 1 evaluation/provenance correctness
  -> Task 2 tokenizer/context contract
  -> Task 3 splits + oracle@K + hard/synthetic rows
  -> Task 4 vocabulary/pretraining corpus
  -> Task 5 tiny model candidates + pretraining
  -> Task 6 listwise fine-tuning + architecture selection
  -> Task 7 quantized export
  -> Task 8 pure-JS numerical parity
  -> Task 9 engine SHADOW integration
  -> Task 10 full-engine calibration + internal-test + one dev run
  -> Task 11 EXPERIMENTAL_ACTIVE decision
  -> Task 12 documentation and handoff
```

Do not start a task until the previous task's tests, artifacts, hashes, and execution-log entry are complete.

---

### Task 0: Freeze classical config and a runtime-only baseline without opening dev

**Files:**

- Create: `tools/run_attention_baseline.mjs`
- Create: `test/test_attention_baseline_scope.mjs`
- Create: `dataset_artifacts/evaluation/attention-classical-runtime-baseline.json` (generated)
- Create: `dataset_artifacts/evaluation/attention-classical-runtime-baseline.json.manifest.json` (generated)
- Create: `docs/plans/execution-log-attention-reranker.md`

**Step 1: Write the failing scope test**

Test that the baseline runner:

- **rejects** `dev`, `test`, `external-test`, `viwiki`, `heldout`, and benchmark external-category path markers before reading;
- forces the exact classical overrides from the agreed baseline;
- records source/config/LM/reranker hashes;
- records full-engine latency, cold-start, RSS, Node version, CPU, warmup, and run count;
- records `attentionContributionMs=0`;
- does not claim authoritative semantic dev metrics.

Do not mock file access. Import pure helpers from the runner and test them directly.

**Step 2: Run RED**

```powershell
node --test test/test_attention_baseline_scope.mjs
```

Expected: FAIL because `tools/run_attention_baseline.mjs` does not exist or still permits dev.

**Step 3: Implement the minimal runtime/config runner**

Use only safe local smoke/profile inputs already permitted by the repository. Do not read VSEC dev. Reuse the existing profiler percentile definition.

Required artifact fields:

```json
{
  "schema": "attention-classical-runtime-baseline-v2",
  "devOpened": false,
  "overrides": {},
  "hashes": {},
  "runtime": {
    "sms160P50Ms": 0,
    "sms160P95Ms": 0,
    "coldStartMs": 0,
    "rssBytes": 0
  }
}
```

Historical quality numbers may be copied into documentation as `historicalReference`, but the runner must not recompute them.

**Step 4: Run GREEN and generate the artifact**

```powershell
node --test test/test_attention_baseline_scope.mjs
node tools/run_attention_baseline.mjs `
  --out dataset_artifacts/evaluation/attention-classical-runtime-baseline.json
```

Expected: test PASS; `devOpened=false`; runtime values and hashes are present. Stop if the classical overrides cannot be reproduced.

**Step 5: Record checkpoint**

Append Task 0 status, commands, config/source hashes, artifact SHA-256, latency, RSS, and deviations to `docs/plans/execution-log-attention-reranker.md`.

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

Both languages must NFC-normalize and Unicode-lowercase the **model input only**. Never strip Vietnamese accents. Original SMS surface text and offsets remain untouched.

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

### Task 3: Create leakage-safe ranking/message splits, choose K, and mine hard training rows

**Files:**

- Create: `tools/extract_attention_ranking_rows.mjs`
- Create: `src/attention-ranking-schema.mjs`
- Create: `test/test_attention_ranking_rows.mjs`
- Create: `.tmp/attention-ranking-{train,calibration,internal-test}.jsonl` (generated)
- Create: `.tmp/attention-messages-{train,calibration,internal-test}.jsonl` (generated)
- Create: `.tmp/attention-ranking-split-manifest.json` (generated)
- Create: `.tmp/attention-shortlist-config.json` (generated and frozen before Task 4)

**Step 1: Write RED schema, leakage, and candidate-oracle tests**

Every file begins with a provenance header. Every ranking row contains:

```json
{
  "recordType": "ranking-row",
  "id": "stable-id",
  "groupId": "normalized-source-sentence-hash",
  "source": "vsec-train|clean-train|synthetic-clean-train|hard-negative-clean-train",
  "context": {},
  "original": "",
  "candidates": [],
  "classicalFeatures": [],
  "labelIndex": 0,
  "lane": "ACCENTED_SAME_KEY|UNACCENTED_SAME_KEY|DIFFERENT_KEY_REAL_WORD|UNKNOWN_TYPO|CLEAN_KEEP|HARD_NEGATIVE_KEEP"
}
```

Rules:

- option index 0 is always `KEEP_ORIGINAL`;
- candidates come only from production `buildCorrectionCandidates` + production diversity shortlist;
- gold is never injected;
- correct VSEC/synthetic target absent from the wide pool is a `candidate-miss` and the correction row is not trainable;
- target present in wide pool but absent from shortlist is a `shortlist-miss` and is counted in oracle diagnostics;
- clean and hard-negative rows always label index 0;
- no `groupId` may occur in more than one attention split;
- path guard rejects forbidden held-out markers before any read;
- base Vietnamese lexicon membership is exported as a classical feature and **must not** cause an automatic skip for real-word rows;
- explicit whitelist/customer dictionary/brand/abbreviation/protected terms remain hard guards and must not create attention rows.

**Step 2: Define deterministic split**

Hash `groupId + "attention-v2-20260825"` with SHA-256:

```text
00..79 train
80..89 calibration
90..99 internal-test
```

This is an internal split of allowed TRAIN sources, not an official external test. Do not inspect internal-test quality during architecture or threshold tuning.

**Step 3: Build wide candidate pools and freeze shortlist K**

Using only train + calibration groups, compute:

```text
wide-pool oracle
oracle@4
oracle@6
oracle@8
```

Freeze:

```text
K = smallest {4,6,8} with oracle@K >= oracle@8 - 0.005
```

Write `.tmp/attention-shortlist-config.json` with `K`, oracle values, generator/config hashes, and selection reason. All later tasks must load this file rather than hard-code candidate count.

Stop condition: if `oracle@8` is still materially below wide-pool oracle, stop and repair candidate retrieval/shortlisting before training attention.

**Step 4: Generate hard-negative KEEP rows**

Run the frozen classical engine on clean-train groups. Whenever it proposes a spelling correction on a clean sentence, emit a `HARD_NEGATIVE_KEEP` row using the **same production candidate pool** and label `KEEP_ORIGINAL`.

Examples of the behavior this is intended to teach:

```text
mang -> KEEP mang, not mạng
đáp  -> KEEP đáp, not đãi
giải -> KEEP giải, not giảm
```

Never whitelist these ordinary words merely to hide false positives.

**Step 5: Generate synthetic correction rows from clean-train**

Create deterministic corruption operators on clean TRAIN groups only:

- remove Vietnamese diacritics from one target token;
- wrong tone/diacritic within the same accent key;
- one-character delete/insert/substitute/transpose where realistic;
- limited real-word corruption only when the corrupted surface is itself in the base lexicon.

After corruption, call production candidate generation. If the clean target is not returned, record the miss and drop that ranking row. Do not inject the answer.

All synthetic rows inherit the original clean sentence `groupId`, so original and corrupt variants cannot cross splits.

**Step 6: Create message-level calibration/internal-test files**

In addition to ranking rows, emit complete message-level evaluation records for calibration and internal-test. These records must preserve all labels/clean status necessary to run the **full ValidationEngine** and count every emitted issue. Task 10 calibrates thresholds on these message-level files, not on target-only ranking rows.

**Step 7: Run RED then GREEN**

```powershell
node --test test/test_attention_ranking_rows.mjs
node tools/extract_attention_ranking_rows.mjs `
  --vsec-train dataset_artifacts/vsec/vsec-train.jsonl `
  --clean-train dataset_artifacts/clean-source/clean-train.txt `
  --out-dir .tmp
```

Expected: PASS; non-empty train/calibration/internal-test; zero group overlap; options are `[KEEP + 1..K candidates]`; manifest contains source hashes, row counts, hard-negative counts, synthetic counts, wide-pool misses, shortlist misses, oracle@4/6/8, and frozen K.

**Step 8: Write pretraining deny list**

The manifest must include normalized sentence hashes for calibration and internal-test groups. Task 4 excludes those exact groups from self-supervised pretraining.

**Step 9: Record checkpoint**

Record counts by source/lane/label, `oracle@4/6/8`, chosen K, candidate-miss/shortlist-miss counts, KEEP/correction balance, and SHA-256 for all outputs.

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
- special IDs exactly match Task 2: `PAD=0`, `UNK=1`, `BOS=2`, `EOS=3`, `TARGET=4`, `PROTECTED=5`, `PUNCT=6`, `MASK=7`;
- vocabulary has at most 8,192 total word IDs including the eight fixed special IDs; hashed character buckets are separate;
- sequences are at most 32 tokens and padding positions have mask zero.

**Step 2: Run RED**

```powershell
python -m unittest test.test_attention_pretrain_data -v
```

Expected: FAIL because the builder does not exist.

**Step 3: Implement the deterministic builder**

- Normalize Unicode to NFC and Unicode-lowercase model-input tokens exactly as Task 2 does; preserve Vietnamese diacritics. Original corpus text is never rewritten in place.
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
  --deny-list .tmp/attention-ranking-split-manifest.json `
  --out-dir .tmp --seed 20260825
```

Expected: PASS; zero denied hashes, vocabulary at most 8,192, and SHA-256 for every shard.

**Step 5: Record checkpoint**

Record source hashes, lines seen/accepted/deduplicated/rejected, vocabulary size, shard count, and peak RSS. Do not copy the raw corpus into `src/data`.

---

### Task 5: Implement and self-pretrain the tiny Pre-LN encoder candidates in PyTorch

**Files**

- Create: `tools/attention_model.py`
- Create: `tools/pretrain_attention_encoder.py`
- Create: `test/test_attention_model.py`
- Create: `test/test_attention_pretraining.py`
- Generate, do not commit: `.tmp/attention-pretrained-{A,B,C}.pt`
- Generate, do not commit: `.tmp/attention-pretrain-report-{A,B,C}.json`

**Step 1: Write RED architecture tests**

Implement only architectures A/B/C from the model-search contract. Assert for each:

- maximum 32 document tokens;
- exact hidden/head/FFN dimensions;
- **Pre-LN** residual formula: `x + Attention(LN(x))`, then `a + FFN(LN(a))`;
- word + hashed-character + position + marker embeddings;
- stable masked softmax;
- no decoder or generation path;
- same tokenizer IDs as Task 2/4;
- parameter-count and estimated int8 artifact ceilings;
- deterministic forward output for the same seed;
- no NaN/Inf.

Also add a test that fails if the old post-LN formula is used.

**Step 2: Write RED pretraining tests**

Use masked/replaced-token prediction:

- select 15% of non-special, non-padding tokens;
- replace selected tokens 80% with `MASK=7`, 10% with a random learned-vocabulary token, and leave 10% unchanged;
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

Starting pretraining configuration for each candidate: AdamW `lr=3e-4`, weight decay `0.01`, batch size `128` or largest safe size, gradient clipping `1.0`, at most 5 epochs, early stop after two validation checks without improvement. Make a deterministic 5% pretraining-validation partition from **train groups only**; calibration/internal-test sentence hashes remain denied.

Checkpoint metadata contains architecture ID, state, vocabulary hash, tokenizer version, shortlist-config hash, seed, and source-manifest hash. It contains no raw text.

**Step 5: Run GREEN and pretrain A/B/C**

```powershell
python -m unittest test.test_attention_model test.test_attention_pretraining -v
python tools/pretrain_attention_encoder.py --arch A `
  --data-manifest .tmp/attention-pretrain-manifest.json `
  --vocab .tmp/attention-vocab.json --output .tmp/attention-pretrained-A.pt --seed 20260825
python tools/pretrain_attention_encoder.py --arch B `
  --data-manifest .tmp/attention-pretrain-manifest.json `
  --vocab .tmp/attention-vocab.json --output .tmp/attention-pretrained-B.pt --seed 20260825
python tools/pretrain_attention_encoder.py --arch C `
  --data-manifest .tmp/attention-pretrain-manifest.json `
  --vocab .tmp/attention-vocab.json --output .tmp/attention-pretrained-C.pt --seed 20260825
```

Expected: PASS; each candidate improves validation loss over its untrained control; no dev/test/held-out path appears in reports.

**Step 6: Stop condition**

If a candidate cannot learn the tiny fixture or real pretraining does not beat its untrained control, mark that architecture failed. Fix masks, tokenizer parity, data, or objective before considering size increases. Do not create architecture D.

---

### Task 6: Fine-tune a single-encode listwise spelling reranker and select architecture on calibration

**Files**

- Create: `tools/train_attention_reranker.py`
- Create: `test/test_attention_finetuning.py`
- Generate, do not commit: `.tmp/attention-finetuned-{A,B,C}.pt`
- Generate, do not commit: `.tmp/attention-finetune-report-{A,B,C}.json`
- Generate, do not commit: `.tmp/attention-selected-model.json`

**Step 1: Write RED ranking tests**

Each row has `2..(K+1)` choices in this exact order:

```text
[KEEP_ORIGINAL, candidate_1, ..., candidate_K]
```

Assert that:

- gold is unmasked;
- KEEP is gold for clean and hard-negative rows;
- a correction is gold only when production candidate generation and the frozen K-shortlist found it;
- padding options receive zero probability;
- candidate permutations preserve remapped scores;
- candidate count comes from `.tmp/attention-shortlist-config.json`, never a literal 4;
- a tiny mixed fixture can overfit;
- no missing target is injected;
- train/calibration/internal-test group hashes are disjoint;
- base-lexicon membership is available as a feature and does not force KEEP/skip;
- explicit hard-guard rows never reach training.

**Step 2: Write RED single-encode tests**

Instrument the model and prove one ranking row performs exactly **one Transformer context encoding**, regardless of whether it has 1, 4, 6, or 8 candidates. Candidate option embeddings are computed separately with the shared word/char embedding path; candidates never trigger another self-attention pass.

**Step 3: Run RED**

```powershell
python -m unittest test.test_attention_finetuning -v
```

Expected: FAIL because the fine-tuner/single-encode scorer does not exist.

**Step 4: Implement listwise scoring**

For one row:

```text
32-token context
    -> Transformer exactly once
    -> context[targetPosition]

KEEP/candidate option embeddings
    -> shared option scorer with context + classical features
    -> masked softmax over all available options
```

Loss:

```text
L = cross_entropy(choice_logits, gold_index)
  + 0.1 * max(0, 0.2 - gold_logit + highest_wrong_logit)
```

Use AdamW `lr=1e-4`, batch 64 or largest safe size, at most 10 epochs, clipping `1.0`. Stratify batches across clean KEEP, hard-negative KEEP, accented, unaccented, unknown typo, and different-key real-word rows. Cap any lane weight at 3x.

Hard negatives and clean KEEP must be sampled often enough that the model learns abstention, not just correction.

**Step 5: Select architecture using calibration only**

Fine-tune A/B/C on ranking-train. Evaluate only on ranking-calibration. Do not open internal-test or VSEC dev.

Select the **smallest** architecture that:

- beats the classical row-level ranker on calibration F0.5;
- does not reduce calibration precision below the agreed classical floor;
- improves KEEP accuracy or at least does not materially regress it;
- does not obtain its gain mainly by increasing clean false corrections;
- is projected to fit the 10 MiB artifact budget.

If multiple models satisfy the gates within `0.5` F0.5 percentage point of the best, choose the smaller one. Write the exact selected architecture/checkpoint/vocab/tokenizer/shortlist hashes to `.tmp/attention-selected-model.json`.

**Step 6: Run GREEN and train**

```powershell
python -m unittest test.test_attention_finetuning -v
python tools/train_attention_reranker.py `
  --rows-dir .tmp `
  --split-manifest .tmp/attention-ranking-split-manifest.json `
  --shortlist-config .tmp/attention-shortlist-config.json `
  --pretrained-pattern .tmp/attention-pretrained-{ARCH}.pt `
  --output-dir .tmp `
  --seed 20260825
```

Expected: PASS; reports include precision, recall, F0.5, KEEP accuracy, candidate-choice accuracy, metrics by lane, architecture, and single-encode timing.

**Step 7: Stop condition**

Stop before export if no deployable architecture beats the classical row-level ranker on calibration without a clean/KEEP regression. Do not inspect internal-test and do not increase model size beyond C.

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

Assert two exports are byte-identical; corrupt magic/version/length/offset/shape/checksum is rejected; weights and embeddings are int8; **word/character embedding tables use symmetric per-row scales**; linear matrices use symmetric per-output-channel scales; biases, layer norms, and scales are float32; total BIN+JSON is at most `10 * 1024 * 1024`; metadata contains architecture, selected K, tokenizer contract, and all input hashes.

**Step 2: Run RED**

```powershell
python -m unittest test.test_attention_export -v
```

Expected: FAIL because exporter/artifacts do not exist.

**Step 3: Implement and export**

Quantize the frozen checkpoint. Do not use ONNX, TensorFlow.js, native addons, `onnxruntime-node`, or runtime downloads. Keep large arrays in BIN, not JSON.

```powershell
python tools/export_attention_model.py `
  --checkpoint-from .tmp/attention-selected-model.json `
  --vocab .tmp/attention-vocab.json `
  --output-bin src/data/attention-reranker.int8.bin `
  --output-meta src/data/attention-reranker.json
python -m unittest test.test_attention_export -v
```

Expected: PASS; deterministic artifact is no larger than 10 MiB.

**Step 4: Stop condition**

If quantization drops calibration row-level F0.5 by more than 0.5 percentage point, drops precision by more than 0.2 point, or changes KEEP/candidate argmax materially, fix quantization/export math. Do not ship a float checkpoint.

---

### Task 8: Implement pure-JavaScript typed-array inference with Python parity

**Files**

- Create: `src/attention-reranker.mjs`
- Create: `tools/export_attention_parity_fixture.py`
- Create: `test/fixtures/attention-parity.json`
- Create: `test/test_attention_reranker.mjs`

**Step 1: Export a frozen parity fixture**

Include at least 20 synthetic/public cases covering NFC/NFD Vietnamese, OOV, padding, one candidate and the frozen maximum K candidates, all spelling lanes, KEEP, short text, and text longer than 32 tokens. Store IDs, masks, option map, and Python logits/probabilities; do not copy private corpus text.

**Step 2: Write RED JavaScript tests**

Assert loader validation, one load per engine instance, built-in JavaScript plus typed arrays only, logit max error `<=1e-3`, probability max error `<=2e-3`, candidate permutation invariance, padding never wins, no NaN/Inf, deterministic outputs, and malformed/missing artifacts return a classical-fallback status instead of throwing through `validate()`.

**Step 3: Run RED**

```powershell
python tools/export_attention_parity_fixture.py `
  --checkpoint-from .tmp/attention-selected-model.json `
  --artifact src/data/attention-reranker.int8.bin `
  --output test/fixtures/attention-parity.json
node --test test/test_attention_reranker.mjs
```

Expected: JavaScript test FAIL because inference does not exist.

**Step 4: Implement the minimal runtime**

Implement only what the selected architecture requires: int8 embedding lookup with per-row scales, float32 residuals, **Pre-LN** with Python's epsilon, Q/K/V projections, selected-head scaled attention, stable masked softmax, identical GELU approximation, FFN projections, shared option head, and final softmax. The runtime must encode context once and reuse that context vector for every option. Parse once into typed-array views and reuse safe scratch buffers.

Do not add a generic tensor framework, dynamic graph, WebAssembly compiler, worker thread, native dependency, or network fetch.

**Step 5: Run GREEN**

```powershell
node --test test/test_attention_reranker.mjs
```

Expected: PASS for parity and corrupt-artifact cases.

**Step 6: Record runtime checkpoint**

Record cold load time, bytes, RSS delta, median and p95 batch-one inference, Node version, CPU, and run count. Task 11 applies the full-engine gate.

---

### Task 9: Integrate attention in SHADOW without weakening hard guards

**Files**

- Modify: `src/config.mjs`
- Modify: `src/engine.mjs`
- Modify: `src/rules/linguistic-rules.mjs`
- Create: `test/test_attention_integration.mjs`

**Step 1: Write RED integration tests**

Cover these invariants:

- `OFF` is behaviorally identical to the frozen classical engine;
- `SHADOW` records a decision but never changes emitted issues;
- attention runs only after deterministic rules, protected-pattern guards, explicit whitelist/customer allowed dictionary, abbreviation, brand, and explicit exception guards;
- **base Vietnamese lexicon membership alone does not hard-skip a real-word token**; it is passed as a feature when the real-word lane considers the token ambiguous;
- it only ranks the frozen production shortlist and can choose KEEP; it never generates a candidate;
- no candidate outside the shortlist can appear in output;
- missing/corrupt/incompatible model falls back to the classical p=0.95/o=1 decision;
- one token produces at most one spelling issue;
- URLs, emails, phone numbers, prices, dates, codes, brandnames, explicit whitelist/customer allowed terms, abbreviations, and explicit exceptions remain protected;
- same-key accented and different-key real-word candidates can reach SHADOW scoring when their hard guards allow it, even if the current classical emission lane is disabled;
- repeated calls are deterministic and do not reload the artifact;
- context self-attention runs once per target token, independent of candidate count.

**Step 2: Run RED**

```powershell
node --test test/test_attention_integration.mjs
```

Expected: FAIL because engine integration does not exist or because base dictionary membership is still treated as a universal hard skip.

**Step 3: Add explicit modes and config validation**

```js
attentionRerankerMode: 'SHADOW', // OFF | SHADOW | EXPERIMENTAL_ACTIVE
attentionMaxTokens: 32,
attentionMaxCandidates: '<load frozen K from artifact/config>',
attentionMinProbability: null,
attentionMinMarginOverKeep: null,
attentionMinMarginOverSecond: null,
```

Invalid values fail at startup. `attentionMaxCandidates` must agree with model metadata and the frozen shortlist config. Thresholds stay null until Task 10 freezes them. The safe fallback is classical p=0.95/o=1.

**Step 4: Integrate at one ownership point**

Flow must be:

```text
token + deterministic/protected/business-guard result
  -> production candidate generation
  -> production diversity shortlist (frozen K <= 8)
  -> classical decision
  -> optional attention single-encode scoring
  -> mode switch
       OFF: classical
       SHADOW: classical + diagnostics
       EXPERIMENTAL_ACTIVE: gated attention or classical fallback
  -> issue deduplication/emission
```

Do not place a generic `baseLexicon.contains(token) -> skip attention` before the real-word ambiguity path. Explicit UI/customer allowlists remain hard skips.

Expose diagnostic fields only in evaluation/debug output: model/version hash, tokenizer hash, K, context indices, shortlist, KEEP/candidate probabilities, margins, chosen option, mode, fallback reason, base-lexicon feature, and elapsed microseconds. Never include full corpus content in deployable logs.

**Step 5: Run GREEN and regression tests**

```powershell
node --test test/test_attention_integration.mjs
node --test test/test_core_engine.mjs test/test_linguistic_rules.mjs test/test_recall_reranker.mjs
```

Expected: PASS; OFF and SHADOW user-visible output hashes equal the frozen classical behavior. Hard-guard behavior remains unchanged.

---

### Task 10: Calibrate on full-engine emissions, open internal-test once, then open VSEC dev once

**Files**

- Create: `tools/run_attention_eval.mjs`
- Create: `tools/calibrate_attention_reranker.mjs`
- Create: `test/test_attention_evaluation.mjs`
- Generate: `dataset_artifacts/evaluation/attention-calibration.json`
- Generate: `dataset_artifacts/evaluation/attention-internal-test.json`
- Generate: `dataset_artifacts/evaluation/attention-classical-dev-baseline.json`
- Generate: `dataset_artifacts/evaluation/attention-shadow-dev.json`

**Step 1: Write RED evaluator tests**

Use tiny fixtures proving:

- every emitted issue at every token is counted, not only the labeled span;
- an extra issue elsewhere in a positive message is a false positive;
- clean messages contribute all unexpected issues as false positives;
- new issues are computed by stable issue identity relative to classical output;
- attention stage attribution uses real returned fields, not empty `ranked`/`cheapKept` arrays;
- candidate miss, shortlist miss, context-selector miss, KEEP choice, wrong-candidate choice, threshold decline, hard-guard block, and fallback are distinct stages;
- provenance contains dataset hash, model hash, tokenizer/vocab hash, shortlist K/config hash, engine config, git/worktree identifier when available, timestamp, Node version, CPU, warmup, run count, and metric definitions;
- evaluator rejects forbidden old held-out paths;
- the dev path cannot be run before a frozen calibration config exists;
- the dev runner writes classical and attention results in the **same invocation** and records `devOpenCount=1` for this milestone.

**Step 2: Run RED**

```powershell
node --test test/test_attention_evaluation.mjs
```

Expected: FAIL, including regression cases for the old span-only precision bug and ranking-row-only calibration.

**Step 3: Implement full-engine evaluator**

For every message, run the actual `ValidationEngine` and compare:

1. frozen classical p=0.95/o=1 output;
2. attention SHADOW raw argmax diagnostics;
3. attention thresholded decision;
4. final user-visible result under the selected mode.

Report semantic precision/recall/F0.5, spelling-only metrics, clean false positives, protected/red-team regressions, metrics by lane/relation, wide-pool and shortlist oracle, KEEP accuracy, attention incremental TP/FP/precision, stage-loss counts, latency median/p95/p99, cold-start, RSS, and artifact bytes.

**Step 4: Calibrate thresholds on message-level calibration only**

Grid-search Task 3 **message-level calibration records through the full engine**, never target-only ranking rows, over:

- minimum selected-choice probability;
- minimum margin over KEEP;
- minimum margin over second-best candidate;
- allowed relation lanes.

Optimization objective:

```text
maximize full-engine semantic F0.5
subject to:
  precision >= classical calibration precision - 0.002
  new clean false positives = 0
  new protected/red-team regressions = 0
```

Break ties by higher precision, higher recall, lower p95, fewer enabled lanes, smaller model, then lexicographic config JSON.

Freeze the winning thresholds, enabled lanes, architecture/model hash, tokenizer hash, K, and artifact hash in `attention-calibration.json` **before opening internal-test**.

**Step 5: Run internal-test exactly once**

Run the frozen full-engine configuration on `.tmp/attention-messages-internal-test.jsonl`. Do not retune from this result. If precision or F0.5 collapses materially versus calibration, remain SHADOW and do not open VSEC dev until the failure is understood with TRAIN/calibration-only diagnostics. Any changed model/threshold requires a newly frozen config and a new internal-test policy checkpoint; do not repeatedly optimize against internal-test.

**Step 6: Open VSEC dev once and evaluate classical + attention together**

Only after architecture, model, K, lanes, and thresholds are frozen:

```powershell
node --test test/test_attention_evaluation.mjs
node tools/calibrate_attention_reranker.mjs `
  --messages .tmp/attention-messages-calibration.jsonl `
  --output dataset_artifacts/evaluation/attention-calibration.json
node tools/run_attention_eval.mjs `
  --config dataset_artifacts/evaluation/attention-calibration.json `
  --split internal-test `
  --output dataset_artifacts/evaluation/attention-internal-test.json
node tools/run_attention_eval.mjs `
  --config dataset_artifacts/evaluation/attention-calibration.json `
  --split dev `
  --classical-out dataset_artifacts/evaluation/attention-classical-dev-baseline.json `
  --attention-out dataset_artifacts/evaluation/attention-shadow-dev.json `
  --open-dev-once
```

Expected: tests PASS; calibration/internal-test/dev artifacts contain complete provenance. The dev invocation evaluates both systems against identical messages and environment settings. **Do not retune or rerun dev after seeing it.** A dev failure means SHADOW.

**Step 7: Record checkpoint**

Record full-engine calibration metrics, internal-test metrics, one-time dev metrics, hashes, `devOpenCount`, candidate/shortlist stage losses, and resource usage.

---

### Task 11: Gate EXPERIMENTAL_ACTIVE mechanically against the same-run classical dev baseline

**Files**

- Create: `config/attention-tuning.json`
- Create: `tools/select_attention_mode.mjs`
- Create: `test/test_attention_activation_gate.mjs`
- Modify only on PASS: `src/config.mjs`

**Step 1: Write RED gate tests**

Create one fixture per failing gate and prove the selector returns `SHADOW`. Only a fixture satisfying every gate returns `EXPERIMENTAL_ACTIVE`. Missing metrics, NaN, stale hashes, a dev-open-count other than one, or provenance mismatch are failures.

**Step 2: Run RED**

```powershell
node --test test/test_attention_activation_gate.mjs
```

Expected: FAIL because the selector does not exist.

**Step 3: Implement the pure selector**

All quality gates are relative to the **authoritative classical result produced in the same Task 10 dev invocation**, not the historical numbers:

| Gate | Required value |
|---|---:|
| Semantic precision | `attention >= classical - 0.002` |
| Semantic recall | `attention >= classical + 0.010` |
| Semantic F0.5 | `attention >= classical + 0.010` |
| Attention actual incremental precision | `>= 0.800` |
| New clean false positives | `0` |
| New protected/red-team regressions | `0` |
| Multidimensional-test drop | `<= 0.5` percentage point |
| Attention-only p95 delta | `<= 3 ms` |
| Full-engine p95 | `<= 15 ms` |
| Cold-start delta | `<= 250 ms` |
| Model RSS delta | `<= 20 MiB` |
| BIN + JSON artifact | `<= 10 MiB` |
| Python/JS parity, core, rules, integration | all PASS |
| Output determinism | identical hashes across repeated non-dev fixtures/runs |
| Frozen K | one of `{4,6,8}` and matches model/config metadata |
| Dev use | one authorized Task 10 dev invocation; no retuning after it |

Also require internal-test not to be materially worse than calibration:

```text
precision drop <= 2.0 percentage points
F0.5 drop     <= 2.0 percentage points
```

The recall/F0.5 improvement gates are intentionally meaningful deltas, not `>0.000x` wins. Do not lower them merely to activate the experiment.

**Step 4: Generate frozen config and verdict**

```powershell
node tools/select_attention_mode.mjs `
  --runtime-baseline dataset_artifacts/evaluation/attention-classical-runtime-baseline.json `
  --classical-dev dataset_artifacts/evaluation/attention-classical-dev-baseline.json `
  --calibration dataset_artifacts/evaluation/attention-calibration.json `
  --internal-test dataset_artifacts/evaluation/attention-internal-test.json `
  --dev dataset_artifacts/evaluation/attention-shadow-dev.json `
  --output config/attention-tuning.json
node --test test/test_attention_activation_gate.mjs
```

Expected: selector prints every gate as PASS/FAIL and writes the verdict with all input hashes.

**Step 5: Activate only if verdict is PASS**

- PASS: set default to `EXPERIMENTAL_ACTIVE` and load exactly the frozen thresholds/model/K hashes.
- FAIL: keep `SHADOW`; do not relax a threshold, change K, remove a lane, or edit evaluation data merely to turn it green.
- At runtime, any hash/config/model/load error automatically uses classical p=0.95/o=1 for that process.

This is experimental activation only. Production promotion still requires a new independent held-out set because the old held-out and current dev are development-consumed.

---

### Task 12: Final regression, documentation, and rollback handoff without reopening dev

**Files**

- Modify: `README.md`
- Create: `docs/attention-reranker.md`
- Create: `dataset_artifacts/evaluation/attention-final-report.md`
- Create: `dataset_artifacts/evaluation/attention-execution-log.md`

**Step 1: Write the operator documentation**

Document:

- what attention does and does not do;
- offline Python training versus dependency-free JavaScript inference;
- exact input sources, synthetic/hard-negative policy, and leakage policy;
- selected architecture, Pre-LN formula, 32-token context policy, and frozen K in `{4,6,8}`;
- single-context-encode behavior;
- distinction between hard guards and base Vietnamese lexicon as a soft feature;
- `OFF`, `SHADOW`, and `EXPERIMENTAL_ACTIVE` behavior;
- all thresholds and model/artifact/tokenizer/shortlist hashes;
- how to rebuild, test, benchmark, evaluate safe internal fixtures, and diagnose fallback reasons;
- how UI-managed whitelist/customer dictionary/abbreviation/brand exceptions still take priority;
- why VSEC dev and the old held-out cannot authorize full production promotion.

**Step 2: Run the complete regression/runtime suite — do not rerun dev**

```powershell
node --test
python -m unittest discover -s test -p test_attention_*.py -v
node tools/profile_engine.mjs
```

Use the frozen Task 10 dev artifacts by hash. Do **not** execute `--split dev` again. Repeated determinism tests must use synthetic/public/internal-safe fixtures, not VSEC dev.

Expected: all tests PASS; timing report includes environment and warmup details; runtime model/config hashes equal the Task 10/11 frozen hashes.

**Step 3: Produce the final report**

The report must contain one comparison table for the already-recorded Task 10 results:

- classical p=0.95/o=1 same-run dev baseline;
- attention SHADOW raw;
- attention thresholded;
- final selected mode.

Include semantic P/R/F0.5, spelling P/R/F0.5, clean FP, lane metrics, wide-pool oracle, oracle@K, shortlist losses, stage losses, incremental TP/FP/precision, median/p95/p99, cold start, RSS, artifact bytes, architecture, K, model hash, tokenizer hash, config hash, and PASS/FAIL for each Task 11 gate.

**Step 4: Verify rollback**

Set `attentionRerankerMode='OFF'` in a test override and prove output equals the frozen classical behavior on the safe regression fixture set. Rollback must require no model rebuild and no data migration. Also prove missing/corrupt model reverts to classical behavior.

**Step 5: Record completion checkpoint**

Since this directory is not currently its own Git repository, do not initialize Git or create fake commits. Append commands, exit codes, hashes, metrics, and changed files to `attention-execution-log.md`. If the project later becomes a real repository, use one focused commit per completed task.

---

## Failure modes the implementation must handle

| Failure | Required behavior |
|---|---|
| Model absent, corrupt, wrong version, or wrong hash | Log diagnostic; use classical fallback |
| Candidate target absent from wide pool | Count `candidate-miss`; never inject gold |
| Target present wide but absent from frozen K shortlist | Count `shortlist-miss`; never inject gold |
| `oracle@8` materially below wide-pool oracle | Stop before attention training and repair candidate retrieval/shortlisting |
| Context truncates target | Test must fail; selector must always preserve target |
| Input exceeds 32 tokens | Keep target, deterministic nearest context, and message-start budget |
| More candidates than frozen K | Use the frozen diversity shortlist; K must be 4/6/8 and <=8 |
| Base lexicon says original is a valid Vietnamese word | Treat as a soft feature; do not hard-skip if real-word ambiguity lane is eligible |
| Explicit whitelist/customer allowed term/brand/abbreviation/protected token | Hard skip attention and preserve rule/business decision |
| All candidate scores weak | KEEP/classical fallback according to frozen thresholds |
| NaN/Inf or invalid probability | Reject attention result; classical fallback |
| Python/JS token IDs or Pre-LN math disagree | Stop; parity tests must fail |
| Quantized embedding/matrix metadata mismatch | Reject artifact; classical fallback |
| Latency, memory, artifact-size, precision, clean-FP, or quality gate fails | Remain in SHADOW |
| Internal-test collapses after calibration | Remain SHADOW; diagnose without optimizing repeatedly against internal-test |
| Dev fails after the one authorized run | Remain SHADOW; do not retune or rerun dev |

---

## Rules for the coding agent

1. Execute tasks in order; do not start a dependent task while an earlier stop condition is unresolved.
2. For every logic change: write the named RED test, run it and capture the expected failure, implement minimally, then run GREEN.
3. Reuse production tokenizer/document, deterministic guards, candidate generation, diversity shortlist, classical features, and issue identity. Do not create a second spelling pipeline.
4. Never manufacture candidates, copy dev/test into training, reopen the old held-out, or tune after the authorized dev run.
5. Never weaken protected ranges, explicit whitelist/customer allowed dictionary, abbreviation, brandname, or explicit exception rules to gain recall. **Do not confuse these hard guards with the base Vietnamese lexicon, which is a soft feature for real-word detection.**
6. Do not add runtime dependencies. Production inference is pure `.mjs` plus typed arrays and local BIN/JSON in this milestone.
7. Encode the SMS context once per target token. Never run self-attention once per candidate.
8. Load the frozen shortlist K from metadata/config; do not hard-code `4` in model, training, inference, or tests.
9. Keep temporary corpora/checkpoints in `.tmp`; only deployable int8 BIN, small metadata, tests, config, docs, and reproducibility reports belong in project paths.
10. If a specified gate fails, record the evidence and keep SHADOW. A truthful failed experiment is an acceptable result.
11. Do not rewrite unrelated files or delete existing user changes.
12. After each task, append this record:

```markdown
### Task N checkpoint
- Status: PASS | STOPPED
- Files changed:
- RED command and observed failure:
- GREEN command and result:
- Data/model/config/tokenizer/shortlist hashes:
- Metrics and resource use:
- Dev opened in this task: yes | no
- Deviations with reason:
- Next task allowed: yes | no
```

---

## Definition of done

Implementation is done only when Tasks 0-12 are recorded, the whole test suite passes, artifacts are reproducible, JavaScript matches quantized Python, candidate K is oracle-justified and frozen, context is encoded once per target, hard guards are preserved while base-lexicon real-word errors remain reachable, full-engine calibration counts all emitted issues, rollback/fallback are proven, VSEC dev was opened only in the authorized Task 10 evaluation, and Task 11 has mechanically selected either SHADOW or EXPERIMENTAL_ACTIVE.

`EXPERIMENTAL_ACTIVE` is not the required outcome. Correct measurement, meaningful quality improvement, safe fallback, small deployable artifacts, and an honest gate verdict are the required outcomes.
