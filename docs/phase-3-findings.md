# Phase 3 — multi-block attention runtime: result

## Status: code landed, arch C **not promoted** (fails the latency gate)

### What shipped
`src/attention-reranker.mjs` now loops `for (blk = 0; blk < numBlocks; blk++)`
over `encoder.blocks.${blk}.*` instead of hard-coding `encoder.blocks.0.*`.
Block count is read from the tensor table (`countEncoderBlocks`) and
cross-checked against `config.blocks`; a mismatch throws at load.

Non-regression for the shipped arch A path is verified:
- 248 tests / 0 fail (`node --test "test/*.mjs"`)
- `split.mjs` byte-identical: R=0.252, PMD 99/48, SPE 180/84
- `clean.mjs`: 0/165 false alarms

### Arch C parity: PASS (exact)
| | tolerance | measured |
|---|---|---|
| max logit diff | 2e-3 | **5.96e-7** |
| max prob diff | 2e-3 | **1.19e-7** |
| argmax mismatches | 0 | **0** / 25 cases |

Fixture is PyTorch-generated (`tools/export_attention_parity_fixture.py`) and
its `binHash` matches the loaded binary, so this is genuine cross-implementation
parity, not a self-comparison. **The multi-block runtime is correct.**

### Arch C latency: FAIL — 10x over gate
300 SMS-length rows, attention `EXPERIMENTAL_ACTIVE`:

| arm | p50 | p95 | p99 |
|---|---|---|---|
| classical (attention OFF, **shipped**) | 5.41ms | **11.27ms** | 13.49ms |
| arch A ACTIVE | 33.19ms | 78.11ms | 93.29ms |
| arch C ACTIVE | 117.30ms | **212.38ms** | 234.89ms |

Gate is p95 <= 20ms. Arch C misses by 10x — but note **arch A already misses
by 4x**. This is not a regression introduced by Phase 3.

Root cause is call volume, not per-call cost:

| | calls/msg | per call | share of runtime |
|---|---|---|---|
| arch A | 11.1 | 1.42ms | 44.8% |
| arch C | 11.1 | 8.71ms | 81.6% |

Arch C is 6.1x the per-call cost of arch A (2 blocks x d96 vs 1 x d48 is ~6x
the FLOPs — consistent). The engine invokes the reranker **11.1 times per
message**, once per candidate token, with no batching and no caching.

### Decision
Keep arch A. Do **not** promote arch C into `src/data/`.

The plan already anticipated this: *"Arch C có thể không dịch thành lợi ích
end-to-end... Nếu Phase 3 xong mà end-to-end không nhúc nhích, giữ arch A và
dồn sức vào Phase 2."* The stronger finding is that the question is currently
moot — **`spelling.attentionMode` ships as `OFF`**, so neither arch is on the
serving path. Arch C's shortlist-level F0.5 advantage (0.9535 vs 0.9370) cannot
be cashed until the attention path is both enabled (Phase 4) and affordable.

The multi-block loop is kept: it is correct, costs nothing when `numBlocks === 1`,
and removes the blocker for any future multi-block model.

### If arch C is wanted later, the work is batching — not the model
11.1 calls/msg x 8.71ms is the whole problem. Batching all candidate tokens of a
message into one forward pass, or caching by (token, context) key, is where the
order of magnitude is. That is a separate piece of work from Phase 3's scope.
