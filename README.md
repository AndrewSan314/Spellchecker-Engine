# Vietnamese SMS Content Validation Engine

POC kiểm tra nội dung SMS tiếng Việt, chạy local bằng Node.js và không gọi API AI bên ngoài. Engine kết hợp rule deterministic với SymSpell, Telex và mô hình ngôn ngữ n-gram để phát hiện lỗi định dạng, thiếu dấu và lỗi chính tả theo ngữ cảnh.

> Trạng thái hiện tại: rule-based đã ổn định; candidate oracle 93.5% và SMS p95 ~3ms đạt gate kỹ thuật. Các recall lane (wrong-diacritic / real-word / word-boundary) đã được xây, huấn luyện model pairwise leakage-safe và calibration đầy đủ nhưng KHÔNG lane nào vượt đủ ràng buộc precision+recall để ACTIVE — recall spelling semantic trên dev/held-out ~0.165-0.169, chưa đạt mục tiêu 0.35 của kế hoạch.

## Tính năng

- Giữ chính xác offset UTF-16 và không thay đổi nội dung đầu vào.
- Kiểm tra whitespace, punctuation, Unicode/control character.
- Bảo vệ URL, email, số điện thoại, ngày giờ, số tiền, product code và placeholder.
- Hỗ trợ `ACCENTED` và `NON_ACCENTED`.
- Nhận biết thiếu dấu bằng accent-family, beam search và n-gram context.
- Sinh spelling candidate bằng SymSpell-style index, Damerau-OSA và Telex.
- Cheap-rank trước context reranking để giới hạn chi phí.
- Whitelist và abbreviation snapshot; abbreviation có phạm vi Brandname.
- Linguistic mode: `OFF`, `SHADOW`, `ACTIVE`.
- Demo UI, REST API, benchmark, profiler và pipeline chống data leakage.

## Kiến trúc

```text
SMS request
  -> normalize + tokenize (giữ offset gốc)
  -> protected ranges
  -> deterministic rules
  -> whitelist / abbreviation / dictionary
  -> accent-family / Telex / SymSpell candidates
  -> cheap ranker
  -> n-gram context reranker
  -> confidence gates
  -> suppression + conflict resolution
  -> ValidationResult
```

N-gram không lưu toàn bộ corpus. Corpus chỉ dùng khi build; artifact runtime lưu vocabulary và count unigram, bigram, trigram.

| Thành phần | File chính |
|---|---|
| Core contract và result | `src/core.mjs` |
| Normalization/tokenization | `src/normalizer.mjs`, `src/tokenizer.mjs` |
| Protected ranges | `src/protected-ranges.mjs` |
| Rule deterministic | `src/rules/*.mjs` |
| Dictionary/whitelist/abbreviation | `src/lexical.mjs` |
| N-gram, beam decoder, SymSpell | `src/language.mjs` |
| Missing-diacritic và spelling | `src/rules/linguistic-rules.mjs` |
| Pipeline và conflict resolution | `src/engine.mjs` |
| Config snapshot | `src/config.mjs` |
| HTTP server và UI | `src/server.mjs`, `public/` |

## Yêu cầu

- Node.js 18 trở lên.
- Python 3 cho pipeline dữ liệu.

## Chạy nhanh

```powershell
cd "F:\AI\Tendoo Marketing\sms-validation-demo"
npm.cmd start
```

Mở `http://localhost:3000`. Đổi port bằng `$env:PORT = 3111` trước khi chạy. Runtime hiện load TSV vào JavaScript `Map`, nên cold start khoảng 4–6 giây và RSS khoảng 800–900 MB. Binary word-ID backend đang trong lộ trình deployment.

## REST API

```http
POST /api/v1/sms/content/validate
Content-Type: application/json

{
  "content": "Kính chao quý khách !",
  "messageMode": "ACCENTED",
  "brandname": "VT_TENDOO",
  "customerId": null
}
```

Response chứa `{ valid, hasErrors, hasWarnings, issues, tookMs }`. Mỗi issue có `ruleId`, `severity`, `start`, `end`, `value`, `message`, `suggestions`, `confidence`. `start` inclusive, `end` exclusive và luôn thỏa `content.substring(start, end) === value`.

Client không điều khiển confidence/margin; payload `options` bị bỏ qua vì server config là nguồn sự thật duy nhất.

| Method | Endpoint | Mục đích |
|---|---|---|
| `POST` | `/api/v1/sms/content/validate` | Kiểm tra SMS |
| `POST` | `/api/v1/sms/content/benchmark` | Benchmark nội bộ |
| `GET` | `/api/v1/config` | Xem runtime config |

Hai endpoint sau chỉ dành cho demo, không nên public trong production.

## Dữ liệu lexical

| Loại | Artifact | Vai trò |
|---|---|---|
| Dictionary | `src/data/lexicon-built.txt`, `lexicon.txt` | Từ hợp lệ, tần suất, candidate |
| Stop-lexicon | `src/data/stoplex.txt` | Giữ entry trong spelling path |
| Whitelist | `src/data/whitelist.txt` | Bảo vệ thuật ngữ/tên riêng |
| Abbreviation | `src/data/abbreviations.json` | Whole-token và phạm vi Brandname |
| N-gram | `src/data/lm-ngrams.tsv` | Count U/B/T cho context ranking |

Brandname exception qua giao diện quản trị chưa được implement trong POC. Thiết kế production dự kiến:

```text
Admin UI -> database (DRAFT/APPROVED/PUBLISHED)
         -> validate + versioned snapshot
         -> engine atomic reload
         -> rollback snapshot cũ nếu load lỗi
```

Không truy vấn database theo từng token. Dictionary/LM build offline; whitelist, abbreviation và Brandname exception publish thành snapshot có version, checksum và audit log.

## Language model và dataset

Artifact hiện tại:

- U = 200.000, B = 1.000.000, T = 2.760.041.
- Weighted tokens = 1.325.600.394.
- Kích thước khoảng 79,9 MB.
- SHA-256: `791f4525653489ef98dae1ed23e238ab4aac4af90df2f8a4378e09ae78560ce6`.

Nguồn, train/held-out role và hash được ghi trong `src/data/lm-ngrams.manifest.json`.

### Rebuild và audit LM

```powershell
python tools/build_lm.py --news-shards 4 --unigram-news-shards 0 `
  --domain-weight 5 --retain-accent-evidence `
  --max-bigrams 1000000 --max-trigrams 500000 `
  --max-lm-bytes 100000000
python tools/rebuild_lm_higher.py --news-shards 4 `
  --domain-overlay-weight 100 --max-trigrams 500000 `
  --max-lm-bytes 100000000
python tools/audit_lm.py
```

### Dataset split và leakage audit

```powershell
python tools/split_spelling_datasets.py
python tools/split_spelling_datasets.py --audit `
  --external-benchmark benchmark/corpus-viwiki-spelling.json
node tools/spelling_benchmark_adapter.mjs test
node tools/corpus_pipeline.mjs
npm.cmd run data:audit
```

Không dùng VSEC dev/test hoặc Viwiki external-test để build vocab/count; không tune trên held-out test; raw downloads không bị pipeline sửa; evaluation artifact phải ghi source/config/artifact hash.

## Test và đánh giá

### Regression

```powershell
npm.cmd test
```

Suite mặc định: 31 core tests + 35 rule tests. Các suite spelling riêng:

```powershell
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
  test/test_recall_calibration.mjs `
  test/test_word_boundary_candidates.mjs
python -m unittest test.test_recall_reranker_training -v
```

### Recall lanes (recall-improvement plan 2026-08-24)

Trạng thái lane sau calibration freeze (`config/spelling-tuning.json` v3):

| Lane | Mode | Ghi chú |
|---|---|---|
| Wrong-diacritic (ACCENTED_SAME_KEY) | OFF | Model pairwise không đạt precision floor 0.70 trên dev (max ~0.67) |
| Real-word typo (DIFFERENT_KEY_REAL_WORD) | ACTIVE | Full-grid calibration: p>=0.95, original windows<=3; lane precision 0.8908 trên replay dev |
| Word-boundary (split/merge) | SHADOW | Tính toán để đo lường, không phát issue; TP 0 / FP 1 trên dev |

Real-word lane được ACTIVE sau khi sửa lỗi calibration v2 chỉ verify ba trial
đầu và bỏ sót nhóm p>=0.95. Chi tiết từng trial:
`dataset_artifacts/evaluation/recall-calibration.json`.

Model pairwise `recall-pairwise-v1` (`src/data/recall-reranker.json`, huấn
luện chỉ từ VSEC train + clean-train negatives, leakage-guard fail-closed)
được inject một lần duy nhất qua `SmsValidationEngine` và chạy kèm mọi shadow
decision.

### Kết quả cuối (frozen configuration)

Dev (VSEC dev, 1.115 labels — `final-recall-dev-report.json`):

| View | Precision | Recall | F0.5 |
|---|---:|---:|---:|
| strictRuleId | 0.7797 | 0.1650 | 0.4468 |
| semanticLinguistic (primary product score) | 0.6502 | 0.2717 | 0.5086 |

Held-out dưới cấu hình v2 trước fix — đã chạy ĐÚNG MỘT LẦN (VSEC test,
1.121 labels — `final-recall-heldout.json`); không chạy lại để tránh biến
held-out thành tập tuning:

| View | Precision | Recall | F0.5 |
|---|---:|---:|---:|
| strictRuleId | 0.9118 | 0.0553 | 0.2225 |
| semanticLinguistic | 0.6469 | 0.1650 | 0.4084 |

Chưa có held-out measurement cho cấu hình v3. Cấu hình v3 chỉ được chọn từ
dev, clean guard, missing-diacritic guard và latency gate; cần một tập test
mới độc lập nếu muốn đánh giá generalization tiếp theo.

Hiệu năng (`profile:engine`, frozen state): SMS ≤160 ký tự p95 **12,79 ms**
(gate ≤15 ms), steady RSS ~853 MB.

### Benchmark tổng

```powershell
npm.cmd run bench
```

Kết quả ghi vào `benchmark/results.json`. Snapshot frozen-state:

| Metric | Kết quả |
|---|---:|
| Expected issues | 2.949 |
| Caught | 1.646 |
| False positives | 28 |
| Overall error recall | 55,8% |
| Precision | 98,3% |

Precision proxy chỉ phản ánh corpus có nhãn/forbid/clean hiện tại, không phải cam kết precision production.

### Candidate và spelling evaluation

```powershell
node tools/audit_candidate_recall.mjs --split dev
npm.cmd run eval:spelling:dev
# Chỉ chạy held-out test sau khi freeze config:
npm.cmd run eval:spelling:test
```

- Candidate oracle coverage: 93,5%, vượt gate 92%.
- Strict rule-ID dev: P=0,7797; R=0,1650; F0.5=0,4468.
- Cross-lane linguistic dev: P=0,6502; R=0,2717; F0.5=0,5086.

Khoảng 76% nhãn VSEC bị prefilter khỏi typo lane vì cùng stripped key và được thiết kế cho missing-diacritic lane. Không nên dùng strict `POSSIBLE_SPELLING_ERROR` recall làm metric duy nhất.

### Profiler

```powershell
npm.cmd run profile:engine
```

Task 6 đã giảm `scoreCandidateOverSurfaces` khoảng 5,6 lần và `_pBiRaw` khoảng 11 lần so với plan-era baseline. P95 SMS <=160 ký tự dưới gate 15 ms; startup/RSS vẫn bị chi phối bởi backend TSV.

## Scripts

| Command | Chức năng |
|---|---|
| `npm.cmd start` | Server + demo UI |
| `npm.cmd test` | Core + rule regression |
| `npm.cmd run bench` | Benchmark tổng |
| `npm.cmd run data:audit` | Split/leakage audit |
| `npm.cmd run eval:spelling:dev` | Dev spelling evaluation |
| `npm.cmd run eval:spelling:test` | Guarded final test |
| `npm.cmd run profile:engine` | Startup/RSS/latency/hotspot |

## Cấu trúc thư mục

```text
src/                  runtime engine, server, rules và data
public/               demo UI
benchmark/            labeled corpora và results
dataset_raw/          raw downloaded datasets
dataset_artifacts/    split-safe/generated evaluation artifacts
tools/                build, audit, evaluation, profiling
test/                 Node/Python tests
docs/                 design, implementation plan, execution log
config/               frozen spelling tuning state
```

## Tiến độ và roadmap

Đã hoàn thành Task 1–9 của spelling optimization plan: Viwiki converter, fail-fast LM loader, trigram formulation, Telex/candidate cascade, oracle audit, context hotspot, length-aware SymSpell, split-safe evaluation và dev-only tuning. Default config vẫn là winner của quá trình tuning.

Việc còn lại:

- Experiment Modified Kneser-Ney độc lập.
- Versioned binary LM và runtime word-ID backend.
- Giảm cold-start/RSS và bỏ full TSV parsing khi deploy.
- Freeze final config rồi mới chạy held-out test cuối.
- Chỉ cân nhắc attention reranker nhẹ nếu classical pipeline vẫn không đạt recall sau khi xử lý lane/ranking.

Chi tiết tại `docs/plans/2026-08-24-spelling-engine-optimization.md` và `docs/plans/execution-log-spelling-optimization.md`.

## Lưu ý production

- Đây là POC, chưa phải service đã harden.
- Phải kiểm tra license dataset/dictionary trước khi phân phối.
- Không public benchmark/config endpoint.
- Không tự động học whitelist/exception từ nội dung người dùng.
- Rule deterministic đáng tin cậy hơn statistical spelling hiện tại.
- Không nới rule mù quáng để tăng recall; mọi thay đổi phải qua dev evaluation và precision constraints.
