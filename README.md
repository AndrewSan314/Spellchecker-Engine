# Vietnamese SMS Content Validation Engine

POC kiểm tra nội dung SMS tiếng Việt, chạy local bằng Node.js và không gọi API AI bên ngoài. Engine kết hợp rule deterministic với SymSpell, Telex và mô hình ngôn ngữ n-gram để phát hiện lỗi định dạng, thiếu dấu và lỗi chính tả theo ngữ cảnh.

> Trạng thái hiện tại (cập nhật 2026-09-03, sau đợt review `spellchecker-engine-review.txt`):
> lớp HTTP đã vá 4 lỗi cấp production; rule deterministic ổn định và đáng tin;
> có thêm **profile `lite` để chạy local** (khởi động 0,7 s, RSS ~250 MB thay vì 5,8 s /
> ~860 MB) và **bộ dữ liệu SMS tiếng Việt trong repo** (`dataset_sms/`) để huấn luyện +
> đánh giá đúng miền. Trên miền SMS held-out: recall 64,8% · precision 93,9%.
> Trên VSEC, điểm số vẫn được nâng đỡ bởi bảng tra sinh từ tập train — xem
> [Rủi ro phương pháp](#rủi-ro-phương-pháp-đọc-trước-khi-tin-số-liệu).
> Đây là POC nghiên cứu, chưa phải service production.

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
- Cảnh báo cấp tin nhắn khi nội dung trông như gõ không dấu (`summary.unaccentedContent`),
  thay vì bắn cảnh báo theo từng token.
- Demo UI, REST API, benchmark, profiler và pipeline chống data leakage.
- Hai profile dữ liệu: `full` (nghiên cứu) và `lite` (SMS-domain, nhẹ cho máy local).
- Bộ dữ liệu SMS tiếng Việt sinh trong repo + công cụ train/eval theo miền.
- CLI `npm run check` để thử nhanh một tin nhắn, không cần bật server.
- CI (`.github/workflows/ci.yml`): Node 20/22 + Python tools + benchmark drift guard.

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

- Node.js 18 trở lên (CI chạy 20 và 22). Engine **không có dependency npm**.
- [`uv`](https://docs.astral.sh/uv/) + Python 3.10+ cho pipeline dữ liệu trong `tools/`.
  Dependency khai báo trong `pyproject.toml` (`numpy`, `pyarrow`, `torch`); cài bằng
  `uv sync`. Serving engine không cần Python.

## Chạy nhanh

Không có dependency npm — engine chỉ dùng `node:http` / `node:fs`.

```bash
git clone https://github.com/AndrewSan314/Spellchecker-Engine.git
cd Spellchecker-Engine

npm start                    # profile lite: sẵn sàng sau ~0,7 s -> http://localhost:3000
npm run start:full           # profile full: ~5,8 s, dùng artifact nghiên cứu đầy đủ
PORT=3111 npm start          # đổi port
```

Thử nhanh một tin nhắn, không cần server:

```bash
npm run check -- "Ma OTP cua quy khach la 123456, hieu luc 5 phut."
npm run check                                  # chế độ gõ từng dòng
npm run check -- --json "Kinh chao quy khach"  # output máy đọc
echo "Cam on quy khach" | npm run check
```

```
WARNINGS  8 issue(s)  11.7 ms
  Ma OTP cua quy khach la 123456, hieu luc 5 phut. Khong chia se ma nay.
  ^^ POSSIBLE_MISSING_DIACRITIC
         ^^^ POSSIBLE_MISSING_DIACRITIC
  ...
  ⚠ nội dung có vẻ được gõ KHÔNG DẤU trong chế độ ACCENTED (8 cảnh báo gộp lại thành một)
```

Trên Windows PowerShell:

```powershell
$env:ENGINE_PROFILE = "lite"; node src\server.mjs
$env:PORT = 3111
```

### Hai profile dữ liệu (`ENGINE_PROFILE`)

Chi phí của engine nằm ở **artifact dữ liệu**, không phải ở code: LM 80 MB nạp vào
`Map` chuỗi, cộng chỉ mục xoá SymSpell dựng từ 51k từ. Profile `lite` dùng cặp artifact
đã cắt theo miền SMS (`tools/build_sms_profile.mjs`), giữ **nguyên count thật** cho những
từ còn lại — `pUni` chia cho `totalTokens + 0,5·V` với `totalTokens ≈ 1,3·10⁹`, nên bỏ
vocabulary từ 200k xuống 20k làm xác suất đổi <0,01% và mọi ngưỡng đã calibrate vẫn còn ý
nghĩa.

| | `full` (mặc định của engine) | `lite` (mặc định của `npm start`) |
|---|---:|---:|
| LM | `lm-ngrams.tsv` 80 MB · U 200k / B 1M / T 2,76M | `lm-ngrams.sms.tsv` 10,5 MB · U 20,5k / B 321k / T 301k |
| Lexicon | `lexicon-built.txt` 51.151 từ | `lexicon-sms.txt` 20.503 từ |
| Cold start | 5,8 s | **1,3 s** |
| RSS sau khi nạp | 857 MB | **299 MB** |
| SMS **dev**: recall / precision | 0,713 / 0,970 | **0,739 / 0,976** |
| SMS **test** (held-out): recall / precision | **0,652** / 0,926 | 0,594 / **0,937** |

Đọc bảng này cho đúng: trên **dev**, `lite` nhỉnh hơn cả hai chiều — nhưng tham số cắt được
chọn trên dev, và trên **test** thì `lite` thấp hơn `full` 5,8 pp recall (đổi lại precision cao
hơn 1,1 pp và báo động giả 0/32). Nói thẳng: cái được của `lite` trên dev một phần là hiệu ứng
chọn tham số, không phải cải thiện thật. Điều chắc chắn là **đánh đổi tài nguyên**: 3× ít RAM và
4× khởi động nhanh, trả bằng vài điểm recall. Cần recall tối đa thì `npm run start:full`.

Xây lại artifact `lite` (chỉ cần khi đổi dataset hoặc tham số):

```bash
npm run sms:dataset    # sinh dataset_sms/ (deterministic)
npm run sms:profile    # sinh src/data/lm-ngrams.sms.tsv + lexicon-sms.txt
```

Biến môi trường:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `ENGINE_PROFILE` | `full` (`npm start` đặt `lite`) | `full` \| `lite` — chọn cặp artifact LM + lexicon |
| `LM_ARTIFACT` / `LEXICON_ARTIFACT` | — | Ghi đè đường dẫn artifact (dùng cho tooling) |
| `PORT` | `3000` | Cổng HTTP |
| `MAX_BODY_BYTES` | `16384` | Giới hạn body request (byte). Vượt → `413` và đóng socket |
| `DEMO_ENDPOINTS` | *(tắt)* | `=1` mới bật `/api/v1/sms/content/benchmark` và `/api/v1/config` |
| `SPELLING_TUNING_FILE` | `config/spelling-tuning.json` | File calibration mà runtime nạp lúc khởi động |

Profile `lite` mà thiếu artifact sẽ **cảnh báo một lần rồi tự quay về `full`** — không bao
giờ im lặng, cũng không crash trên clone sạch. `GET /healthz` trả về profile đang chạy.

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

Response chứa `{ valid, hasErrors, hasWarnings, issues, summary, tookMs }`. Mỗi issue có
`ruleId`, `severity`, `start`, `end`, `value`, `message`, `suggestions`, `confidence`.
`start` inclusive, `end` exclusive và luôn thỏa `content.substring(start, end) === value`.

`confidence` trên dây là giá trị **đã làm tròn 2 chữ số để hiển thị**; engine giữ giá trị
thô cho mọi so sánh ngưỡng nội bộ (trước đây engine so sánh chính giá trị đã làm tròn nên
0,9550–0,9599 bị nâng lên 0,96 và lọt cửa — review B3).

`summary` là quan sát cấp tin nhắn, không thay thế `issues`:

```json
"summary": { "wordCount": 17, "linguisticIssueCount": 11,
             "unaccentedWordRatio": 1, "unaccentedContent": true }
```

`unaccentedContent = true` nghĩa là nội dung trông như gõ không dấu trong khi
`messageMode = ACCENTED`. UI nên hiển thị MỘT cảnh báo cấp tin nhắn thay vì bắn cảnh báo
theo từng token (một câu 15 từ không dấu sinh 11–12 issue). Ngưỡng: `linguistic.unaccentedRatioThreshold`
(0,8) và `linguistic.unaccentedMinWords` (5).

| Method | Endpoint | Mục đích | Mặc định |
|---|---|---|---|
| `POST` | `/api/v1/sms/content/validate` | Kiểm tra SMS | bật |
| `GET` | `/healthz` | Liveness + đếm rule OPTIONAL bị degraded | bật |
| `POST` | `/api/v1/sms/content/benchmark` | Benchmark nội bộ (chạy đồng bộ trên 1.439 dòng corpus) | **tắt** |
| `GET` | `/api/v1/config` | Xem runtime config + nguồn calibration | **tắt** |

Hai endpoint demo chỉ bật khi `DEMO_ENDPOINTS=1`; mặc định trả `404`. Trước đây chúng luôn
mở và `/benchmark` có thể treo server hàng chục giây mà không cần auth (review A4).

Client không điều khiển confidence/margin; payload `options` bị bỏ qua vì server config là
nguồn sự thật duy nhất.

### Hardening lớp HTTP (review A1–A4)

| Mã | Vấn đề cũ | Cách sửa |
|---|---|---|
| A1 | `new URL(req.url, http://${req.headers.host})` nằm ngoài try/catch — một request với `Host: bad host` giết cả process (DoS 1 packet) | Toàn bộ handler nằm trong try/catch + fallback 500; path parse thẳng từ `req.url`, không cần Host |
| A2 | `data += chunk` gọi `toString('utf8')` cho từng chunk → ký tự tiếng Việt bị cắt giữa chuỗi UTF-8 nhiều byte thành `U+FFFD` | Gom `Buffer` rồi `Buffer.concat(...).toString('utf8')` một lần |
| A3 | Giới hạn 1 MB đếm theo ký tự, reject promise nhưng vẫn nhận tiếp dữ liệu, trả 400 | Đếm **byte**, mặc định 16 KB, trả `413` rồi mới destroy socket |
| A4 | `/benchmark` và `/config` luôn mở | Sau cờ `DEMO_ENDPOINTS=1`, mặc định 404 |

Test tương ứng: `test/test_server_hardening.mjs` (gửi request thô qua socket, bao gồm
trường hợp cắt body giữa chữ "í").

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

```bash
uv run python tools/build_lm.py --news-shards 4 --unigram-news-shards 0 \
  --domain-weight 5 --retain-accent-evidence \
  --max-bigrams 1000000 --max-trigrams 500000 \
  --max-lm-bytes 100000000
uv run python tools/rebuild_lm_higher.py --news-shards 4 \
  --domain-overlay-weight 100 --max-trigrams 500000 \
  --max-lm-bytes 100000000
uv run python tools/audit_lm.py
```

### Dataset split và leakage audit

```bash
uv run python tools/split_spelling_datasets.py
uv run python tools/split_spelling_datasets.py --audit \
  --external-benchmark benchmark/corpus-viwiki-spelling.json
node tools/spelling_benchmark_adapter.mjs test
node tools/corpus_pipeline.mjs
npm run data:audit
```

Không dùng VSEC dev/test hoặc Viwiki external-test để build vocab/count; không tune trên held-out test; raw downloads không bị pipeline sửa; evaluation artifact phải ghi source/config/artifact hash.

## Test và đánh giá

### Regression

```bash
npm test          # TOÀN BỘ node test (~35 file, ~2 phút)
npm run test:fast # chỉ core + rules, dùng khi lặp nhanh
npm run test:python   # uv run python -m unittest discover -s test -t .
npm run test:all      # cả hai
```

Trước đây `npm test` chỉ chạy 2 file (66 test) trong khi repo có hơn 30 file `.mjs` và
toàn bộ test Python — nghĩa là CI mặc định không bảo vệ phần lớn code (review E3). Hiện
`npm test` chạy tất cả và `.github/workflows/ci.yml` chạy Node (20/22) + Python + benchmark
trên mỗi PR.

Trạng thái hiện tại: **276 node test — 271 pass, 0 fail, 5 skip** và **62 python test pass**
(trước review: 244 pass / 1 fail / 4 skip, và 38 python test với 6 lỗi import).
Các test skip đều là skip có điều kiện/có ghi chú (fixture `.tmp/` không có trong clone
sạch — review E1 — và 2 "Phase 1 known regression").

Test mới bổ sung theo review:

| File | Bảo vệ |
|---|---|
| `test/test_server_hardening.mjs` | A1–A4 + B3 (gửi request thô qua socket) |
| `test/test_frozen_tuning_config.mjs` | B9 — config frozen phải đúng bằng config đang chạy |
| `test/test_review_fixes.mjs` | B1, B2, B3, B5, B7, B8, cảnh báo cấp tin nhắn |

### Recall lanes (recall-improvement plan 2026-08-24)

Trạng thái lane **đang chạy thật** (đọc từ `config/spelling-tuning.json`, xác thực bằng
`test/test_frozen_tuning_config.mjs`):

| Lane | Mode | Ghi chú |
|---|---|---|
| Missing-diacritic (UNACCENTED_SAME_KEY) | ACTIVE | Lane chính, chiếm phần lớn issue phát ra |
| Wrong-diacritic (ACCENTED_SAME_KEY) | **ACTIVE** | Bật từ breakthrough 2026-08-27 (dev R 0,252 → 0,421; P 0,679 → 0,773) |
| Real-word typo (DIFFERENT_KEY_REAL_WORD) | ACTIVE | p>=0,97, original windows<=3 |
| Word-boundary (split/merge) | SHADOW | Không phát issue; ruleId riêng `POSSIBLE_WORD_BOUNDARY_ERROR` khi ACTIVE |
| Attention reranker | OFF khi phục vụ | File calibration ghi SHADOW; xem `servingOverrides` bên dưới |

Bảng lane trong README cũ ghi wrong-diacritic = OFF trong khi code đặt `ACTIVE` (review D2).

### Config: cái đã "đóng băng" == cái đang phục vụ

`config/spelling-tuning.json` tự nhận `"frozen": true` nhưng **không file nào trong `src/`
đọc nó** — runtime lấy hằng số từ `DEFAULT_SNAPSHOT`, và hai bên đã lệch nhau ở nhóm key
attention (`attentionMode` SHADOW vs OFF; `attentionMinProbability` 0,5 vs fallback 0,80).
Mọi con số đánh giá vì thế không chứng minh được là đo trên cấu hình đang chạy (review B9).

Nay `src/config.mjs` nạp file này lúc khởi động:

- mọi key trong `frozenParams` và `confidenceCalibration.constants` được áp vào snapshot;
- key lạ / JSON hỏng → **throw lúc khởi động**, không im lặng bỏ qua;
- khác biệt cố ý giữa "winner của calibration" và "cái phục vụ" phải khai báo trong
  `servingOverrides` kèm lý do — hiện chỉ có một mục: `attentionMode = OFF` (nhánh attention
  không nằm trên serving path, arch A vượt gate latency 4×, arch C 10×; SHADOW không đổi
  issue nào mà chỉ thêm ~11 forward pass/tin nhắn);
- `GET /api/v1/config` (khi bật `DEMO_ENDPOINTS=1`) trả kèm `tuningSource` để đối chiếu.

Model pairwise `recall-pairwise-v1` (`src/data/recall-reranker.json`, huấn luyện chỉ từ
VSEC train + clean-train negatives, leakage-guard fail-closed) được inject một lần duy nhất
qua `SmsValidationEngine` và chạy kèm mọi shadow decision.

### Kết quả cuối (frozen configuration)

Dev (VSEC dev, 1.115 labels — `final-recall-dev-report.json`):

| View | Precision | Recall | F0.5 |
|---|---:|---:|---:|
| strictRuleId | 0.7797 | 0.1650 | 0.4468 |
| semanticLinguistic (primary product score) | 0.6502 | 0.2717 | 0.5086 |

Held-out (VSEC test) đã được chạy **nhiều hơn một lần**: một lần dưới cấu hình v2
(`final-recall-heldout.json`) và một lần nữa cho wrong-diacritic breakthrough
(`config/spelling-tuning.json → wrongDiacriticBreakthrough.heldOutTestMetrics`,
R 0,24 → 0,397; P 0,695 → 0,764). README cũ khẳng định "đã chạy ĐÚNG MỘT LẦN" và "chưa có
held-out measurement cho cấu hình v3" — cả hai đều sai (review C2/D3):

| Lần chạy | View | Precision | Recall | F0.5 |
|---|---|---:|---:|---:|
| v2, trước fix | strictRuleId | 0.9118 | 0.0553 | 0.2225 |
| v2, trước fix | semanticLinguistic | 0.6469 | 0.1650 | 0.4084 |
| wrong-diacritic, sau | (lane view) | 0.764 | 0.397 | 0.645 |

Vì tập test đã tham gia vòng lặp quyết định, **nó không còn là tập generalization không
thiên lệch**. Cần một tập test mới, khoá lại và chạy đúng một lần.

Hiệu năng (`npm run profile:engine`, máy review 2026-09-03, Node v26.3.0):

| Đo | Giá trị |
|---|---:|
| Cold start (load LM TSV) | ~6,0 s |
| RSS sau load / steady | 831 MB / 1.065 MB |
| SMS ≤160 ký tự p95 (corpus benchmark 1.439 dòng) | 27,97 ms |
| 6 mẫu SMS marketing ngắn, 720 lần gọi | p50 2,39 ms · p95 10,20 ms |

Các tài liệu cũ ghi 12,79 ms / 18,2 ms / 24,04 ms / 11,27 ms cho cùng khái niệm "p95"
(review D5). Con số phụ thuộc mạnh vào máy và vào phân bố độ dài của corpus, nên gate
latency chỉ có nghĩa khi so cùng máy + cùng corpus; đừng so hai dòng khác corpus với nhau.

### Benchmark tổng

```bash
npm run bench
```

Kết quả ghi vào `benchmark/results.json` (file này được git track — chạy bench sẽ làm bẩn
working tree). Số liệu đo lại trên HEAD ngày 2026-09-03:

| Metric | README cũ (snapshot commit cũ) | HEAD hiện tại |
|---|---:|---:|
| Expected issues | 2.949 | 2.949 |
| Caught | 1.646 | 1.827 |
| False positives | 28 | 59 |
| Overall error recall | 55,8% | **62,0%** |
| Precision proxy | 98,3% | **96,9%** |

README cũ là snapshot của một commit trước `b925c98` và chưa bao giờ được cập nhật
(review D1). CI nay chạy `npm run bench` mỗi PR và upload `benchmark/results.json` để
drift không âm thầm nữa.

Ảnh hưởng của các fix trong đợt review này (đo bằng A/B trên cùng corpus):

- B3 (không dùng confidence đã làm tròn làm ngưỡng): caught 1.800 → 1.790. 10 issue từng
  lọt cửa nhờ 0,955x được làm tròn lên 0,96 — recall giảm 0,3 pp, đổi lấy một ngưỡng đúng
  nghĩa.
- B8 (gán đúng ruleId cho nhánh error-channel): caught không đổi, false positive 61 → 59.

Precision proxy chỉ phản ánh corpus có nhãn/forbid/clean hiện tại, không phải cam kết
precision production.

### Candidate và spelling evaluation

```bash
node tools/audit_candidate_recall.mjs --split dev
npm run eval:spelling:dev
# Chỉ chạy held-out test sau khi freeze config — và lưu ý held-out đã bị dùng
# lại nhiều lần (review C2), xem "Rủi ro phương pháp" bên dưới:
npm run eval:spelling:test
```

- Candidate oracle coverage: 93,5%, vượt gate 92%.
- Strict rule-ID dev: P=0,7797; R=0,1650; F0.5=0,4468.
- Cross-lane linguistic dev: P=0,6502; R=0,2717; F0.5=0,5086.

Khoảng 76% nhãn VSEC bị prefilter khỏi typo lane vì cùng stripped key và được thiết kế cho missing-diacritic lane. Không nên dùng strict `POSSIBLE_SPELLING_ERROR` recall làm metric duy nhất.

### Bộ dữ liệu SMS trong repo và kết quả trên miền SMS

`dataset_sms/` là bộ SMS brandname tiếng Việt **viết tay trong repo** (62 template / 35
group / 8 miền), sinh ra bằng `npm run sms:dataset`. Chi tiết: `dataset_sms/README.md`.

Vì sao cần: mọi số liệu trước đây đo trên VSEC — văn xuôi kiểu Wikipedia, sai miền, và
~68% nhãn trả lời được từ bảng tra sinh từ chính tập train của nó. Bộ SMS này đúng miền và
**bảng tra error-channel không biết gì về nó**, nên recall đo được là recall thật của LM +
rule.

Chống leakage được thiết kế sẵn: split chia theo **group template** (`sha256(group)%100 →
70/15/15`), nên không cách diễn đạt nào xuất hiện ở hai split; chỉ `sms-clean-train.txt`
được dùng để build artifact, và có test khẳng định điều đó.

```bash
npm run sms:eval          # dev, profile lite
npm run sms:eval:full     # dev, artifact đầy đủ
```

**Kết quả (2026-09-04).** Dev dùng để chọn tham số; test được khai báo trong
`config/acceptance-gates.json` trước khi chạy:

| Split | Profile | Strict recall | Semantic recall | Semantic precision | Tin sạch bị báo nhầm |
|---|---|---:|---:|---:|---:|
| dev (174 tin, 713 nhãn) | lite | 0,749 | 0,739 | 0,976 | 0/30 |
| dev | full | 0,720 | 0,713 | 0,970 | 1/30 |
| **test (138 tin, 500 nhãn)** | lite | 0,594 | 0,594 | 0,937 | 0/32 |
| **test** | **full** | **0,651** | **0,652** | 0,926 | 0/32 |

Theo lớp lỗi (test, profile lite): TYPO 81,8% · TELEX 68,4% · SOME_UNACCENTED 65,2% ·
ALL_UNACCENTED 58,1% · WRONG_DIACRITIC 16,7% · BOUNDARY 0% (lane SHADOW, theo thiết kế).

**Phải đọc kèm ba cảnh báo:**

1. **Dev tăng nhưng test giảm với profile `lite`** (dev 0,684 → 0,739; test 0,648 → 0,594
   giữa hai lần đo). Tham số cắt LM được chọn trên dev, nên phần tăng đó có mùi chọn tham số
   hơn là cải thiện thật. Ngược lại, phần sửa **rule** thì chuyển giao được: profile `full`
   (artifact không đổi, chỉ đổi rule) đi từ 0,618 → 0,652 trên chính tập test này, và
   benchmark VSEC độc lập đi từ 60,7% → 62,0% recall trên 2.949 nhãn.
2. **Cỡ mẫu quá nhỏ để phân giải vài điểm phần trăm.** Test có 500 nhãn nhưng chỉ 138 tin, và
   377 nhãn trong số đó đến từ **28 tin** ALL_UNACCENTED. Đơn vị mẫu thật là *tin*, không phải
   *nhãn*. Chênh ±5 pp ở đây chưa phải tín hiệu.
3. **Tập test này đã dùng 2 lần** (lần 2 vì artifact và rule đều đổi sau khi sửa lỗi
   "ma nay"). Lần khẳng định tổng quát hoá tiếp theo cần một tập MỚI — sinh lại bằng seed
   khác hoặc gán nhãn SMS thật.

**Giới hạn:** dữ liệu tổng hợp từ template, không phải SMS thật của khách hàng. Nó không có
đuôi dài của thực tế (tên riêng lạ, teencode, trộn tiếng Anh, emoji), nên vẫn là cận trên
nhẹ. Bước tiếp theo vẫn là một tập SMS thật do team gán nhãn.

### Rủi ro phương pháp (đọc trước khi tin số liệu)

**Điểm số hiện tại được nâng đỡ bởi một bảng tra sinh từ tập train.**
`src/data/error-channel.json` (3.790 cặp `pairProof`, 1.588 entry `direct`) được sinh từ
`dataset_artifacts/vsec/vsec-train.jsonl`, và trên đường phục vụ `pairProven` là điều kiện
CHÍNH cho phép phát issue. Nếu phân bố lỗi của dev/test trùng train thì P/R đo được một
phần là đo trí nhớ, không phải khả năng tổng quát hoá.

Đo lại được bằng:

```bash
npm run eval:overlap        # node tools/measure_error_channel_overlap.mjs
```

Kết quả trên HEAD (2026-09-03):

| Split | Cặp (error→đúng) phân biệt cũng có trong train | Nhãn được bảng tra bao phủ |
|---|---:|---:|
| dev | 503/841 = 59,8% | 743/1.088 = **68,3%** |
| test | 489/852 = 57,4% | 723/1.092 = **66,2%** |

Split chia theo message nên không leak câu; cái leak là **phân bố lỗi**. Khoảng hai phần ba
nhãn dev/test có thể được trả lời thẳng từ bảng tra. Trên SMS marketing thật (miền khác,
kiểu gõ khác, tên riêng/brand khác), recall sẽ thấp hơn con số công bố ở đây (review C1).

Ba rủi ro còn lại, chưa sửa được bằng code:

- **C2 — held-out đã bị dùng lại.** Xem bảng ở mục "Kết quả cuối". Cần tập test mới, tốt
  nhất là SMS thật do team gán nhãn, khoá trước khi chạy và chạy đúng một lần.
- **C3 — gate trôi theo kết quả.** Gate latency từng xuất hiện với 3 giá trị (p95 ≤15 ms,
  ≤20 ms, "limit subsequently authorized at 100 ms") và precision floor dao động 0,70/0,90.
  Gate được nới sau khi biết kết quả thì không còn tác dụng kiểm soát. Nay gộp về một
  nguồn duy nhất: `config/acceptance-gates.json` (kèm danh sách giá trị lịch sử đã bị thay
  thế, luật đổi gate, và nhật ký các lần chạy held-out).
- **B4 — `pairCount !== 2` là số fit vào dev.** Bộ lọc verified loại đúng nhóm cặp xuất hiện
  chính xác 2 lần trong VSEC train vì nhóm đó xấu trên dev (48 TP / 9 FP). Không có cơ sở
  ngôn ngữ học. Hiện nó là config `linguistic.errorChannelPairCountVeto` (đặt `null` để tắt)
  thay vì hằng số chôn trong code, nhưng vẫn cần chứng minh lại bằng OOF, không phải dev.

**C5 — recall thực tế còn thấp:** strict rule-ID dev R=0,165; semantic dev R=0,2717, nghĩa là
73–83% lỗi chính tả trong tập nhãn không được bắt. Phần dùng được cho production hiện nay
là rule deterministic (whitespace / punctuation / unicode / protected ranges).

**C4 — nhánh attention là code chết trên runtime:** `src/attention-reranker.mjs` (577 dòng),
tokenizer, schema, `src/data/attention-reranker.int8.bin` (289 KB), ~10 tool Python và ~10
file test tồn tại nhưng `attentionMode = OFF` khi phục vụ. Đây là gánh nặng bảo trì; cần
quyết định dứt điểm: batch hoá rồi bật, hoặc tách sang branch nghiên cứu.

### Profiler

```bash
npm run profile:engine
```

Task 6 đã giảm `scoreCandidateOverSurfaces` khoảng 5,6 lần và `_pBiRaw` khoảng 11 lần so
với plan-era baseline. Startup/RSS vẫn bị chi phối bởi backend TSV. Xem bảng hiệu năng ở
mục "Kết quả cuối": p95 phụ thuộc máy và corpus, đừng trích một con số rời khỏi ngữ cảnh.

## Scripts

| Command | Chức năng |
|---|---|
| `npm start` | Server + demo UI, profile `lite` (nhanh, nhẹ) |
| `npm run start:full` | Server với artifact đầy đủ |
| `npm run check -- "<tin nhắn>"` | Kiểm tra nhanh một tin trong terminal |
| `npm test` | Toàn bộ node test |
| `npm run test:fast` | Chỉ core + rule regression |
| `npm run test:python` | Test cho `tools/` (qua `uv`) |
| `npm run test:all` | Node + Python |
| `npm run sms:dataset` | Sinh lại `dataset_sms/` (deterministic) |
| `npm run sms:profile` | Build artifact `lite` từ train split |
| `npm run sms:eval` | Đánh giá trên SMS dev (profile lite) |
| `npm run sms:eval:full` | Đánh giá trên SMS dev (artifact đầy đủ) |
| `npm run bench` | Benchmark tổng (corpus VSEC/synthetic) |
| `npm run eval:overlap` | Đo chồng lấn phân bố lỗi train ↔ dev/test (review C1) |
| `npm run data:audit` | Split/leakage audit |
| `npm run eval:spelling:dev` | Dev spelling evaluation |
| `npm run eval:spelling:test` | Guarded final test |
| `npm run profile:engine` | Startup/RSS/latency/hotspot |

(Trên Windows dùng `npm.cmd`; các script có tiền tố `ENGINE_PROFILE=lite` cần chạy bằng
`$env:ENGINE_PROFILE = "lite"` rồi gọi `node` trực tiếp.)

## Cấu trúc thư mục

```text
src/                  runtime engine, server, rules và data
dataset_sms/          bộ SMS tiếng Việt viết trong repo (template + split + manifest)
benchmark/sms/        cùng dữ liệu ở định dạng benchmark row
public/               demo UI
benchmark/            labeled corpora và results
dataset_raw/          raw downloaded datasets
dataset_artifacts/    split-safe/generated evaluation artifacts
tools/                build, audit, evaluation, profiling
test/                 Node/Python tests
docs/                 design, implementation plan, execution log
config/               frozen spelling tuning state + acceptance gates
```

## Bốn khiếm khuyết mô hình tìm ra từ một gợi ý sai

Một tin nhắn thật — `"Khong chia se ma nay cho bat ky ai"` — bị gợi ý `"ma" → "mà"` trong khi
đúng phải là `"mã này"`. Truy ngược ra bốn lỗi độc lập, tất cả đều đã sửa:

1. **Cắt LM phá vỡ cuộc cạnh tranh ứng viên.** Cắt theo tần suất từng dòng giữ `"mà nay"`
   (285) và bỏ `"mã này"` (63, hạng ~322.000). Bộ giải mã sau đó thấy bằng chứng *nhất trí*
   cho đúng một cách viết. Nay cắt theo **nhóm ứng viên** (gom theo khoá đã bỏ dấu, `"ma nay"`
   gom cả `mà nay / mã này / mã nay / mà này`), giữ hoặc bỏ cả nhóm, tối đa 6 đối thủ mỗi nhóm.
   *Mô hình được phép nhỏ, không được phép thiên vị một phía.*
2. **Artifact tự mâu thuẫn.** Trigram `"chia sẻ mã"` = 102 tồn tại nhưng bigram `"sẻ mã"` = 0,
   vì bản build gốc cắt bigram và trigram độc lập nhau. Back-off `P(mã|sẻ)` rơi về ~0 và dìm
   luôn bằng chứng trigram. Builder nay khôi phục bất biến
   `c(a,b) ≥ Σ_c c(a,b,c)` (+20.949 dòng thêm, 33.501 dòng nâng count). Lỗi này **có sẵn trong
   artifact 80 MB gốc** — đáng sửa cả ở `tools/build_lm.py`.
3. **Rule không dùng lại quyết định của chính nó.** Ngữ cảnh trái lấy từ lựa chọn của beam;
   beam phân giải `"se"` theo tần suất thuần thành `"sẽ"` (3,0M) thay vì `"sẻ"` (446k), rồi
   `"sẽ mà"` thắng một `"sẽ mã"` không tồn tại. Nay vị trí đã quyết được đưa tiếp sang vị trí sau.
4. **Dạng thô được quyền bỏ phiếu trong tin không dấu.** `"nay"` cũng là từ có thật, nên
   `"mà nay"` (285) áp đảo `"mã này"` (63) chỉ vì bản thân `"nay"` được tính là bằng chứng —
   trong khi đó chính là thứ đang cần khôi phục dấu. Khi tin trông như gõ không dấu, hàng xóm
   không dấu chỉ đóng góp các cách đọc *đã khôi phục*.

Kết quả: engine **không còn gợi ý sai** (nó im lặng ở `"ma"` thay vì khẳng định `"mà"`, và bắt
thêm được `"se" → "sẻ"`). Nó vẫn **chưa** chủ động đề xuất `"mã"`, vì kho ngữ liệu báo chí thật
sự có `"mà nay"` phổ biến hơn `"mã này"`, còn tập train SMS không chứa cách nói OTP (nhóm
template đó rơi vào dev — đúng theo thiết kế chống leakage). Muốn bắt được, cần thêm SMS thật
vào train, không phải nới ngưỡng.

## Tiến độ và roadmap

Đã hoàn thành Task 1–9 của spelling optimization plan: Viwiki converter, fail-fast LM loader, trigram formulation, Telex/candidate cascade, oracle audit, context hotspot, length-aware SymSpell, split-safe evaluation và dev-only tuning. Default config vẫn là winner của quá trình tuning.

Việc còn lại:

- Experiment Modified Kneser-Ney độc lập.
- Versioned binary LM và runtime word-ID backend.
- Giảm cold-start/RSS và bỏ full TSV parsing khi deploy.
- Khoá một tập test MỚI (tốt nhất là SMS thật do team gán nhãn) và chạy đúng một lần —
  tập held-out hiện tại đã thành tập tuning (review C2).
- Chứng minh lại `errorChannelPairCountVeto` bằng OOF thay vì dev (review B4).
- Quyết định dứt điểm nhánh attention: batch hoá rồi bật, hoặc tách sang branch nghiên cứu
  để runtime nhẹ đi (review C4).
- Chuyển artifact >10 MB sang Git LFS / storage ngoài và thêm LICENSE (review E5/E6).
- Mở rộng `dataset_sms/` bằng SMS thật do team gán nhãn (bộ hiện tại là tổng hợp từ template).
- Bật lane word-boundary: dataset đã có lớp lỗi `BOUNDARY` để calibrate.
- Nâng recall lớp `ALL_UNACCENTED` (63,9% trên test) — đây là lớp lỗi phổ biến nhất thực tế.

Chi tiết tại `docs/plans/2026-08-24-spelling-engine-optimization.md` và `docs/plans/execution-log-spelling-optimization.md`.

## Lưu ý production

- Đây là POC nghiên cứu, chưa phải service đã harden.
- Bốn lỗi cấp production ở lớp HTTP (A1–A4) đã sửa và có test; nhưng vẫn còn thiếu auth,
  rate limit, TLS termination, structured logging và metrics.
- Phải kiểm tra license dataset/dictionary trước khi phân phối. Repo **chưa có LICENSE**
  và chưa ghi license cho VSEC / viwiki / underthesea (review E6) — cần chủ repo quyết định.
- `DEMO_ENDPOINTS` phải để tắt ở môi trường thật.
- Không tự động học whitelist/exception từ nội dung người dùng.
- Rule deterministic đáng tin cậy hơn statistical spelling hiện tại.
- Không nới rule mù quáng để tăng recall; mọi thay đổi phải qua dev evaluation và precision
  constraints.
- Repo nặng vì commit thẳng dữ liệu (`.git` 55 MB, `dataset_artifacts` 94 MB, `src/data`
  47 MB — riêng `lm-ngrams.tsv` 42 MB). Nên chuyển artifact >10 MB sang Git LFS hoặc storage
  ngoài (review E5); việc này đụng vào lịch sử git nên chưa thực hiện.

## Thay đổi theo review 2026-09-03

Nguồn: `spellchecker-engine-review.txt`.

| Mã | Vấn đề | Trạng thái |
|---|---|---|
| A1 | 1 request với `Host` hỏng giết cả server | ✅ đã sửa + test |
| A2 | Body chia chunk làm hỏng ký tự tiếng Việt (`U+FFFD`) | ✅ đã sửa + test |
| A3 | Giới hạn payload 1 MB, không destroy socket, trả 400 | ✅ 16 KB, `413`, destroy sau khi flush |
| A4 | `/benchmark` + `/config` mở mặc định | ✅ sau `DEMO_ENDPOINTS=1` |
| B1 | Trùng `ruleId` giữa spelling và word-boundary | ✅ `POSSIBLE_WORD_BOUNDARY_ERROR` |
| B2 | Lane SHADOW vẫn tính toán mỗi request | ✅ kiểm tra mode trước vòng lặp / trong `supports()` |
| B3 | Dùng confidence đã làm tròn làm ngưỡng quyết định | ✅ giữ giá trị thô, làm tròn ở serialize |
| B4 | Magic number `pairCount !== 2` fit vào dev | ⚠️ thành config `errorChannelPairCountVeto`, vẫn cần OOF |
| B5 | Code chết `.concat(shadowMode ? [] : [])` | ✅ đã bỏ |
| B6 | Nhánh lexicon không làm gì, `src:'both'` không bao giờ đạt tới | ✅ merge đối xứng, comment đúng hành vi |
| B7 | `degraded` phình vô hạn | ✅ ring 50 + counter + `/healthz` |
| B8 | Message tiếng Anh + gán sai rule + hardcode 0.96 | ✅ tiếng Việt, `POSSIBLE_MISSING_DIACRITIC`, config |
| B9 | Config "frozen" khác config đang chạy | ✅ runtime nạp file + `servingOverrides` + test |
| C1 | Điểm số dựa vào bảng tra từ train | ⚠️ đã đo lại và ghi rõ (`npm run eval:overlap`) |
| C2 | Held-out đã bị dùng lại | ⚠️ đã ghi đúng vào README; cần tập test mới |
| C3 | Gate trôi theo kết quả | ⚠️ ghi nhận, chưa thống nhất một gate |
| C4 | Nhánh attention là code chết | ⚠️ ghi rõ trong `servingOverrides`, chưa gỡ/bật |
| D1–D5 | README lệch code | ✅ cập nhật + CI chạy bench mỗi PR |
| E1 | Test phụ thuộc `.tmp/` fail trên clone sạch | ✅ skip có điều kiện |
| E2 | Không khai báo dependency Python | ✅ `pyproject.toml` + `uv sync` (62 test pass) |
| E3 | `npm test` chỉ chạy 2 file | ✅ chạy toàn bộ + `test:python` |
| E4 | Không có CI | ✅ `.github/workflows/ci.yml` |
| E5 | Artifact lớn commit thẳng | ❌ cần quyết định về Git LFS/history |
| E6 | Không có LICENSE | ❌ cần chủ repo chọn license |
| §7 | Ngập cảnh báo với SMS không dấu | ✅ cờ `summary.unaccentedContent` cấp tin nhắn |

## Cải tiến cho chạy local (2026-09-03)

| Việc | Kết quả |
|---|---|
| Profile dữ liệu `lite` (`ENGINE_PROFILE`) | cold start 5,8 s → **0,7 s**, RSS 857 → **249 MB**, p95 9,6 → **4,9 ms** |
| Bộ dữ liệu SMS trong repo (`dataset_sms/`) | 62 template / 35 group / 1.942 tin có nhãn, split theo group, license sạch |
| LM + lexicon huấn luyện theo miền SMS | 80 MB → 5,4 MB; SMS dev precision 0,957 → **0,976**, báo động giả trên tin sạch 1/30 → **0/30** |
| Đánh giá đúng miền (`tools/eval_sms.mjs`) | held-out SMS: recall **0,648**, precision **0,939** (chạy một lần, khai báo trước) |
| CLI `npm run check` | thử một tin trong terminal, có gạch chân vị trí lỗi |
| Sửa lỗi gợi ý sai `"ma nay" → "mà"` (2026-09-04) | 4 khiếm khuyết mô hình, xem mục dưới; benchmark VSEC 60,7% → **62,0%** recall |
| `runBenchmark(engine, corpusDir)` | tham số corpus trước đây bị `void` bỏ đi — nay dùng được, `benchmark/sms/` chấm riêng |
