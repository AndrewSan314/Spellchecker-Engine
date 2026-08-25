# SMS VALIDATION ENGINE — FIX PLAN NGẮN GỌN

> Mục tiêu: giữ nguyên architecture hiện tại, chỉ sửa các bug và nâng chất lượng linguistic engine.  
> Làm theo đúng thứ tự bên dưới. Không nhảy sang NLP/AI khác.

---

# 1. Sửa benchmark trước

## Vấn đề
Benchmark hiện có thể báo `100% precision` dù engine trả thêm issue sai.

## Cách sửa
Trong `benchmark/run-benchmark.mjs`:

- Sau khi match hết `expected issues`, mọi issue còn lại trên row `fullyLabeled=true` phải tính là **False Positive**.
- Tính đúng:

```text
TP = expected issue match được
FP = issue thừa
FN = expected issue bị miss

precision = TP / (TP + FP)
recall    = TP / (TP + FN)
```

Thêm field:

```json
{
  "fullyLabeled": true
}
```

Nếu row chỉ dùng để test recall thì:

```json
{
  "fullyLabeled": false
}
```

## Test bắt buộc

Case:

```text
Expected:
MULTIPLE_WHITESPACE

Actual:
MULTIPLE_WHITESPACE
POSSIBLE_SPELLING_ERROR
```

Kết quả phải là:

```text
TP = 1
FP = 1
```

---

# 2. Tách train / dev / test, không để leakage

## Vấn đề
Một số câu benchmark gần như có mặt trong corpus train sau khi bỏ dấu.

## Cách sửa

Tạo:

```text
data/train.txt
benchmark/dev.json
benchmark/test.json
```

Tạo tool:

```text
tools/check-data-leakage.mjs
```

Normalize trước khi check overlap:

```text
lowercase
NFC
strip Vietnamese accents
replace number/code/url
collapse spaces
```

Ví dụ:

```text
"Kính chào quý khách"
"Kinh chao quy khach"
```

phải được coi là cùng một cluster.

Nếu cluster có trong train thì không được xuất hiện trong test.

## Done khi

```text
train/test normalized overlap = 0
```

---

# 3. Fix Unicode NFD bypass

## Vấn đề

`quý` dạng NFC bắt được.

Nhưng dạng:

```text
quy\u0301
```

có thể không bị phát hiện trong `NON_ACCENTED`.

## Cách sửa

Trong accent-mode rule:

```javascript
const analysis = token.original.normalize('NFC');
```

Detect accent trên `analysis`.

Nhưng output:

```text
start/end/value
```

vẫn lấy từ original text.

## Test

Cả hai phải cảnh báo:

```text
quý
quy\u0301
```

---

# 4. Fix URL protected range

## Vấn đề

Regex hiện có thể nuốt:

```text
https://example.com,Quy
https://a.vn!Nhan
```

làm punctuation rule không chạy.

## Cách sửa

URL detector phải dừng ở URL thật.

Sau khi regex match URL candidate, trim punctuation cuối:

```text
, ! ? ; :
```

và dấu kết câu:

```text
.
```

nếu nó không phải một phần path/domain.

Ví dụ:

```text
https://example.com,Quy
```

protected range chỉ là:

```text
https://example.com
```

Sau đó rule phải detect thiếu space sau `,`.

## Test

Không warning trong:

```text
https://example.com
```

Có warning trong:

```text
https://example.com,Quy khach
https://a.vn!Nhan qua
```

---

# 5. Fix config thực sự có tác dụng

## Vấn đề

Một số config reload nhưng rule vẫn dùng giá trị cũ.

Ví dụ:

```text
zeroWidthSeverity
beamWidth
originalPrior
emojiPolicy
```

## Cách sửa

Không capture config trong constructor.

Rule phải lấy snapshot lúc `validate()`:

```javascript
const snap = configService.snapshot();

const severity =
  snap.get('rules.zeroWidthSeverity');
```

Beam decoder nên thành stateless:

```javascript
decoder.decode(positions, {
  beamWidth,
  originalPrior
});
```

Không:

```javascript
new BeamSearchDecoder(lm, beamWidth, prior)
```

rồi giữ config cũ mãi.

## Test

Reload:

```text
beamWidth: 5 -> 1
zeroWidthSeverity: WARNING -> ERROR
```

behavior phải đổi ngay mà không restart engine.

---

# 6. Implement SHADOW đúng nghĩa

## Vấn đề

Hiện `SHADOW` vẫn trả linguistic warning cho user.

## Cách sửa

Semantics:

```text
OFF
→ không chạy

SHADOW
→ chạy
→ collect metric/debug
→ KHÔNG đưa vào ValidationResult

ACTIVE
→ chạy
→ trả issue cho user
```

Trong engine:

```javascript
if (mode === 'SHADOW' && isLinguistic(issue)) {
  shadowCollector.add(issue);
  continue;
}
```

## Test

Input:

```text
Kính chao quý khách
```

SHADOW:

```text
response.issues = []
shadowIssues contains chao -> chào
```

---

# 7. Không cho client chỉnh confidence threshold

## Vấn đề

Public API hiện cho client gửi:

```json
{
  "minConfidence": 0,
  "minMargin": 0
}
```

=> client có thể làm engine spam warning.

## Cách sửa

Xóa threshold khỏi public request.

Threshold chỉ lấy từ server config:

```text
configService.snapshot()
```

Benchmark nội bộ muốn override thì inject config riêng trong test.

---

# 8. Rule critical crash không được fail-open

## Vấn đề

Nếu deterministic rule throw, engine hiện có thể log rồi tiếp tục và trả:

```text
valid = true
```

## Cách sửa

Chia rule:

```text
CRITICAL
OPTIONAL
```

Critical:

```text
accent mode
unicode invalid
whitespace/punctuation business rules
abbreviation nếu đang enable
```

Optional:

```text
missing diacritic
spelling
```

Nếu CRITICAL throw:

```text
throw ValidationEngineError
```

Nếu OPTIONAL throw:

```text
skip rule
record degraded metric
```

---

# 9. Thay lexicon 888 từ

## Vấn đề

888 từ quá ít.

Từ đúng như:

```text
giúp
đáp
đều
suốt
thoáng
```

có thể bị coi là unknown rồi spell-correct sai.

## Cách sửa

Target:

```text
30k–100k+ Vietnamese entries
```

Mỗi entry:

```text
word
frequency
source
type
```

Ví dụ:

```text
khách    582314    corpus    GENERAL
Viettel   93120    business  BRAND
voucher   12431    approved  FOREIGN
```

Không hardcode thêm từng từ chỉ để test pass.

Phải build từ dictionary/corpus có license rõ.

---

# 10. Bỏ frequency giả

## Vấn đề

Nhiều từ hiện có cùng:

```text
150000
```

nhưng frequency này không phải count thật.

Trong khi ranker lại dùng:

```text
log(freq)
```

## Cách sửa

Build frequency từ corpus thật:

```text
approved SMS corpus
+
approved Vietnamese corpus
```

Tạo:

```text
tools/build-lexicon-frequency.mjs
```

Nếu từ business được whitelist nhưng không có trong corpus:

```text
frequency = 0
```

và thêm field riêng:

```text
priorWeight
```

Không dùng frequency giả để boost.

---

# 11. Fix missing-diacritic logic

## Vấn đề

Hiện chỉ local-evaluate token nếu global beam đã đổi token.

Nếu beam giữ original thì token đó không được xét tiếp.

## Cách sửa

Beam chỉ dùng để suy ra context.

Sau beam:

```javascript
for (const ambiguousToken of ambiguousTokens) {
  const ranked = rankAllCandidates(
    ambiguousToken,
    contextFromBeam
  );

  const best = ranked[0];
  const second = ranked[1];

  if (best.word === original) continue;
  if (!passesConfidence(best, second)) continue;

  emitWarning();
}
```

Tức:

```text
EVERY ambiguous token
→ local ranking
→ confidence gate
```

Không phụ thuộc beam đã flip hay chưa.

---

# 12. Tăng context ranking bằng trigram

## Vấn đề

Case:

```text
đọc kỹ huong dẫn
```

engine có thể chọn:

```text
hưởng
```

thay vì:

```text
hướng
```

## Cách sửa

Score candidate bằng:

```text
unigram
+
left bigram
+
right bigram
+
centered trigram
```

Ví dụ:

```text
đọc hướng dẫn
```

phải mạnh hơn:

```text
đọc hưởng dẫn
```

Pseudo:

```text
score =
  0.2 * unigram
+ 1.0 * leftBigram
+ 1.0 * rightBigram
+ 1.5 * centeredTrigram
```

Weights chỉ là seed, tune bằng dev set.

## Test contrast

```text
đọc huong dẫn
=> hướng

được huong ưu đãi
=> hưởng
```

---

# 13. Confidence không được coi softmax = xác suất thật

## Vấn đề

Engine đang có:

```text
confidence = 0.98
1.00
```

nhưng đây chỉ là softmax trên heuristic score.

## Cách sửa

Trước mắt:

- vẫn giữ field `confidence`;
- hiểu nó là heuristic score.

Sau đó calibrate bằng dev set.

Input cho calibrator:

```text
top1 score
top2 score
margin
candidate frequency
context score
candidate count
correct/incorrect
```

Có thể dùng logistic regression offline:

```text
P(correct) =
sigmoid(
  b0
+ b1 * margin
+ b2 * context
+ b3 * frequency
)
```

Runtime chỉ load vài coefficients.

---

# 14. Fix known linguistic false positives

Tạo regression corpus với các câu sạch:

```text
Bạn vui lòng mang theo giấy tờ tùy thân khi đến nhận hàng.

Mọi thắc mắc của bạn sẽ được giải đáp trong giờ làm việc.

Phiếu bảo hành cần được giữ lại trong suốt thời gian sử dụng.

Sản phẩm này được bảo quản ở nơi khô ráo và thoáng mát.

Chào bạn  mình có thể giúp gì.

Hỏi gì đáp nấy:không sai.

A,B,C đều được tặng quà.
```

Các từ này không được sửa sai:

```text
mang -> mạng
giải -> giảm
đáp -> đãi
suốt -> suất
thoáng -> thông
giúp -> giữ
nấy -> này
đều -> để
```

Không fix bằng whitelist từng từ nếu chúng là từ tiếng Việt bình thường.

Fix bằng:

```text
better lexicon
real frequency
better context
confidence gate
```

---

# 15. Tạo independent missing-accent test

Thêm ít nhất các case:

```text
giay -> giấy
tuy -> tùy
than -> thân
phuc -> phục
duong -> đường
dia -> địa
thoai -> thoại
nhan -> nhân
tien -> tiền
mat -> mặt
kiem -> kiểm
roi -> rời
buoi -> buổi
toi -> tối
nang -> nâng
cap -> cấp
xu -> xử
ly -> lý
huong -> hướng
doi -> đổi
```

Không đưa exact test sentence này vào training corpus.

---

# 16. Optimize SymSpell lookup

## Vấn đề

Resolver hiện có thể scan toàn lexicon để tìm display surface.

Với:

```text
80k words
```

sẽ chậm.

## Cách sửa

Precompute lúc startup:

```javascript
Map<accentKey, bestSurface>
```

Ví dụ:

```text
khach -> khách
chao  -> chào
```

Lookup runtime:

```javascript
bestSurfaceByKey.get(key)
```

O(1).

Không loop toàn dictionary.

---

# 17. Whitelist phải có scope

## Vấn đề

Whitelist hiện global.

## Cách sửa

Tạo:

```text
SYSTEM
BRAND
CUSTOMER
```

Structure:

```javascript
{
  system: Set,
  byBrand: Map,
  byCustomer: Map
}
```

Lookup:

```text
customer
→ brand
→ system
```

Ví dụ:

```text
MyProduct
```

có thể valid cho customer A nhưng không valid cho customer B.

---

# 18. Abbreviation phải reload được

Hiện abbreviation seed file chỉ load startup.

Refactor service giống whitelist:

```text
immutable snapshot
+
reload()
+
atomic swap
```

Rule không query DB mỗi token.

---

# 19. N-gram production model

Current string-map đủ cho POC.

Production:

```text
word -> integer id
```

Build offline:

```text
unigram
bigram
pruned trigram
```

Prune trigram:

```text
count < 2 hoặc 3
→ bỏ
```

Artifact có:

```text
modelVersion
corpusHash
lexiconHash
vocabSize
bigramCount
trigramCount
```

Load 1 lần startup.

---

# 20. Thêm explain tool

Tạo:

```text
tools/explain-linguistic.mjs
```

Input:

```bash
node tools/explain-linguistic.mjs \
  "Vui lòng đọc kỹ huong dẫn sử dụng"
```

Output:

```text
token: huong

candidate: hướng
frequency:
leftBigram:
rightBigram:
trigram:
rawScore:

candidate: hưởng
...

best:
confidence:
margin:
decision:
```

Tool này bắt buộc để debug linguistic FP/FN.

---

# 21. Metrics cần có

Không log raw SMS.

Add:

```text
validation_latency
issues_total{ruleId}
rule_failure_total{ruleId}
shadow_issue_total{ruleId}
suppressed_total{reason}
config_reload_total
```

Linguistic suppression reasons:

```text
LOW_CONFIDENCE
LOW_MARGIN
WHITELIST
PROTECTED_RANGE
NAME_GUARD
SHADOW_MODE
```

---

# 22. Load test lại bằng dictionary thật

Hiện:

```text
p95 ~2.5 ms
```

không có nhiều ý nghĩa vì lexicon chỉ 888 từ.

Tạo benchmark với:

```text
30k
80k
100k
```

lexicon entries.

Measure:

```text
p50
p95
p99
req/s
heap
RSS
```

Target ban đầu:

```text
p95 < 50 ms
```

---

# 23. Feature state sau khi sửa

## ACTIVE

Sau P0:

```text
ACCENT_CHARACTER_IN_NON_ACCENT_MODE
LEADING_WHITESPACE
TRAILING_WHITESPACE
MULTIPLE_WHITESPACE
WHITESPACE_BEFORE_PUNCTUATION
REPEATED_PUNCTUATION
MISSING_WHITESPACE_AFTER_PUNCTUATION
ZERO_WIDTH_CHARACTER
NON_BREAKING_SPACE
INVALID_CHARACTER
ABBREVIATION_DETECTED
```

## SHADOW

Cho tới khi benchmark độc lập đạt target:

```text
POSSIBLE_MISSING_DIACRITIC
POSSIBLE_SPELLING_ERROR
```

---

# 24. Quality gate trước khi ACTIVE linguistic

## Missing diacritic

Target:

```text
Precision >= 95%
Clean sentence FP <= 2%
```

## Spelling

Target:

```text
Precision >= 97%
```

Recall thấp hơn chấp nhận được.

Bài toán này:

```text
Precision > Recall
```

Nếu không chắc:

```text
NO WARNING
```

---

# 25. Thứ tự agent phải làm

```text
1. Fix benchmark FP
2. Fix train/test leakage
3. Fix NFD accent
4. Fix URL range
5. Fix config runtime/reload
6. Fix SHADOW
7. Remove client thresholds
8. Fix critical fail-open
9. Run full tests

10. Replace lexicon
11. Replace fake frequency
12. Fix ambiguous-token local evaluation
13. Improve trigram/context rank
14. Add clean regression corpus
15. Add independent accent corpus
16. Optimize SymSpell O(1)
17. Scoped whitelist
18. Reloadable abbreviation
19. Confidence calibration
20. Version/prune n-gram model
21. Metrics
22. Production-size load test
```

Sau mỗi bước:

```bash
npm test
npm run bench
```

Không đi bước tiếp theo nếu test fail.

---

# 26. Những thứ KHÔNG được làm

Không:

```text
add LLM
add external AI API
copy test sentence vào training
hide FP bằng allowExtra
whitelist từng từ bình thường để chữa cháy
lower confidence chỉ để tăng recall
hardcode frequency giả
normalize original rồi trả offset normalized
cho client chỉnh threshold
trả SHADOW issue cho user
silently ignore critical rule crash
```

---

# 27. End state mong muốn

```text
SMS Validator
=
Deterministic rules
+
Protected ranges
+
Large licensed Vietnamese lexicon
+
Real word frequencies
+
Accent reverse index
+
SymSpell candidate retrieval
+
N-gram contextual ranking
+
Calibrated confidence
+
Strict suppression
```

Không cần LLM.

Mục tiêu cuối:

```text
local
fast
low false positive
explainable
configurable
production-safe
```
