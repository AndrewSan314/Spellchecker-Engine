# IMPLEMENTATION PLAN — SMS Content Validation Engine

> **Mục đích của file này:** Đây là tài liệu triển khai để một coding agent có thể đi từ đầu đến cuối và xây được SMS Content Validation Engine mà gần như không phải tự suy luận kiến trúc.
>
> **Nguyên tắc quan trọng:** Agent phải làm theo thứ tự. Không được nhảy cóc sang spell-check / NLP khi deterministic rules, span mapping, protected ranges và test corpus chưa hoàn thành.
>
> **Định hướng bắt buộc:** Local-first, precision-first, rule-based trước, không gọi external AI/API, không tự sửa nội dung người dùng.

---

# 0. TL;DR — KIẾN TRÚC CHỐT

Xây một **hybrid deterministic + statistical text linter** cho SMS bằng Java/Spring Boot.

Pipeline cuối:

```text
Original SMS
    │
    ▼
ValidationContext
    │
    ▼
ValidationDocumentBuilder
    ├── Unicode inspection
    ├── Tokenization
    └── ProtectedRangeDetector
            ├── URL
            ├── email
            ├── phone
            ├── date/time
            ├── money/number
            ├── transaction/product code
            └── placeholders
    │
    ▼
Deterministic Rule Engine
    ├── Message mode rules
    ├── Whitespace rules
    ├── Punctuation rules
    ├── Unicode/character rules
    └── Brandname abbreviation rules
    │
    ▼
Lexical Gate
    ├── protected? => skip
    ├── whitelist? => skip
    ├── allowed brand/technical term? => skip
    ├── valid dictionary word? => skip
    ├── possible missing accent? => Accent Candidate Pipeline
    └── otherwise => Typo Candidate Pipeline
    │
    ├──────── Accent Candidate Pipeline
    │          ├── reverse accent index
    │          ├── n-gram contextual ranker
    │          └── confidence/margin gate
    │
    └──────── Typo Candidate Pipeline
               ├── SymSpell-style candidate retrieval
               ├── Vietnamese-aware edit score
               ├── n-gram contextual ranker
               └── confidence/margin gate
    │
    ▼
IssueSuppressor
    │
    ▼
IssueConflictResolver
    │
    ▼
Sort by original UTF-16 position
    │
    ▼
ValidationResult
```

Không dùng LLM. Không dùng external API. Không auto-correct.

---

# 1. YÊU CẦU NGHIỆP VỤ PHẢI GIỮ NGUYÊN

Engine nhận nội dung SMS và metadata, sau đó **phát hiện + locate + classify + warn**.

Engine **không**:

- tự thay thế nội dung;
- tự lưu bản sửa;
- thay đổi original text;
- rewrite SMS;
- gửi SMS;
- quyết định business flow gửi SMS thay frontend/backend campaign service.

Các nhóm lỗi bắt buộc:

1. Chọn `NON_ACCENTED` nhưng có ký tự tiếng Việt có dấu.
2. Chọn `ACCENTED` nhưng có token có khả năng thiếu dấu.
3. Brandname `VT_TENDOO`: cảnh báo abbreviation theo danh sách cấu hình.
4. Brandname khác: không áp dụng abbreviation rule của `VT_TENDOO`.
5. Leading whitespace.
6. Trailing whitespace.
7. Multiple whitespace.
8. Whitespace trước dấu câu.
9. Repeated punctuation.
10. Missing whitespace sau punctuation, có exception.
11. Unicode/ký tự không hợp lệ hoặc bất thường.
12. Possible spelling error.
13. Whitelist / dictionary / allowed terms phải cấu hình được.
14. Trả chính xác `start/end` để frontend highlight.
15. Severity có `ERROR/WARNING/INFO`.
16. Ưu tiên false-positive thấp hơn recall cao.

---

# 2. QUYẾT ĐỊNH KỸ THUẬT CỐ ĐỊNH

## 2.1 Runtime

Nếu triển khai trong repo hiện có:

- Giữ nguyên Java/Spring Boot/Maven hoặc Gradle version của repo.
- Không tự upgrade framework chỉ vì plan này.

Nếu greenfield POC:

- Java 21.
- Spring Boot.
- Maven.
- JUnit 5.
- AssertJ.
- Micrometer nếu app đã dùng metrics.
- ICU4J cho Unicode.
- Không thêm NLP framework lớn ở Phase 1.

## 2.2 Index convention

**Bắt buộc:**

```text
start = inclusive
end   = exclusive
```

Index dùng **UTF-16 code units**, tương thích tự nhiên với:

- Java `String` offsets.
- JavaScript string offsets.

Ví dụ:

```text
"Xin chao"
     ^^^^
```

Nếu `chao` bắt đầu ở Java offset 4:

```json
{
  "start": 4,
  "end": 8
}
```

### Tuyệt đối không

- convert offset sang Unicode code-point index ở một số rule nhưng rule khác lại dùng UTF-16;
- normalize string rồi lấy offset trên normalized string trả thẳng cho frontend;
- trim content trước validation.

## 2.3 Precision-first

Với linguistic rules:

```text
Nếu không chắc => không cảnh báo.
```

Không dùng triết lý "có khả năng thì warning hết".

Mục tiêu:

```text
Precision > Recall
```

---

# 3. REPO / LIBRARY THAM KHẢO VÀ CHÍNH XÁC PHẦN ĐƯỢC LẤY

## 3.1 LanguageTool

Repo:

- https://github.com/languagetool-org/languagetool

Lấy:

- `Rule` abstraction.
- Stable `ruleId`.
- `RuleMatch` chứa range + message + suggestion.
- active/disabled rule concept.
- match post-processing/filter concept.
- testing kiểu `should warn` / `should not warn`.

Không lấy:

- toàn bộ LanguageTool runtime;
- language modules;
- grammar XML system;
- POS/morphology stack.

Implementation nội bộ tương đương:

```java
public interface SmsValidationRule {
    String id();
    int priority();
    boolean supports(ValidationContext context);
    List<ValidationIssue> validate(
        ValidationContext context,
        ValidationDocument document
    );
}
```

## 3.2 RedPen

Repo:

- https://github.com/redpen-cc/redpen

Lấy:

- validator độc lập;
- enable/disable/configure validator;
- error suppression / exception;
- writing-standard linter mindset.

Không lấy dependency runtime.

## 3.3 ICU4J

Docs:

- https://unicode-org.github.io/icu/
- `com.ibm.icu.text.Normalizer2`

Dùng trực tiếp:

- NFC/NFD inspection.
- Unicode properties khi cần.
- UnicodeSet nếu có lợi.

**Lưu ý:** Normalization chỉ dùng trong analysis data; original text không được mutate.

## 3.4 Vietnamese-Accent-Prediction

Repo:

- https://github.com/tienthanhdhcn/Vietnamese-Accent-Prediction

Lấy ý tưởng:

- unigram/bigram language model;
- generate accented candidates;
- contextual scoring;
- top-N sequence;
- dynamic programming / path decoding mindset.

Không copy implementation cũ.

Không giữ:

- hard-coded `maxWordLength`;
- candidate array fixed-size;
- generic K-shortest-path graph;
- dataset chưa được legal approve.

Thay bằng:

- reverse accent index;
- Viterbi/beam search trên layered token candidates;
- own domain n-gram model.

## 3.5 VNOpenAI/vn-accent

Repo:

- https://github.com/VNOpenAI/vn-accent

Lấy:

- n-gram + beam-search approach;
- corpus preprocessing ideas;
- 2-gram/3-gram tradeoff.

Không lấy:

- LSTM;
- Transformer;
- Evolved Transformer;
- model neural.

## 3.6 SymSpell / customized-symspell

Repos:

- https://github.com/wolfgarbe/SymSpell
- https://github.com/MighTguY/customized-symspell

Lấy:

- Symmetric Delete candidate generation;
- fast typo candidate lookup;
- exclusion/whitelist concept;
- weighted edit reranking concept.

Production rule:

- Chỉ dùng stable artifact đã được dependency/license review.
- Không phụ thuộc `SNAPSHOT` trong production.
- Nếu không có stable artifact phù hợp, implement internal minimal SymSpell-style delete index sau interface `TypoCandidateProvider`.

## 3.7 Underthesea dictionary / Viet74K

Repos:

- https://github.com/undertheseanlp/dictionary
- https://github.com/duyet/vietnamese-wordlist

Chỉ dùng:

- POC;
- coverage comparison;
- benchmark.

Không mặc định bundle production vì license phải review.

Production ưu tiên:

- dictionary do tổ chức sở hữu / có quyền sử dụng rõ;
- approved SMS corpus;
- approved brand/customer dictionaries.

## 3.8 VnCoreNLP

Repo:

- https://github.com/vncorenlp/VnCoreNLP

Không dùng trong core Phase 1/2 mặc định.

Chỉ POC nếu sau benchmark vẫn còn false positive lớn ở:

- person names;
- organization names;
- multi-syllable Vietnamese tokenization.

---

# 4. PROJECT STRUCTURE BẮT BUỘC

Nếu service riêng:

```text
src/main/java/com/<org>/smsvalidation/
├── api/
│   ├── SmsValidationController.java
│   ├── ValidateSmsRequest.java
│   ├── ValidateSmsResponse.java
│   └── ValidationIssueResponse.java
│
├── core/
│   ├── SmsValidationEngine.java
│   ├── SmsValidationRule.java
│   ├── ValidationContext.java
│   ├── ValidationDocument.java
│   ├── ValidationIssue.java
│   ├── ValidationResult.java
│   ├── Severity.java
│   ├── MessageMode.java
│   └── RuleIds.java
│
├── preprocess/
│   ├── ValidationDocumentBuilder.java
│   ├── SmsTokenizer.java
│   ├── UnicodeInspector.java
│   ├── VietnameseTextNormalizer.java
│   └── protectedrange/
│       ├── ProtectedRangeDetector.java
│       ├── CompositeProtectedRangeDetector.java
│       ├── UrlRangeDetector.java
│       ├── EmailRangeDetector.java
│       ├── PhoneRangeDetector.java
│       ├── DateTimeRangeDetector.java
│       ├── NumericRangeDetector.java
│       ├── CodeRangeDetector.java
│       └── PlaceholderRangeDetector.java
│
├── rule/
│   ├── mode/
│   │   └── AccentCharacterInNonAccentModeRule.java
│   ├── whitespace/
│   │   ├── LeadingWhitespaceRule.java
│   │   ├── TrailingWhitespaceRule.java
│   │   ├── MultipleWhitespaceRule.java
│   │   └── WhitespaceBeforePunctuationRule.java
│   ├── punctuation/
│   │   ├── MissingWhitespaceAfterPunctuationRule.java
│   │   └── RepeatedPunctuationRule.java
│   ├── unicode/
│   │   ├── InvalidCharacterRule.java
│   │   ├── ZeroWidthCharacterRule.java
│   │   └── NonBreakingSpaceRule.java
│   ├── abbreviation/
│   │   └── AbbreviationDetectedRule.java
│   └── linguistic/
│       ├── PossibleMissingDiacriticRule.java
│       └── PossibleSpellingErrorRule.java
│
├── lexical/
│   ├── LexiconService.java
│   ├── Lexeme.java
│   ├── LexemeType.java
│   ├── WhitelistService.java
│   ├── AbbreviationService.java
│   ├── AccentIndex.java
│   ├── AccentStripper.java
│   └── repository/
│       ├── LexiconRepository.java
│       ├── WhitelistRepository.java
│       └── AbbreviationRepository.java
│
├── language/
│   ├── NGramLanguageModel.java
│   ├── NGramScore.java
│   ├── Candidate.java
│   ├── CandidateRanker.java
│   ├── BeamSearchDecoder.java
│   ├── ConfidencePolicy.java
│   └── typo/
│       ├── TypoCandidateProvider.java
│       ├── SymSpellCandidateProvider.java
│       └── VietnameseEditScorer.java
│
├── suppression/
│   ├── IssueSuppressor.java
│   ├── ProtectedRangeIssueSuppressor.java
│   ├── WhitelistIssueSuppressor.java
│   └── CompositeIssueSuppressor.java
│
├── conflict/
│   └── IssueConflictResolver.java
│
├── config/
│   ├── ValidationProperties.java
│   ├── ValidationConfigSnapshot.java
│   ├── ValidationConfigService.java
│   └── ValidationConfigReloader.java
│
└── metrics/
    └── ValidationMetrics.java
```

Test tree mirror source tree.

---

# 5. DOMAIN CONTRACTS — CODE TRƯỚC, RULE SAU

## 5.1 Severity

```java
public enum Severity {
    ERROR,
    WARNING,
    INFO
}
```

## 5.2 MessageMode

```java
public enum MessageMode {
    ACCENTED,
    NON_ACCENTED
}
```

Không dùng boolean `hasAccent`.

## 5.3 ValidationContext

```java
public record ValidationContext(
    String content,
    MessageMode messageMode,
    String brandname,
    String customerId
) {
    public ValidationContext {
        Objects.requireNonNull(content, "content");
        Objects.requireNonNull(messageMode, "messageMode");
    }
}
```

`customerId` có thể nullable nếu hệ thống chưa cần.

## 5.4 ValidationIssue

```java
public record ValidationIssue(
    String ruleId,
    Severity severity,
    int start,
    int end,
    String value,
    String message,
    List<String> suggestions,
    Double confidence
) {
    public ValidationIssue {
        Objects.requireNonNull(ruleId);
        Objects.requireNonNull(severity);
        Objects.requireNonNull(value);
        Objects.requireNonNull(message);

        if (start < 0) {
            throw new IllegalArgumentException("start < 0");
        }
        if (end < start) {
            throw new IllegalArgumentException("end < start");
        }

        suggestions = suggestions == null
            ? List.of()
            : List.copyOf(suggestions);
    }
}
```

### Invariant bắt buộc

Cho mọi issue:

```java
context.content().substring(issue.start(), issue.end())
```

phải bằng:

```java
issue.value()
```

ngoại trừ issue zero-length nếu sau này có loại đó. Phase này không tạo zero-length issue.

Viết test invariant global.

## 5.5 ValidationResult

Khuyến nghị:

```java
public record ValidationResult(
    boolean valid,
    boolean hasErrors,
    boolean hasWarnings,
    List<ValidationIssue> issues
) {}
```

Quy ước:

```text
hasErrors = tồn tại issue severity ERROR
hasWarnings = tồn tại WARNING
valid = !hasErrors
```

Nếu business sau này yêu cầu một số ERROR vẫn non-blocking, thêm `blocking` riêng; không overload ý nghĩa severity.

## 5.6 SmsValidationRule

```java
public interface SmsValidationRule {

    String id();

    int priority();

    boolean supports(ValidationContext context);

    List<ValidationIssue> validate(
        ValidationContext context,
        ValidationDocument document
    );
}
```

### Quy tắc

- Rule không mutate context/document.
- Rule không query DB.
- Rule không gọi network.
- Rule không throw vì input người dùng bình thường.
- Rule phải deterministic nếu cùng context + config snapshot.

---

# 6. VALIDATION DOCUMENT — GIỮ ORIGINAL OFFSETS

```java
public record ValidationDocument(
    String originalText,
    String analysisText,
    List<SmsToken> tokens,
    List<ProtectedRange> protectedRanges
) {}
```

`analysisText` chỉ để lookup/comparison.

`originalText` là nguồn duy nhất của output offsets/value.

## 6.1 SmsToken

```java
public record SmsToken(
    String original,
    String normalized,
    int start,
    int end,
    TokenType type
) {}
```

Token types baseline:

```java
public enum TokenType {
    WORD,
    NUMBER,
    PUNCTUATION,
    WHITESPACE,
    OTHER
}
```

## 6.2 ProtectedRange

```java
public record ProtectedRange(
    int start,
    int end,
    ProtectedRangeType type
) {
    public boolean overlaps(int s, int e) {
        return s < end && e > start;
    }

    public boolean contains(int s, int e) {
        return s >= start && e <= end;
    }
}
```

Types:

```java
URL,
EMAIL,
PHONE,
DATE,
TIME,
NUMBER,
MONEY,
TRANSACTION_CODE,
PRODUCT_CODE,
PLACEHOLDER
```

---

# 7. PREPROCESSING PIPELINE

Method:

```java
ValidationDocument build(ValidationContext context)
```

Thứ tự:

1. `original = context.content()`.
2. Không trim.
3. Không lowercase original.
4. Tạo analysis forms khi cần.
5. Detect protected ranges trên original.
6. Tokenize original.
7. Gắn token với protected-range metadata qua offset.
8. Return immutable `ValidationDocument`.

## 7.1 VietnameseTextNormalizer

Có 3 operation khác nhau. Không trộn.

```java
String normalizeNfc(String value);
String lowerForLookup(String value);
String stripVietnameseDiacritics(String value);
```

### stripVietnameseDiacritics

Dùng NFD + bỏ combining marks, nhưng nhớ:

```text
đ / Đ
```

không tự biến thành:

```text
d / D
```

nên phải map explicit.

Pseudocode:

```java
String stripVietnameseDiacritics(String input) {
    String nfd = Normalizer2.getNFDInstance().normalize(input);
    StringBuilder out = new StringBuilder();

    for each Unicode code point cp:
        if cp is combining mark:
            continue;
        if cp == 'đ': append 'd';
        else if cp == 'Đ': append 'D';
        else append cp;

    return out.toString();
}
```

Test tối thiểu:

```text
quý        -> quy
khách      -> khach
đăng       -> dang
Đà         -> Da
Viettel    -> Viettel
SMS        -> SMS
```

---

# 8. PROTECTED RANGE DETECTION

Mục đích:

- rule punctuation không flag nhầm URL/number/date/code;
- spell checker không check token bên trong URL/code;
- giữ false positive thấp.

## 8.1 Detector interface

```java
public interface ProtectedRangeDetector {
    List<ProtectedRange> detect(String originalText);
}
```

Composite:

```java
@Component
public class CompositeProtectedRangeDetector {

    private final List<ProtectedRangeDetector> detectors;

    public List<ProtectedRange> detect(String text) {
        // 1. run all
        // 2. merge exact duplicates
        // 3. resolve overlap by type priority
        // 4. sort by start/end
    }
}
```

## 8.2 Priority khi overlap

Đề xuất:

```text
URL
EMAIL
PLACEHOLDER
TRANSACTION_CODE / PRODUCT_CODE
PHONE
DATE/TIME
MONEY
NUMBER
```

URL thắng NUMBER nếu URL có digit.

## 8.3 URL

Không tự phát minh URL parser phức tạp.

Baseline nhận:

```text
http://...
https://...
www....
domain.tld
```

Có thể dùng regex đã được test hoặc URI detector từ project existing.

Test:

Should protect:

```text
https://example.com
http://a.vn/x?id=10
www.viettel.vn
example.com
abc.com/path?a=1
```

Should not protect toàn câu:

```text
Xin chào.Quý khách
```

## 8.4 Email

Protect:

```text
abc@example.com
a.b+c@company.vn
```

## 8.5 Phone

Baseline Việt Nam:

```text
0912345678
+84912345678
84 912 345 678
```

Cẩn thận không protect mọi chuỗi số 3 ký tự.

## 8.6 Date / time

Protect:

```text
12/10/2026
12-10-2026
23:59
08:30
```

Không cần hiểu semantic calendar sâu trong validator này.

## 8.7 Number / money

Protect punctuation nội bộ:

```text
10.000
1,000,000
10,5
1.5
```

Nhưng không nhất thiết skip number khỏi mọi rule; mục tiêu chính là punctuation exception.

## 8.8 Code

Patterns ví dụ:

```text
DH123456
VT001
ABC-2026
OTP123
```

Không được regex quá rộng đến mức mọi uppercase word thành code.

Baseline:

- chữ + số;
- hoặc chữ uppercase + `-` + số;
- min length hợp lý.

## 8.9 Placeholder

Nếu SMS hỗ trợ template variables, thêm từ đầu.

Ví dụ:

```text
{{customer_name}}
${otp}
{amount}
```

**Agent phải kiểm tra syntax template hiện có của sản phẩm trước khi implement.**
Nếu chưa có thông tin, tạo detector interface + test fixture, không đoán production syntax.

---

# 9. RULE ENGINE

```java
@Service
public class SmsValidationEngine {

    private final ValidationDocumentBuilder documentBuilder;
    private final List<SmsValidationRule> rules;
    private final IssueSuppressor issueSuppressor;
    private final IssueConflictResolver conflictResolver;

    public ValidationResult validate(ValidationContext context) {

        ValidationDocument doc = documentBuilder.build(context);

        List<ValidationIssue> raw = rules.stream()
            .filter(rule -> rule.supports(context))
            .flatMap(rule -> rule.validate(context, doc).stream())
            .toList();

        List<ValidationIssue> unsuppressed =
            issueSuppressor.apply(context, doc, raw);

        List<ValidationIssue> resolved =
            conflictResolver.resolve(unsuppressed);

        List<ValidationIssue> sorted = resolved.stream()
            .sorted(
                Comparator.comparingInt(ValidationIssue::start)
                    .thenComparingInt(ValidationIssue::end)
                    .thenComparing(ValidationIssue::ruleId)
            )
            .toList();

        boolean hasErrors = sorted.stream()
            .anyMatch(i -> i.severity() == Severity.ERROR);

        boolean hasWarnings = sorted.stream()
            .anyMatch(i -> i.severity() == Severity.WARNING);

        return new ValidationResult(
            !hasErrors,
            hasErrors,
            hasWarnings,
            sorted
        );
    }
}
```

## 9.1 Rule order

Rule correctness không được phụ thuộc execution order.

Priority chỉ dùng conflict resolution.

---

# 10. RULE IDS — ĐỊNH NGHĨA CỐ ĐỊNH

```java
public final class RuleIds {

    public static final String ACCENT_CHARACTER_IN_NON_ACCENT_MODE =
        "ACCENT_CHARACTER_IN_NON_ACCENT_MODE";

    public static final String POSSIBLE_MISSING_DIACRITIC =
        "POSSIBLE_MISSING_DIACRITIC";

    public static final String ABBREVIATION_DETECTED =
        "ABBREVIATION_DETECTED";

    public static final String LEADING_WHITESPACE =
        "LEADING_WHITESPACE";

    public static final String TRAILING_WHITESPACE =
        "TRAILING_WHITESPACE";

    public static final String MULTIPLE_WHITESPACE =
        "MULTIPLE_WHITESPACE";

    public static final String WHITESPACE_BEFORE_PUNCTUATION =
        "WHITESPACE_BEFORE_PUNCTUATION";

    public static final String REPEATED_PUNCTUATION =
        "REPEATED_PUNCTUATION";

    public static final String MISSING_WHITESPACE_AFTER_PUNCTUATION =
        "MISSING_WHITESPACE_AFTER_PUNCTUATION";

    public static final String INVALID_CHARACTER =
        "INVALID_CHARACTER";

    public static final String ZERO_WIDTH_CHARACTER =
        "ZERO_WIDTH_CHARACTER";

    public static final String NON_BREAKING_SPACE =
        "NON_BREAKING_SPACE";

    public static final String POSSIBLE_SPELLING_ERROR =
        "POSSIBLE_SPELLING_ERROR";

    private RuleIds() {}
}
```

Không rename ID sau khi frontend đã tích hợp.

---

# 11. DETERMINISTIC RULES — IMPLEMENT TỪNG RULE

---

## 11.1 `ACCENT_CHARACTER_IN_NON_ACCENT_MODE`

### supports

```java
context.messageMode() == MessageMode.NON_ACCENTED
```

### Severity

`ERROR`.

### Algorithm

Scan original string theo code point.

Một character bị xem là Vietnamese accented character nếu:

- là chữ cái tiếng Việt có dấu/tone;
- hoặc `đ/Đ`.

Không flag ký tự Latin foreign chỉ vì Unicode.

### Output granularity

Khuyến nghị **group contiguous accented letters theo token**, để UI thấy `"quý"` thay vì 1 issue chỉ `"ý"`.

Ví dụ:

```text
Chuc mung sinh nhat quý khach
                   ^^^
```

Issue value = `quý`.

### Tests

Should warn:

```text
Chuc mung quý khach
Xin chào
Đang ky ngay
```

Should not warn:

```text
Chuc mung quy khach
SMS OTP API
https://example.com
```

### DoD

- đúng UTF-16 offsets;
- không mutate;
- unit tests >= 10 cases.

---

## 11.2 `LEADING_WHITESPACE`

### Severity

`WARNING`.

### Algorithm

Từ index 0, lấy run whitespace liên tục.

Không chỉ ASCII space; policy cần quyết định:

- `' '` space;
- `\t`;
- newline.

Baseline:

- whitespace Unicode ở đầu => issue;
- nếu character là NBSP thì `NON_BREAKING_SPACE` có priority cao hơn và conflict resolver có thể giữ specific rule.

### Test

```text
" Xin chào"
"   Xin chào"
"\tXin chào"
```

No warning:

```text
"Xin chào"
""
```

---

## 11.3 `TRAILING_WHITESPACE`

Tương tự leading nhưng scan từ end.

Output contiguous run.

---

## 11.4 `MULTIPLE_WHITESPACE`

### Definition baseline

Hai hoặc nhiều **ASCII spaces** liên tiếp trong body.

Không bắt newline + indentation trừ khi business quyết định.

Regex:

```regex
 {2,}
```

Không dùng `\s{2,}` ngay vì có thể gom newline/tab ngoài ý muốn.

### Severity

`WARNING`.

### Tests

Warn:

```text
Xin chào  quý khách
Xin chào     quý khách
```

No warning:

```text
Xin chào quý khách
```

Nếu protected placeholder có spaces nội bộ thì suppression layer xử lý.

---

## 11.5 `WHITESPACE_BEFORE_PUNCTUATION`

Punctuation set baseline:

```text
. , ! ? : ;
```

Detect contiguous whitespace ngay trước punctuation.

Ví dụ:

```text
Xin chào !
        ^
```

Issue value nên là whitespace, không gồm punctuation.

Severity `WARNING`.

No warning:

```text
Xin chào!
https://example.com
```

---

## 11.6 `REPEATED_PUNCTUATION`

Baseline:

Warn khi:

```text
!!+
??+
,,+
;;+
::+
```

Dấu `...`:

- mặc định allow đúng `...`;
- `....` trở lên warning;
- config cho phép thay đổi.

Không group mixed punctuation ở Phase 1 nếu chưa có business definition:

```text
?!
!?
```

có thể để INFO sau này, không tự đoán.

Severity `WARNING`.

Tests:

Warn:

```text
Khuyến mãi!!!!!
Đồng ý?????
Hello....
```

No warning:

```text
Xin chào!
...
https://a.vn/...
```

URL span suppress.

---

## 11.7 `MISSING_WHITESPACE_AFTER_PUNCTUATION`

### Punctuation baseline

```text
. , ! ? : ;
```

### Detect

Punctuation tại index `i`, sau nó có character `i+1`, và:

- char sau không whitespace;
- char sau không punctuation hợp lệ trong repeated/ellipsis;
- match không nằm trong protected range;
- punctuation không phải decimal/date/domain separator đã protected.

Ví dụ:

```text
Xin chào.Quý khách
        ^
```

Issue có thể value `"."` hoặc span `".Q"`.

Khuyến nghị value = `"."`, message giải thích thiếu space sau dấu.

### Tests bắt buộc

Warn:

```text
Xin chào.Quý khách
A,B
Xin chào!Quý khách
```

No warning:

```text
https://example.com
example.com
10.000
1,5
12/10/2026
A.B.C
...
```

`A.B.C` cần protected code/initialism detection hoặc punctuation exception.

---

## 11.8 `ZERO_WIDTH_CHARACTER`

Detect:

- U+200B ZERO WIDTH SPACE.
- U+200C ZERO WIDTH NON-JOINER.
- U+200D ZERO WIDTH JOINER.
- U+FEFF ZERO WIDTH NO-BREAK SPACE/BOM khi xuất hiện trong body.

Severity:

- `WARNING` mặc định;
- config cho phép `ERROR`.

Message phải hiển thị code point vì `value` có thể invisible.

Ví dụ:

```text
"Phát hiện ký tự ẩn U+200B (ZERO WIDTH SPACE)."
```

---

## 11.9 `NON_BREAKING_SPACE`

Detect U+00A0.

Severity `WARNING`.

Không tự replace bằng ASCII space.

Suggestion có thể là `" "` nhưng frontend chỉ hiển thị.

---

## 11.10 `INVALID_CHARACTER`

Phải có **allow policy**, không làm kiểu "mọi non-ASCII là invalid".

Define:

```java
CharacterPolicy
```

Allowed baseline:

- letters;
- digits;
- whitespace hợp lệ;
- punctuation hợp lệ;
- currency symbols được business chấp nhận;
- URL characters nếu trong protected URL;
- Vietnamese characters;
- configured allowed symbols.

Không tự cấm emoji nếu requirement chưa chốt.

### Agent instruction

Tạo config:

```yaml
sms-validation:
  characters:
    emoji-policy: ALLOW
```

Allowed:

```text
ALLOW
INFO
ERROR
```

Default `ALLOW` nếu business chưa quyết định.

---

## 11.11 `ABBREVIATION_DETECTED`

### supports

Không hardcode toàn bộ business trong rule.

```java
return abbreviationService.isRuleEnabledFor(
    context.brandname()
);
```

Seed config ban đầu:

```text
VT_TENDOO => enabled
others => disabled
```

### Lookup

Token normalize case theo policy.

Abbreviation entries:

```text
KH
SDT
ĐT
TT
KM
DV
CT
```

phải load config/DB.

### Severity

`WARNING`.

### Boundary

Không match substring.

Sai:

```text
KH trong KHACH
```

Đúng:

```text
token == "KH"
```

### Test

With brandname `VT_TENDOO`:

```text
Quy KH vui long kiem tra SDT
```

=> 2 issues.

With brandname `ABC_BANK`:

=> 0 abbreviation issues.

---

# 12. CONFIG & DATA MODEL

Nếu hiện tại project đã có config platform, reuse.

Nếu chưa, baseline SQL:

## 12.1 `sms_validation_rule_config`

```sql
CREATE TABLE sms_validation_rule_config (
    id BIGINT PRIMARY KEY,
    rule_id VARCHAR(128) NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    severity VARCHAR(16) NOT NULL,
    brandname VARCHAR(128) NULL,
    customer_id VARCHAR(128) NULL,
    config_json TEXT NULL,
    version BIGINT NOT NULL DEFAULT 1,
    updated_at TIMESTAMP NOT NULL
);
```

Unique key theo scope phù hợp hệ thống.

## 12.2 `sms_validation_abbreviation`

```sql
CREATE TABLE sms_validation_abbreviation (
    id BIGINT PRIMARY KEY,
    value VARCHAR(64) NOT NULL,
    brandname VARCHAR(128) NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    description VARCHAR(255) NULL,
    updated_at TIMESTAMP NOT NULL
);
```

## 12.3 `sms_validation_whitelist`

```sql
CREATE TABLE sms_validation_whitelist (
    id BIGINT PRIMARY KEY,
    value VARCHAR(255) NOT NULL,
    scope_type VARCHAR(32) NOT NULL,
    scope_value VARCHAR(128) NULL,
    term_type VARCHAR(32) NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMP NOT NULL
);
```

`scope_type`:

```text
SYSTEM
BRAND
CUSTOMER
```

`term_type`:

```text
BRAND
PERSON
TECHNICAL
FOREIGN
PRODUCT
OTHER
```

## 12.4 Lexicon

Nếu lexicon lớn, có thể không cần SQL online lookup.

Build artifact:

```text
lexicon.tsv
word<TAB>frequency<TAB>type
```

Load memory startup.

---

# 13. CONFIG SNAPSHOT — KHÔNG QUERY DB TRONG RULE

Sai:

```java
for (SmsToken token : tokens) {
    repository.findByWord(token.normalized());
}
```

Đúng:

```text
DB/config source
    │
    ▼
ValidationConfigService
    │
    ▼
immutable ValidationConfigSnapshot
    │
    ├── HashSet whitelist
    ├── HashSet abbreviation
    ├── rule settings
    └── character policy
```

Rule dùng snapshot/read-only services.

Reload:

- scheduled every N seconds; hoặc
- application event; hoặc
- admin endpoint internal.

Không yêu cầu restart khi thêm whitelist/abbreviation.

---

# 14. LEXICAL GATE — THỨ TỰ CỐ ĐỊNH

Mọi WORD token đi qua đúng thứ tự:

```text
1. token nằm hoàn toàn trong ProtectedRange?
       YES => skip linguistic checks

2. exact/normalized whitelist?
       YES => skip

3. allowed system/brand/customer term?
       YES => skip

4. token pattern giống product/transaction code?
       YES => skip

5. abbreviation rule đã xử lý token?
       YES => không tạo UNKNOWN_WORD thêm

6. token có trong Vietnamese lexicon?
       YES => valid, skip

7. token ASCII/unaccented và AccentIndex có candidates?
       YES => missing-diacritic pipeline

8. otherwise
       => typo pipeline

9. nếu candidate confidence thấp
       => no issue
```

**Không được đổi order nếu không có benchmark chứng minh tốt hơn.**

---

# 15. LEXICON SERVICE

```java
public interface LexiconService {

    boolean contains(String normalizedWord);

    long frequency(String normalizedWord);

    Optional<Lexeme> find(String normalizedWord);
}
```

Implementation:

```java
Map<String, Lexeme>
```

load startup.

Normalize key:

```text
NFC
lowercase Locale.ROOT
```

Không strip dấu cho primary dictionary key.

---

# 16. ACCENT REVERSE INDEX

## 16.1 Data structure

```java
public final class AccentIndex {

    private final Map<String, List<AccentCandidateEntry>> index;
}
```

Build:

```java
for each lexeme in Vietnamese lexicon:
    String stripped = stripVietnameseDiacritics(
        lexeme.word()
    ).toLowerCase(Locale.ROOT);

    index[stripped].add(
        new AccentCandidateEntry(
            lexeme.word(),
            lexeme.frequency()
        )
    );
```

Sort each list descending frequency.

## 16.2 Example

Lexicon:

```text
chào    1000000
cháo      50000
chảo      30000
chạo        100
```

Index:

```text
chao => [
    chào,
    cháo,
    chảo,
    chạo
]
```

## 16.3 Candidate cap

Để tránh explosion:

```text
MAX_ACCENT_CANDIDATES_PER_TOKEN = 12
```

Giữ top frequency.

Configurable.

## 16.4 Rule: `POSSIBLE_MISSING_DIACRITIC`

### Precondition

- messageMode == `ACCENTED`;
- token không trong protected range;
- token không whitelist;
- token không dictionary-valid;
- token có accent-index candidates;
- token đủ dài; mặc định >= 2;
- không phải all-uppercase technical/code token.

### Không cảnh báo chỉ vì có candidates

Bắt buộc qua contextual ranker + confidence gate.

---

# 17. N-GRAM LANGUAGE MODEL

## 17.1 Mục tiêu

Rank candidate theo context SMS.

Không cần generative model.

## 17.2 Data

Production ưu tiên:

1. Approved historical SMS đã được phép sử dụng.
2. Approved Vietnamese corpus.
3. Brand/business phrases.
4. Không đưa user secret/PII raw vào artifact model nếu chưa anonymize.

## 17.3 Training pipeline

Offline job:

```text
raw corpus
    │
    ▼
privacy cleanup
    │
    ▼
NFC normalize
    │
    ▼
tokenize
    │
    ▼
lowercase lookup form
    │
    ▼
count unigram
count bigram
count trigram
    │
    ▼
prune
    │
    ▼
assign integer word IDs
    │
    ▼
write binary/compact model
```

## 17.4 Counts

Need:

```java
long unigramCount(wordId);
long bigramCount(prevId, wordId);
long trigramCount(prev2Id, prevId, wordId);
long vocabularySize();
long totalTokens();
```

## 17.5 Scoring

Start simple and explainable.

Use log probability with backoff.

Pseudo:

```text
score(w_i | context) =
    if trigram count sufficient:
        log P(w_i | w_i-2, w_i-1)
    else if bigram count sufficient:
        log P(w_i | w_i-1)
    else:
        log P(w_i)
```

Smoothing baseline:

```text
add-k / simple backoff
```

POC không cần Kneser-Ney trước.

Nếu quality chưa đủ mới thử Modified Kneser-Ney.

## 17.6 Bidirectional local context

Để rank một token đơn:

```text
score(candidate) =
    A * logP(candidate)
  + B * logP(candidate | prev)
  + C * logP(next | candidate)
  + D * trigram terms if available
```

Initial weights:

```text
A = 0.2
B = 1.0
C = 1.0
D = 1.2
```

**Đây chỉ là seed POC. Phải tune bằng labeled corpus.**

Không hardcode coi đây là business truth.

---

# 18. SEQUENCE DECODING — NHIỀU TỪ THIẾU DẤU

Ví dụ:

```text
Quy khach vui long nhap thong tin
```

Nếu mỗi token có candidates, không rank từng token độc lập.

## 18.1 Candidate lattice

```text
Quy       khach     vui      long      nhap
│         │         │        │         │
Quý       khách     vui      lòng      nhập
Quỳ                 vùi      lông      nháp
Quỷ
...
```

## 18.2 Beam Search

Config:

```text
BEAM_WIDTH = 5
MAX_CANDIDATES_PER_TOKEN = 12
```

Pseudocode:

```java
List<BeamState> beam = List.of(initialState());

for tokenPosition in sequence:

    List<Candidate> candidates =
        candidatesFor(tokenPosition);

    PriorityQueue<BeamState> next =
        new bounded top-k queue(BEAM_WIDTH);

    for state in beam:
        for candidate in candidates:
            double incremental =
                languageModel.scoreTransition(
                    state,
                    candidate
                );

            next.offer(
                state.append(candidate, incremental)
            );

    beam = next.bestDescending();

return beam;
```

Không enumerate Cartesian product.

## 18.3 Preserve original token option

Candidate list phải luôn có original token với một score/prior.

Mục đích:

- engine có thể kết luận "giữ nguyên" tốt hơn;
- không ép mọi ASCII token thành tiếng Việt có dấu.

---

# 19. CONFIDENCE POLICY CHO MISSING DIACRITIC

Không trực tiếp convert raw n-gram score sang probability nếu chưa calibrate.

POC có thể dùng rank margin:

```text
top1 = best candidate sentence
top2 = second-best candidate sentence

margin = normalized(top1Score - top2Score)
```

Rule emit khi:

1. best candidate khác original;
2. best candidate chứa accented form;
3. `confidence >= MIN_CONFIDENCE`;
4. `margin >= MIN_MARGIN`.

Seed config:

```yaml
linguistic:
  missing-diacritic:
    min-confidence: 0.85
    min-margin: 0.20
```

Các số này bắt buộc tune.

Case:

```text
Kính chao quý khách

chào = 0.95
cháo = 0.02
original = 0.01
```

=> warning.

Case:

```text
long

lòng = 0.39
lông = 0.34
long = 0.22
```

=> **NO WARNING**.

---

# 20. TYPO PIPELINE

Chỉ chạy nếu:

- không protected;
- không whitelist;
- không lexicon valid;
- không được missing-accent pipeline xử lý tự tin.

## 20.1 `TypoCandidateProvider`

```java
public interface TypoCandidateProvider {

    List<TypoCandidate> candidates(
        String normalizedToken,
        int maxEditDistance,
        int limit
    );
}
```

## 20.2 Dynamic edit distance

Baseline:

```text
length <= 3  => maxDistance 1
length 4..7  => maxDistance 1
length >= 8  => maxDistance 2
```

Không dùng edit distance 2 cho mọi từ ngắn.

## 20.3 SymSpell-style

Dùng delete index để retrieve nhanh.

Nếu external library:

- adapter nằm trong `SymSpellCandidateProvider`;
- không leak library types ra core.

Nếu thay library sau này, rule không đổi.

---

# 21. VIETNAMESE EDIT SCORER

SymSpell retrieve candidates; final ranker của hệ thống quyết định.

```java
public interface EditScorer {
    double score(String input, String candidate);
}
```

Có thể dùng weighted Damerau-Levenshtein.

Operations:

- insertion;
- deletion;
- substitution;
- transposition.

Thêm Vietnamese keyboard/typing hints chỉ để rank.

Ví dụ mappings:

```text
dd ↔ đ
aw ↔ ă
aa ↔ â
ee ↔ ê
oo ↔ ô
ow ↔ ơ
uw ↔ ư
```

Các confusion như:

```text
s/x
ch/tr
d/gi/r
l/n
```

**không được cho weight mạnh mặc định** vì dễ false positive.

Chỉ bật sau corpus evaluation.

---

# 22. FINAL TYPO RANKING

```text
finalScore(candidate) =
    - alpha * editCost
    + beta * logWordFrequency
    + gamma * ngramContextScore
```

Seed:

```text
alpha = 1.0
beta  = 0.3
gamma = 1.2
```

Tune sau.

Rule `POSSIBLE_SPELLING_ERROR` chỉ emit khi:

- best candidate khác input;
- edit distance/cost trong threshold;
- frequency đủ;
- context score đủ;
- confidence/margin pass.

Suggestions max 3.

---

# 23. WHITELIST SERVICE

Lookup order:

```text
CUSTOMER
BRAND
SYSTEM
```

Không phân biệt case cho technical/brand term trừ config đặc biệt.

API:

```java
public interface WhitelistService {

    boolean isAllowed(
        String token,
        ValidationContext context
    );
}
```

Cache in-memory.

Examples seed:

```text
SMS
OTP
API
URL
Viettel
Tendoo
iPhone
Samsung
online
mobile
voucher
```

Seed chỉ để POC; production data phải business approve.

---

# 24. ISSUE SUPPRESSION

## 24.1 Interface

```java
public interface IssueSuppressor {

    List<ValidationIssue> apply(
        ValidationContext context,
        ValidationDocument document,
        List<ValidationIssue> issues
    );
}
```

## 24.2 ProtectedRange suppression

Rules thường suppress nếu issue nằm trong protected range:

```text
MISSING_WHITESPACE_AFTER_PUNCTUATION
REPEATED_PUNCTUATION
POSSIBLE_MISSING_DIACRITIC
POSSIBLE_SPELLING_ERROR
```

Rules **không** suppress chỉ vì protected:

```text
ZERO_WIDTH_CHARACTER
INVALID_CHARACTER
```

vì ký tự nguy hiểm trong URL vẫn có thể cần cảnh báo.

Tạo map policy:

```java
Map<String, Set<ProtectedRangeType>>
```

không hardcode `if` khắp nơi.

---

# 25. ISSUE CONFLICT RESOLUTION

## 25.1 Priority đề xuất

Specific > generic.

```text
ZERO_WIDTH_CHARACTER              1000
INVALID_CHARACTER                  950
ACCENT_CHARACTER_IN_NON_ACCENT     900
ABBREVIATION_DETECTED              800
POSSIBLE_MISSING_DIACRITIC         700
POSSIBLE_SPELLING_ERROR            600
NON_BREAKING_SPACE                 550
WHITESPACE_*                       400
PUNCTUATION_*                      300
```

## 25.2 Conflict examples

Token `SDT`:

```text
ABBREVIATION_DETECTED
POSSIBLE_SPELLING_ERROR
```

=> giữ abbreviation.

Token `chao`:

```text
POSSIBLE_MISSING_DIACRITIC
POSSIBLE_SPELLING_ERROR
```

=> giữ missing diacritic.

NBSP:

```text
NON_BREAKING_SPACE
MULTIPLE_WHITESPACE
```

=> ưu tiên specific NBSP nếu cùng span.

## 25.3 Không merge unrelated issues

Ví dụ:

```text
"  quý"
```

có leading whitespace + accent-in-non-accent.

Giữ cả hai vì span khác / semantics khác.

---

# 26. REST API

Endpoint gợi ý:

```http
POST /api/v1/sms/content/validate
Content-Type: application/json
```

Request:

```json
{
  "content": "Kính chao quý khách !",
  "messageMode": "ACCENTED",
  "brandname": "VT_TENDOO",
  "customerId": "optional"
}
```

Response:

```json
{
  "valid": true,
  "hasErrors": false,
  "hasWarnings": true,
  "issues": [
    {
      "ruleId": "POSSIBLE_MISSING_DIACRITIC",
      "severity": "WARNING",
      "start": 5,
      "end": 9,
      "value": "chao",
      "message": "Từ \"chao\" có thể đang thiếu dấu.",
      "suggestions": ["chào"],
      "confidence": 0.96
    },
    {
      "ruleId": "WHITESPACE_BEFORE_PUNCTUATION",
      "severity": "WARNING",
      "start": 19,
      "end": 20,
      "value": " ",
      "message": "Phát hiện khoảng trắng trước dấu câu.",
      "suggestions": [],
      "confidence": null
    }
  ]
}
```

## API rules

- Never return corrected full SMS.
- Never mutate request.
- `suggestions` optional semantically, but serialize as `[]` for stable frontend.
- `confidence` null cho deterministic rule.
- validate request length theo SMS/campaign platform constraints, nhưng length policy có thể là rule riêng.

---

# 27. ERROR HANDLING

Invalid request:

```text
content = null
messageMode = null
```

=> HTTP 400.

Empty content:

- Nếu campaign layer đã validate required content thì engine có thể return no issues.
- Nếu engine chịu trách nhiệm required-content, tạo rule riêng `EMPTY_CONTENT`.
- Không tự thêm rule này nếu business requirement chưa nói.

Internal dictionary/model load failure:

- service startup fail-fast nếu linguistic feature được configured required;
- hoặc disable feature + health warning nếu optional.

Không silently run half-loaded model mà không metric/log.

---

# 28. THREAD SAFETY & PERFORMANCE

Toàn bộ runtime model phải immutable/read-only:

```text
ValidationConfigSnapshot
Lexicon map
AccentIndex
NGramLanguageModel
SymSpell index
```

Có thể share giữa threads.

Request path không:

- write model;
- query DB per token;
- load file;
- rebuild index;
- call network.

Target POC:

```text
p95 < 50 ms per normal SMS
```

Deterministic-only nên thấp hơn đáng kể.

Benchmark riêng:

- p50;
- p95;
- p99;
- allocations;
- heap model size.

---

# 29. METRICS

Không log raw SMS mặc định.

Metrics:

```text
sms_validation_requests_total
sms_validation_latency_ms
sms_validation_issues_total{ruleId,severity}
sms_validation_suppressed_total{ruleId,reason}
sms_validation_linguistic_skipped_low_confidence_total
sms_validation_dictionary_reload_total
sms_validation_model_load_seconds
```

Nếu cần debug raw content:

- gated secure debug;
- redact PII;
- không để production log mặc định.

---

# 30. TEST STRATEGY — BẮT BUỘC

## 30.1 Unit test per rule

Mỗi rule có:

```text
shouldWarn
shouldNotWarn
shouldReturnCorrectOffsets
shouldNotMutateInput
```

## 30.2 Global offset invariant

Parameterized integration test:

```java
for (ValidationIssue issue : result.issues()) {
    assertThat(
        input.substring(issue.start(), issue.end())
    ).isEqualTo(issue.value());
}
```

## 30.3 Regression corpus

File:

```text
src/test/resources/corpus/
├── deterministic-positive.jsonl
├── deterministic-negative.jsonl
├── linguistic-positive.jsonl
├── linguistic-negative.jsonl
└── edge-cases.jsonl
```

Format:

```json
{"text":"Xin chào  quý khách","expectedRules":["MULTIPLE_WHITESPACE"]}
```

Negative:

```json
{"text":"https://example.com","forbiddenRules":["MISSING_WHITESPACE_AFTER_PUNCTUATION"]}
```

## 30.4 Required edge cases

At least:

```text
""
" "
"  "
"\tXin chào"
"Xin chào "
"Xin  chào"
"Xin chào !"
"Xin chào.Quý khách"
"https://example.com"
"example.com"
"abc@example.com"
"10.000"
"1,5"
"12/10/2026"
"23:59"
"A.B.C"
"ABC-2026"
"DH123456"
"Khuyến mãi!!!!!"
"..."
"...."
"Xin chào\u200Bquý khách"
"Xin\u00A0chào"
"Chuc mung quý khach"
"Kính chao quý khách"
"Quy khach vui long nhap thong tin"
"Quy KH vui long kiem tra SDT"
```

---

# 31. LABELED BENCHMARK CORPUS

Trước khi bật linguistic rules production, tạo ít nhất:

```text
500–1000 SMS
```

Nếu có dữ liệu thật, sample theo distribution.

Labels:

```text
ruleId
start
end
expected suggestion
shouldWarn true/false
```

Categories:

```text
normal clean SMS
whitespace
punctuation
URLs
phones
dates
transaction codes
brand names
technical words
foreign terms
missing diacritics
typos
names
mixed accented/unaccented
```

Measure:

```text
precision
recall
false positive rate
latency p50/p95
```

### Gate production

Deterministic:

```text
Precision >= 99%
```

Linguistic:

Không hardcode metric cuối cùng trước benchmark, nhưng ưu tiên:

```text
Precision >= 95%
```

và false-positive rate phải được product owner chấp nhận.

Nếu không đạt:

```text
DO NOT ENABLE RULE
```

---

# 32. TEST EXAMPLES CHI TIẾT

## Case A — non-accent mode

Input:

```text
mode = NON_ACCENTED
text = "Chuc mung sinh nhat quý khach"
```

Expected:

```text
ACCENT_CHARACTER_IN_NON_ACCENT_MODE
value = "quý"
severity = ERROR
```

No auto-replace.

## Case B — accented mode missing accent

```text
mode = ACCENTED
text = "Kính chao quý khách"
```

If confidence high:

```text
POSSIBLE_MISSING_DIACRITIC
value = "chao"
suggestion = "chào"
WARNING
```

## Case C — abbreviation

```text
brand = VT_TENDOO
text = "Quy KH vui long kiem tra SDT"
```

Expected:

```text
KH
SDT
```

2 abbreviation issues.

## Case D — customer brand

```text
brand = ABC_BANK
same text
```

No abbreviation issue from VT_TENDOO rule.

## Case E — punctuation protected URL

```text
text = "Truy cap https://example.com ngay."
```

No missing-space issue inside URL.

## Case F — numeric

```text
text = "So tien 10.000 dong."
```

No punctuation-space issue at decimal/thousand separator.

## Case G — ambiguity

```text
text = "long"
```

If candidate score ambiguous:

```text
no linguistic issue
```

This case is mandatory to enforce precision-first.

---

# 33. PHASED IMPLEMENTATION — AGENT PHẢI LÀM THEO THỨ TỰ

---

## PHASE 0 — BOOTSTRAP

### Tasks

- [ ] Inspect existing project build/runtime conventions.
- [ ] Create package structure.
- [ ] Create enums and core records.
- [ ] Create `SmsValidationRule`.
- [ ] Create empty engine.
- [ ] Create endpoint/request/response.
- [ ] Add base unit test infrastructure.
- [ ] Add offset invariant helper.

### Must pass

```bash
mvn test
```

hoặc project-equivalent.

### Commit checkpoint

```text
feat(sms-validation): add validation core contracts
```

Không làm rule trước khi compile pass.

---

## PHASE 1 — PREPROCESSING & PROTECTED RANGES

### Tasks

- [ ] Implement `VietnameseTextNormalizer`.
- [ ] Implement `AccentStripper`.
- [ ] Implement tokenizer.
- [ ] Implement `ProtectedRange` model.
- [ ] Implement URL detector.
- [ ] Email detector.
- [ ] Phone detector.
- [ ] Date/time detector.
- [ ] Numeric detector.
- [ ] Code detector.
- [ ] Placeholder hook.
- [ ] Composite overlap resolution.
- [ ] Build `ValidationDocument`.

### Tests

Minimum 30 detector tests.

### Must pass

- exact offsets;
- no detector throws;
- overlapping ranges deterministic.

### Commit

```text
feat(sms-validation): add preprocessing and protected ranges
```

---

## PHASE 2 — DETERMINISTIC RULES

Implement one rule at a time.

Order:

1. `LEADING_WHITESPACE`
2. `TRAILING_WHITESPACE`
3. `MULTIPLE_WHITESPACE`
4. `WHITESPACE_BEFORE_PUNCTUATION`
5. `REPEATED_PUNCTUATION`
6. `MISSING_WHITESPACE_AFTER_PUNCTUATION`
7. `ZERO_WIDTH_CHARACTER`
8. `NON_BREAKING_SPACE`
9. `ACCENT_CHARACTER_IN_NON_ACCENT_MODE`
10. `INVALID_CHARACTER`
11. `ABBREVIATION_DETECTED`

After each rule:

- [ ] positive tests;
- [ ] negative tests;
- [ ] offset test;
- [ ] integration corpus update.

### Commit strategy

Có thể commit từng nhóm:

```text
feat(sms-validation): add whitespace rules
feat(sms-validation): add punctuation rules
feat(sms-validation): add unicode rules
feat(sms-validation): add message-mode validation
feat(sms-validation): add abbreviation validation
```

---

## PHASE 3 — CONFIG SNAPSHOT

### Tasks

- [ ] schema/migration;
- [ ] repository;
- [ ] config load;
- [ ] immutable snapshot;
- [ ] whitelist service;
- [ ] abbreviation service;
- [ ] refresh mechanism;
- [ ] tests for config refresh;
- [ ] prove no per-token DB lookup.

### Commit

```text
feat(sms-validation): add configurable rule dictionaries
```

---

## PHASE 4 — SUPPRESSION & CONFLICT RESOLUTION

### Tasks

- [ ] protected-range suppressor;
- [ ] whitelist suppressor;
- [ ] conflict priority table;
- [ ] duplicate issue removal;
- [ ] stable sorting;
- [ ] tests.

### Critical tests

```text
URL dot => no punctuation issue
SDT => abbreviation, not spelling
chao => missing-diacritic, not spelling duplicate
NBSP => no duplicate generic whitespace if same span
```

### Commit

```text
feat(sms-validation): add issue suppression and conflict resolution
```

---

## PHASE 5 — LEXICON & ACCENT INDEX

### Tasks

- [ ] define approved POC lexicon source;
- [ ] license note;
- [ ] lexicon parser;
- [ ] frequency support;
- [ ] `LexiconService`;
- [ ] build `AccentIndex`;
- [ ] candidate caps;
- [ ] unit tests.

### Must prove

```text
chao => candidate chào
khach => candidate khách
dang => candidates include đăng/đang/... if lexicon contains
```

No warning yet. Only infrastructure.

### Commit

```text
feat(sms-validation): add Vietnamese lexicon and accent index
```

---

## PHASE 6 — N-GRAM MODEL

### Offline tools

Create separate package/module:

```text
tools/ngram-builder
```

Input:

```text
UTF-8 text corpus
```

Output:

```text
vocabulary
unigram
bigram
pruned trigram
metadata
```

Metadata:

```json
{
  "version": "2026-01",
  "createdAt": "...",
  "sourceDescription": "...",
  "vocabSize": 0,
  "unigramCount": 0,
  "bigramCount": 0,
  "trigramCount": 0
}
```

### Runtime

- [ ] load once;
- [ ] immutable;
- [ ] score API;
- [ ] unknown-word fallback;
- [ ] unit tests with tiny deterministic corpus.

### Tiny corpus test

Train:

```text
kính chào quý khách
kính chào quý khách
cháo ngon
```

Assert:

```text
score("chào" in "kính _ quý") >
score("cháo" in "kính _ quý")
```

### Commit

```text
feat(sms-validation): add local ngram language model
```

---

## PHASE 7 — MISSING DIACRITIC RULE

### Tasks

- [ ] lexical gate;
- [ ] candidate construction;
- [ ] preserve original candidate;
- [ ] single-token contextual ranking;
- [ ] beam search for sequences;
- [ ] confidence policy;
- [ ] rule output;
- [ ] suggestions;
- [ ] confidence field;
- [ ] regression tests.

### Never

- auto-replace;
- warning on every unaccented token;
- flag whitelisted English/brand/code;
- output when ambiguity is high.

### Commit

```text
feat(sms-validation): add contextual missing-diacritic detection
```

---

## PHASE 8 — TYPO CANDIDATE PROVIDER

### Tasks

- [ ] create interface first;
- [ ] choose stable SymSpell dependency or implement internal adapter;
- [ ] load lexicon frequency;
- [ ] dynamic edit distance;
- [ ] candidate top-K;
- [ ] no final decision inside provider.

### Commit

```text
feat(sms-validation): add typo candidate retrieval
```

---

## PHASE 9 — SPELLING RULE

### Tasks

- [ ] weighted edit scorer;
- [ ] contextual reranker;
- [ ] confidence;
- [ ] conflict with missing-diacritic;
- [ ] suggestions max 3;
- [ ] corpus benchmark.

### Commit

```text
feat(sms-validation): add precision-first spelling warnings
```

---

## PHASE 10 — BENCHMARK & FEATURE FLAGS

Feature flags:

```yaml
sms-validation:
  rules:
    deterministic-enabled: true
    missing-diacritic-enabled: false
    spelling-enabled: false
```

Deploy deterministic first.

Enable linguistic only after benchmark approval.

### Commit

```text
feat(sms-validation): add linguistic feature flags and metrics
```

---

# 34. AGENT EXECUTION PROTOCOL

Coding agent **phải** tuân thủ:

## Before editing

1. Read this file fully.
2. Inspect repository structure.
3. Locate current SMS DTO/domain classes.
4. Locate current Brandname model.
5. Locate message-mode representation.
6. Locate DB migration convention.
7. Locate testing convention.
8. Do not duplicate existing abstractions unnecessarily.

## During each phase

1. Implement smallest coherent change.
2. Compile.
3. Run relevant unit tests.
4. Run full test suite if reasonable.
5. Fix failures before next phase.
6. Update checklist in this file or task tracker.
7. Never comment out failing tests.
8. Never weaken assertion to make tests pass.
9. Never silently remove requirement.

## When requirement is ambiguous

Do **not** invent high-impact behavior.

Examples requiring config/default-safe behavior:

- emoji allowed?
- newline allowed?
- mixed `?!` allowed?
- ERROR blocks send?
- exact template placeholder syntax?

Implement safe extension point and document assumption.

## Never

- call OpenAI/DeepSeek/Gemini/API;
- add vector DB;
- add Transformer;
- add Redis only for this feature unless existing architecture requires;
- query DB per token;
- mutate SMS;
- run spell check before whitelist/protected gate;
- use one gigantic regex for all punctuation exceptions;
- swallow exceptions silently;
- log raw SMS by default;
- add GPL data to production artifact without approval.

---

# 35. CODING STYLE REQUIREMENTS

## Rules

Rule class should be small.

Target:

```text
< 150 LOC per deterministic rule
```

If larger, extract helper.

## Regex

- precompile as `static final Pattern`;
- name it;
- comment only non-obvious semantics;
- add positive/negative tests.

## Immutable data

Prefer:

- records;
- `List.copyOf`;
- immutable maps after loading.

## No magic strings

Use:

- `RuleIds`;
- enums;
- typed config.

---

# 36. SAMPLE RULE IMPLEMENTATION

```java
@Component
public final class MultipleWhitespaceRule
        implements SmsValidationRule {

    private static final Pattern MULTIPLE_SPACES =
        Pattern.compile(" {2,}");

    @Override
    public String id() {
        return RuleIds.MULTIPLE_WHITESPACE;
    }

    @Override
    public int priority() {
        return 400;
    }

    @Override
    public boolean supports(ValidationContext context) {
        return true;
    }

    @Override
    public List<ValidationIssue> validate(
            ValidationContext context,
            ValidationDocument document) {

        String text = document.originalText();

        Matcher matcher = MULTIPLE_SPACES.matcher(text);

        List<ValidationIssue> issues =
            new ArrayList<>();

        while (matcher.find()) {
            issues.add(
                new ValidationIssue(
                    id(),
                    Severity.WARNING,
                    matcher.start(),
                    matcher.end(),
                    text.substring(
                        matcher.start(),
                        matcher.end()
                    ),
                    "Phát hiện nhiều khoảng trắng liên tiếp.",
                    List.of(" "),
                    null
                )
            );
        }

        return List.copyOf(issues);
    }
}
```

Note: Suggestion `" "` là optional; nếu UI không dùng thì có thể để empty.

---

# 37. SAMPLE MISSING DIACRITIC FLOW

```java
public List<ValidationIssue> validate(
        ValidationContext context,
        ValidationDocument doc) {

    if (context.messageMode() != MessageMode.ACCENTED) {
        return List.of();
    }

    List<SmsToken> candidates =
        lexicalGate.findPotentialMissingAccentTokens(
            context,
            doc
        );

    if (candidates.isEmpty()) {
        return List.of();
    }

    DecodingResult result =
        beamSearchDecoder.decode(
            doc.tokens(),
            candidates
        );

    return issueFactory.fromDecodingResult(
        context,
        doc,
        result
    );
}
```

`beamSearchDecoder` không biết HTTP, DB, severity.

---

# 38. FEATURE FLAGS

Minimum:

```yaml
sms-validation:
  enabled: true

  rules:
    accent-mode: true
    whitespace: true
    punctuation: true
    unicode: true
    abbreviation: true
    missing-diacritic: false
    spelling: false

  linguistic:
    max-accent-candidates-per-token: 12
    beam-width: 5
    missing-diacritic-min-confidence: 0.85
    missing-diacritic-min-margin: 0.20
    max-spelling-suggestions: 3
```

Production launch:

```text
deterministic = ON
missing-diacritic = OFF initially
spelling = OFF initially
```

Enable after shadow/benchmark.

---

# 39. SHADOW MODE CHO LINGUISTIC RULE

Highly recommended.

Mode:

```text
linguistic rules run
but do not return issue to user
```

Chỉ metric/sample review.

Config:

```yaml
sms-validation:
  linguistic:
    mode: SHADOW
```

Modes:

```text
OFF
SHADOW
ACTIVE
```

Shadow allows tune threshold without gây UX false-positive.

Raw SMS privacy policy phải được tuân thủ; metric có thể chỉ count aggregate.

---

# 40. ACCEPTANCE CRITERIA — PHASE 1

Engine được coi là Phase 1 complete khi:

- [ ] API hoạt động.
- [ ] Original content không bị mutate.
- [ ] Offset UTF-16 đúng.
- [ ] Protected URL/email/number/date/code hoạt động.
- [ ] Non-accent mode detect Vietnamese accents.
- [ ] Leading/trailing/multiple whitespace.
- [ ] Whitespace before punctuation.
- [ ] Missing whitespace after punctuation với exceptions.
- [ ] Repeated punctuation.
- [ ] Abbreviation only đúng Brandname config.
- [ ] Unicode hidden-space rules.
- [ ] Whitelist/config reload.
- [ ] No per-token DB calls.
- [ ] >= 99% deterministic precision trên corpus POC.
- [ ] p95 latency đạt target.
- [ ] Full unit/integration tests pass.

---

# 41. ACCEPTANCE CRITERIA — LINGUISTIC

Missing diacritic/spelling chỉ ACTIVE khi:

- [ ] lexicon source legally approved;
- [ ] n-gram model source documented;
- [ ] labeled corpus available;
- [ ] confidence thresholds calibrated;
- [ ] precision accepted;
- [ ] false positive reviewed manually;
- [ ] latency/heap acceptable;
- [ ] shadow mode stable;
- [ ] feature flag rollback works.

---

# 42. DEFINITION OF DONE CHO MỖI RULE

Một rule chưa "done" nếu thiếu bất kỳ mục nào:

- [ ] Stable Rule ID.
- [ ] Severity.
- [ ] `supports`.
- [ ] Implementation.
- [ ] Exact offset output.
- [ ] Message tiếng Việt.
- [ ] Positive tests.
- [ ] Negative tests.
- [ ] Protected-range tests nếu liên quan.
- [ ] Whitelist tests nếu liên quan.
- [ ] Conflict tests nếu liên quan.
- [ ] No mutation.
- [ ] Metrics.
- [ ] Config documented.

---

# 43. NHỮNG THỨ KHÔNG IMPLEMENT TRONG V1 TRỪ KHI CÓ YÊU CẦU MỚI

Không tự scope creep:

- grammar correction;
- rewrite câu cho "hay hơn";
- semantic toxicity;
- marketing compliance bằng LLM;
- NER;
- POS tagging;
- dependency parsing;
- Transformer;
- spell autocorrect;
- sentence paraphrase;
- automatic send blocking policy;
- SMS segment cost calculator GSM-7/UCS-2, trừ khi được bổ sung scope.

Có thể thiết kế extension point nhưng không code feature.

---

# 44. OPTIONAL FUTURE EXTENSIONS

Sau khi V1 ổn định:

## 44.1 SMS encoding validator

- GSM-7 vs UCS-2.
- segment count.
- unusual Unicode cost.
- business cost warning.

## 44.2 Brand-specific phrase rules

Ví dụ mandatory disclaimer hoặc forbidden phrase.

Implement dưới cùng `SmsValidationRule`.

## 44.3 Local NER

Chỉ khi tên riêng false positive quá cao.

## 44.4 Advanced LM

Nếu n-gram không đủ:

- quantized local model;
- ONNX;
- isolated behind `ContextScorer`.

Nhưng phải chứng minh improvement bằng benchmark trước.

---

# 45. FILES / ARTIFACTS NÊN TẠO THÊM TRONG REPO

Sau khi implementation bắt đầu:

```text
docs/sms-validation/
├── IMPLEMENTATION_PLAN.md
├── RULE_CATALOG.md
├── DATA_LICENSES.md
├── BENCHMARK.md
└── OPERATIONS.md
```

Tuy nhiên **file hiện tại là source-of-truth cho plan**.

`RULE_CATALOG.md` chỉ được tạo khi code bắt đầu để sync rule status.

---

# 46. RULE CATALOG INITIAL

| Rule ID | Severity | Phase | Deterministic | Protected-aware |
|---|---|---:|---:|---:|
| ACCENT_CHARACTER_IN_NON_ACCENT_MODE | ERROR | 1 | Yes | No |
| LEADING_WHITESPACE | WARNING | 1 | Yes | No |
| TRAILING_WHITESPACE | WARNING | 1 | Yes | No |
| MULTIPLE_WHITESPACE | WARNING | 1 | Yes | Partial |
| WHITESPACE_BEFORE_PUNCTUATION | WARNING | 1 | Yes | Yes |
| REPEATED_PUNCTUATION | WARNING | 1 | Yes | Yes |
| MISSING_WHITESPACE_AFTER_PUNCTUATION | WARNING | 1 | Yes | Yes |
| ZERO_WIDTH_CHARACTER | WARNING | 1 | Yes | No |
| NON_BREAKING_SPACE | WARNING | 1 | Yes | No |
| INVALID_CHARACTER | ERROR/config | 1 | Yes | Partial |
| ABBREVIATION_DETECTED | WARNING | 1 | Yes | Yes |
| POSSIBLE_MISSING_DIACRITIC | WARNING | 2 | Statistical | Yes |
| POSSIBLE_SPELLING_ERROR | WARNING | 2 | Statistical | Yes |

---

# 47. REVIEW CHECKLIST CHO PULL REQUEST

Reviewer kiểm:

### Architecture

- [ ] rule isolated?
- [ ] no network call?
- [ ] no per-token DB?
- [ ] original offsets preserved?
- [ ] config not hardcoded unnecessarily?

### Correctness

- [ ] protected ranges?
- [ ] Unicode edge?
- [ ] substring invariant?
- [ ] duplicate/conflict behavior?

### Tests

- [ ] positive?
- [ ] negative?
- [ ] real false-positive cases?
- [ ] regression corpus update?

### Linguistic

- [ ] whitelist before spell?
- [ ] ambiguity suppress?
- [ ] candidate count bounded?
- [ ] confidence threshold?
- [ ] original candidate retained?

---

# 48. DEBUGGING GUIDE

Nếu frontend highlight sai:

1. Log/request ID, không log full content mặc định.
2. Reproduce exact input in test.
3. Assert Java substring on issue range.
4. Check normalization offset drift.
5. Check surrogate pair/emoji before issue.
6. Ensure frontend uses JS UTF-16 indices directly.
7. Không "fix" bằng +1/-1 magic number.

Nếu punctuation false positive:

1. Check protected range.
2. Add protected detector or exception.
3. Add negative regression case.
4. Không mở rộng regex mù quáng.

Nếu spelling spam warning:

1. Check whitelist.
2. Check lexicon coverage.
3. Increase confidence.
4. Increase margin.
5. Check domain n-gram data.
6. Disable rule/shadow if precision below gate.

---

# 49. DATA & LICENSE CHECKLIST

Trước production:

- [ ] list every external dataset.
- [ ] repository URL.
- [ ] license.
- [ ] version/commit/hash.
- [ ] redistribution rights.
- [ ] derivative model obligations.
- [ ] organization legal approval if needed.

Do not assume "GitHub public" means production-safe.

---

# 50. FINAL IMPLEMENTATION PRINCIPLE

Bài toán này không phải "AI spelling assistant".

Nó là:

```text
SMS Content Linter
=
Deterministic Rule Engine
+
Configurable Business Dictionaries
+
Unicode-safe Span Tracking
+
Protected Range Suppression
+
Vietnamese Lexical Candidate Generation
+
Small Statistical Context Model
+
Strict Confidence Thresholds
```

Tối ưu cho:

```text
accuracy
low false positives
local execution
low latency
explainability
configurability
easy extension
```

Không tối ưu cho:

```text
creative rewriting
perfect grammatical correction
maximum recall bằng mọi giá
```

---

# 51. CODING AGENT — START HERE

Agent bắt đầu bằng đúng sequence:

```text
1. Inspect repo
2. Implement core contracts
3. Compile/test
4. Implement preprocessing/protected spans
5. Compile/test
6. Implement deterministic rules
7. Compile/test
8. Implement config snapshot
9. Compile/test
10. Implement suppression/conflict
11. Run deterministic benchmark
12. ONLY THEN implement lexicon/accent index
13. Implement n-gram
14. Add missing-diacritic shadow mode
15. Benchmark/tune
16. Add typo candidate provider
17. Add spelling shadow mode
18. Benchmark/tune
19. Enable only approved rules
```

Nếu một phase fail tests:

```text
STOP.
FIX.
DO NOT CONTINUE.
```

Nếu requirement thiếu:

```text
choose safe/non-destructive default
+
expose config/extension point
+
document assumption
```

Nếu linguistic confidence thấp:

```text
RETURN NO ISSUE.
```

Đó là behavior đúng, không phải bug.

---

# 52. SOURCE REFERENCES REVIEWED FOR THIS PLAN

Architecture / rules:

- LanguageTool: https://github.com/languagetool-org/languagetool
- LanguageTool Java rules example: https://github.com/languagetool-org/languagetool/blob/master/languagetool-core/src/main/java/org/languagetool/rules/DemoRule.java
- LanguageTool robust-rule testing: https://github.com/languagetool-org/languagetool-org.github.io/blob/master/developing-robust-rules.md
- RedPen: https://github.com/redpen-cc/redpen

Unicode:

- ICU / ICU4J: https://unicode-org.github.io/icu/
- Normalization: https://unicode-org.github.io/icu/userguide/transforms/normalization/

Vietnamese accent restoration:

- Vietnamese-Accent-Prediction: https://github.com/tienthanhdhcn/Vietnamese-Accent-Prediction
- VNOpenAI/vn-accent: https://github.com/VNOpenAI/vn-accent

Spell candidate retrieval:

- SymSpell: https://github.com/wolfgarbe/SymSpell
- customized-symspell Java: https://github.com/MighTguY/customized-symspell

Vietnamese lexical references for POC/coverage:

- https://github.com/undertheseanlp/dictionary
- https://github.com/duyet/vietnamese-wordlist

Optional future NLP reference:

- https://github.com/vncorenlp/VnCoreNLP

---

# 53. ONE-SENTENCE HANDOFF TO ANOTHER AGENT

> Implement a Java/Spring Boot SMS content linter exactly in the phase order above; preserve original UTF-16 offsets, run deterministic rules before any linguistic logic, protect URLs/codes/numbers from false positives, load config/dictionaries in memory, detect missing Vietnamese accents via reverse accent candidates + local n-gram beam ranking, use SymSpell-style lookup only for true typo candidates, emit warnings only above calibrated confidence/margin thresholds, and never auto-correct or call external AI/API.
