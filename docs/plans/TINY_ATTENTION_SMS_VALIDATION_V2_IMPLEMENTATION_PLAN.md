# PLAN V2 --- Tiny Attention Contextual Reranker for SMS Validation Engine

## Mục tiêu

Nâng cấp engine hiện tại nhưng không phá kiến trúc:

-   Giữ nguyên deterministic rules.
-   Giữ nguyên Protected Range.
-   Giữ nguyên AccentIndex + SymSpell candidate generation.
-   Thay tầng quyết định cuối bằng Unified Contextual Reranker.
-   Model nhỏ, local inference, deploy web được.

Không dùng: - LLM API. - Generative correction. - PhoBERT/BERT vài trăm
MB.

Target: - Model INT8 \< 10MB. - p95 latency tăng tối đa vài ms. -
Precision-first.

------------------------------------------------------------------------

# 1. Kiến trúc cuối

    SMS
     |
     v
    Normalize + Tokenize
     |
     v
    Protected Range Detection
     |
     v
    Deterministic Rules
     |
     v
    Lexical Layer
     |
     v
    Candidate Generation
     |----------------|
     |                |
    AccentIndex     SymSpell
     |                |
     -----------------
            |
            v
     Candidate Pool
            |
            v
     Diversity Shortlist (K=8)
            |
            v
     Tiny Transformer Context Encoder
            |
            v
     Unified Candidate Scorer
            |
            v
     KEEP_ORIGINAL vs Candidates
            |
            v
     Confidence Gate
            |
            v
     ValidationIssue

------------------------------------------------------------------------

# 2. Nguyên tắc quan trọng

## Không thay thế text

Model chỉ quyết định:

    KEEP_ORIGINAL
    hoặc
    candidate nào tốt hơn

Không generate câu mới.

------------------------------------------------------------------------

## Candidate generation không đổi

Giữ:

    AccentIndex
    SymSpell
    Dictionary
    Frequency
    Edit distance

Attention chỉ làm ranking.

------------------------------------------------------------------------

# 3. Sửa lỗi architecture hiện tại

## Fix 1 --- Candidate phải có character representation

Hiện tại nhiều candidate là OOV.

Không chỉ dùng:

    candidate word id

Mà dùng:

    candidate embedding =
    word embedding
    +
    character ngram embedding

Ví dụ:

    hướng
    hưởng
    hương

phải tạo vector khác nhau dù không nằm trong vocab.

------------------------------------------------------------------------

## Fix 2 --- Không hard skip dictionary word

Sai:

    dictionary.contains(token)
        -> skip

Vì real-word typo:

    lên -> nên
    mày -> máy
    đế -> đến
    các -> cách

đều là từ hợp lệ.

Dictionary chỉ là feature:

    original_is_valid_word = true

Không phải quyết định cuối.

------------------------------------------------------------------------

## Fix 3 --- Attention phải xử lý real-word error

Không tách:

    missing accent
    typo
    real word typo
    wrong diacritic

thành nhiều decision engine.

Tất cả đưa vào:

    KEEP + candidate ranking

------------------------------------------------------------------------

## Fix 4 --- Train/runtime phải dùng cùng shortlist

Không được:

Train:

    K=8

Runtime:

    K=4

Bắt buộc:

    Candidate shortlist function dùng chung

------------------------------------------------------------------------

# 4. Tiny Transformer model

Bắt đầu:

    Layers: 2
    Hidden: 64
    Heads: 4
    FFN: 192
    Pre-LN Transformer

Không tăng model trước khi sửa pipeline.

Input:

    SMS context window

Không encode từng candidate.

------------------------------------------------------------------------

# 5. Context encoding

Sai:

    candidate1 -> transformer
    candidate2 -> transformer
    candidate3 -> transformer

Đúng:

    SMS
     |
    Transformer
     |
    token contextual embeddings

Encode một lần.

Sau đó:

    context vector
    +
    candidate vector

đưa vào scorer.

------------------------------------------------------------------------

# 6. Target masking

Khi encode context:

Không để model nhìn trực tiếp token đang sửa.

Ví dụ:

Input:

    Tôi muốn [TARGET] Hà Nội

Model học:

    context -> candidate phù hợp

Không học:

    token lỗi giống token đúng

------------------------------------------------------------------------

# 7. Candidate scorer

Input:

    context embedding
    candidate embedding
    ngram score
    edit distance
    frequency
    lexical features

Output:

    score(KEEP)
    score(candidate1)
    score(candidate2)
    ...

Loss:

CrossEntropy:

    correct candidate hoặc KEEP là target

------------------------------------------------------------------------

# 8. Data training

## Positive correction

Sinh từ clean corpus:

    hướng dẫn

corruption:

    huong dẫn

target:

    hướng

------------------------------------------------------------------------

## KEEP examples

Bắt buộc:

    SMS đúng
    +
    candidate sai
    +
    target KEEP

Để tránh model sửa quá nhiều.

------------------------------------------------------------------------

## Hard negatives

Mine từ engine hiện tại:

    clean SMS
     |
    current engine
     |
    false correction
     |
    KEEP training sample

Ví dụ:

    mang giấy tờ

không được sửa:

    mạng giấy tờ

------------------------------------------------------------------------

# 9. Inference flow

Runtime:

    Token
     |
    Candidate generation
     |
    Shortlist K=8
     |
    Attention rank
     |
    Confidence
     |
    Emit / Skip

Không chạy attention nếu:

    token chắc chắn đúng

------------------------------------------------------------------------

# 10. Evaluation

Không đánh giá chỉ ranking.

Đánh giá full engine:

Metrics:

    Precision
    Recall
    F0.5
    False positive
    Clean message FP
    Latency p95
    Memory

So sánh:

    Baseline N-gram
    Attention V1
    Hybrid V2

------------------------------------------------------------------------

# 11. Deployment

Export:

    PyTorch
     |
    ONNX
     |
    INT8
     |
    Node/Java inference

Yêu cầu:

    Không GPU
    Không Python runtime production
    Không API

------------------------------------------------------------------------

# 12. Thứ tự implement

## Phase 1

Fix pipeline:

-   Candidate K=8.
-   Unify train/runtime.
-   Dictionary soft feature.
-   Add char candidate embedding.

## Phase 2

Train model:

-   Clean corpus.
-   Synthetic corruption.
-   Hard negatives.
-   KEEP training.

## Phase 3

Inference:

-   ONNX export.
-   INT8.
-   Integrate scorer.

## Phase 4

Benchmark:

-   Full engine evaluation.
-   Threshold calibration.
-   Shadow mode.

## Phase 5

Activate only if:

-   Precision không giảm đáng kể.
-   Recall tăng có ý nghĩa.
-   Latency đạt requirement.
