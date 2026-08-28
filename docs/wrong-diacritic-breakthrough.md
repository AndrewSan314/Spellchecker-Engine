# Breakthrough: lane sai dấu (ACCENTED_SAME_KEY)

Ngày 2026-08-27. Thay thế Phase 6 trong kế hoạch gốc — hóa ra đây mới là nút thắt
chính, không phải contextual ranker.

## Chẩn đoán

Phân rã 1.106 nhãn vàng VSEC dev theo **loại lỗi** (dùng `accentKey` thật, không
phải hàm strip tự viết — lỗi đo ban đầu của tôi đảo ngược hoàn toàn tỷ lệ):

| loại lỗi | số nhãn | % tổng | bắt được | recall |
|---|---|---|---|---|
| ACCENTED_SAME_KEY (sai dấu: `phái`→`phải`) | 588 | **53,2%** | 85 | **0,145** |
| DIFFERENT_KEY (typo thật: `đế`→`đến`) | 398 | 36,0% | 95 | 0,239 |
| UNACCENTED_SAME_KEY (thiếu dấu: `quy khach`) | 120 | 10,8% | 99 | 0,825 |

Engine rất giỏi đúng thứ nó được xây để làm — nhưng thứ đó chỉ chiếm 10,8% lỗi thật.
Loại chiếm đa số nằm ở 0,145, tức **503 nhãn mất ≈ 45% toàn bộ khoảng trống recall**.

## Nguyên nhân gốc: lane bị vô hiệu hoàn toàn

Trace từng token trên 588 nhãn đó:

- 98,1% token được tìm thấy, 84,0% vào được lane
- **98% ứng viên đúng ĐÃ có sẵn trong `accentIndex`** — engine *biết* từ đúng
- nhưng `isRealWordTone = false` trên **mọi** token

Lý do (`src/rules/linguistic-rules.mjs`):

1. `:1340-1343` — `realWordTypoMode: 'ACTIVE'` nhận **mọi** token `DICTIONARY`,
   mà token sai dấu cũng là `DICTIONARY`.
2. `:1401` — token đó đi đường light path DIFFERENT_KEY, không có feature nào cho
   bằng chứng same-key.
3. `:1636` — `realWordLegacySuppressed` nuốt nốt phần còn lại.

Hệ quả: `wrongDiacriticMode` OFF / SHADOW / ACTIVE cho ra **metrics giống hệt nhau
từng chữ số**. Lane đã tồn tại trong code nhưng chưa bao giờ chạy.

Ghi chú: veto `realWordProofFail` (`:1625`) **không** phải thủ phạm — đo được nó chỉ
chặn 6,8% qua `origAttested`, ngược với ước lượng ban đầu của tôi.

## Bốn thay đổi code

Tất cả trong `src/rules/linguistic-rules.mjs`:

1. **Hai lane bổ trợ, không loại trừ** (`:1406`). Token `isRealWordTone` chạy cascade
   trước; nếu cascade từ chối thì rơi xuống light path DIFFERENT_KEY (`:1650`).
   Đúng nguyên tắc "bổ trợ, không thay thế" bạn đã chốt. Nhờ đó DIFFERENT_KEY
   không những không mất mà còn nhích lên (95 → 96).

2. **Miễn suppression cho token sai dấu** (`:1640`). Chúng đã qua chứng minh trigram
   đối xứng — chặt hơn hẳn các gate mà suppression này sinh ra để bù.

3. **Veto bigram mới `realWordBigramProofFail`** (sau `:1630`). Bắt được FP thật:
   `chính hãng` (bigram 547) là cụm đúng, nhưng chỉ `hành chính hàng` tình cờ có
   trigram nên veto trigram bỏ phiếu sửa. Bigram gốc được chứng thực và ≥ bigram
   ứng viên là bằng chứng phản bác quyết định, đúng ở vùng trigram im lặng.
   Riêng veto này đẩy precision 0,721 → 0,771.

4. **Pre-check rẻ cho latency** (`:1406`). Định tuyến token sai dấu qua cascade làm
   p95 tăng 2,7× (15,4 → 42,0ms). Pre-check same-key phản chiếu **chính xác** cửa sổ
   của veto cuối (lần đầu tôi viết lệch thứ tự và mất 13 TP), chi phí ≤12 ứng viên ×
   3 cửa sổ hash. SHADOW được miễn — chế độ đó tồn tại để *báo cáo* headroom kể cả
   khi veto sẽ từ chối.

Config: `wrongDiacriticMode: 'OFF' → 'ACTIVE'`, `realWordTypoMinProbability: 0,95 → 0,97`.

## Kết quả

VSEC dev (tập dùng để tune):

| | R | P | F0.5 | FA/165 |
|---|---|---|---|---|
| trước | 0,252 | 0,679 | 0,507 | 0 |
| sau | **0,421** | **0,773** | **0,662** | **0** |
| mục tiêu kế hoạch | ≥0,42 | ≥0,72 | ≥0,62 | ≤2 |

VSEC **test** — chưa từng dùng trong quá trình tune, xác nhận không overfit:

| | R | P | F0.5 | FA/165 |
|---|---|---|---|---|
| trước | 0,240 | 0,695 | 0,504 | 0 |
| sau | **0,397** | **0,764** | **0,645** | 0 |

Recall theo loại lỗi trên dev: ACC 85→283 (0,145 → **0,481**), DIFF 95→96,
UNACC 99/120 không đổi.

Precision **tăng** cùng lúc với recall — không phải đánh đổi. Đúng ràng buộc
precision-first.

Latency SMS thật (165 tin): p50 8,1ms · **p95 18,2ms** (gate 20ms) · p99 22,8ms.
VSEC dev p95 34,1ms — văn bản dài hơn SMS đáng kể, không phải đường phục vụ.

Test: 245 pass / 0 fail / 3 skipped.

## Hai test phải sửa (và tại sao không phải là che lỗi)

- `test_real_word_typo_lane.mjs` "explicit OFF mode" — test đo lane DIFFERENT_KEY
  đơn lẻ, nhưng `đen` cũng same-key với `đến` nên lane sai dấu (giờ ACTIVE mặc định)
  kéo nó qua prefilter một cách chính đáng. Đã pin `wrongDiacriticMode: 'OFF'` để
  test đo đúng lane nó nói tên.
- `test_wrong_diacritic_lane.mjs` "wouldEmit in SHADOW" — pre-check latency ban đầu
  short-circuit cả SHADOW, làm mất báo cáo headroom. Đây là **lỗi thật của tôi**, đã
  sửa ở code (miễn SHADOW), không sửa test.

`rules.test.mjs` "red-team rows stay silent" từng đỏ vì FP `chính hãng` → đã sửa
bằng veto bigram ở code, test giữ nguyên.

## Điều này nói gì về kế hoạch gốc

Kế hoạch xếp lane này ở **Phase 6**, sau cùng, với lý do "phải chờ Phase 1 + 5".
Thực tế nó là nguồn recall lớn nhất và chỉ cần bốn thay đổi cục bộ. Phán đoán
"trigram coverage phải cao hơn trước đã" sai vì nó dựa trên giả định veto đang chặn
— trong khi lane thậm chí chưa từng chạy để bị veto.

Phase 2 (sinh lại dữ liệu ở prior đúng) vẫn nên hủy: xem `docs/phase-2-findings.md`.
