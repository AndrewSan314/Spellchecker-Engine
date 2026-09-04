# dataset_sms — bộ dữ liệu SMS tiếng Việt trong repo

## Vì sao có bộ này

Mọi con số engine công bố trước đây đều đo trên **VSEC** (văn xuôi kiểu Wikipedia).
Hai vấn đề:

1. **Sai miền.** Engine phục vụ SMS brandname/marketing: câu ngắn, nhiều URL/số tiền/mã
   đơn, rất nhiều tin gõ không dấu. VSEC không giống thứ đó.
2. **Điểm số được nâng đỡ.** `tools/measure_error_channel_overlap.mjs` cho thấy ~68% nhãn
   dev VSEC có cặp (lỗi → đúng) nằm sẵn trong bảng tra sinh từ tập train, nên `pairProven`
   trả lời được bằng trí nhớ.

Bộ này sửa cả hai: đúng miền, và **bảng tra error-channel không biết gì về nó** (bảng đó
sinh từ VSEC train), nên recall đo được ở đây là recall thật của mô hình ngôn ngữ + rule,
không phải của một bảng tra.

## Nguồn gốc và license

Toàn bộ văn bản **viết tay trong repo** (`templates.json`), không lấy từ nguồn ngoài, không
chứa dữ liệu khách hàng thật. Slot (tên thương hiệu, số điện thoại, URL) là giá trị hư cấu.
Vì vậy bộ này không kèm ràng buộc license nào từ bên thứ ba — khác với VSEC/viwiki/underthesea.

## Cách sinh

```bash
npm run sms:dataset       # node tools/build_sms_dataset.mjs
```

Deterministic: cùng `templates.json` + cùng seed → byte giống hệt. `manifest.json` ghi
sha256 của từng file; `test/test_sms_dataset.mjs` kiểm tra file trên đĩa khớp manifest.

### Chống leakage — chia theo *group*, không theo dòng

Mỗi template thuộc một `group` (ví dụ `promo-discount` gồm 3 template cùng ý). Split được
quyết định bởi `sha256(group) % 100 → 70/15/15`, nên **một cách diễn đạt không bao giờ
xuất hiện ở hai split**. Đây chính là lỗ hổng mà review chỉ ra trên VSEC (chia theo message
nhưng phân bố lỗi vẫn trùng), ở đây được thiết kế để tránh ngay từ đầu.

Chỉ `sms-clean-train.txt` (train) được dùng để huấn luyện artifact. `test/test_sms_dataset.mjs`
khẳng định không có câu dev/test nào lọt vào file đó.

## Nội dung

| File | Vai trò |
|---|---|
| `templates.json` | 62 template / 35 group / 8 miền (bán lẻ, ngân hàng, giao vận, viễn thông, tiện ích, y tế, giáo dục, ví điện tử, du lịch) + slot |
| `sms-train.jsonl` | 1.630 tin có nhãn (train) |
| `sms-dev.jsonl` | 174 tin — dùng để tune |
| `sms-test.jsonl` | 138 tin — held-out, đã khai báo và chạy ĐÚNG MỘT LẦN (xem `config/acceptance-gates.json`) |
| `sms-clean-train.txt` | 1.231 câu sạch, **nguồn duy nhất** để build LM miền |
| `manifest.json` | số lượng + sha256 |
| `../benchmark/sms/corpus-sms-{dev,test}.json` | cùng dữ liệu ở định dạng benchmark row |

Một dòng jsonl:

```json
{"id":"bank-01-002","group":"bank-otp","domain":"banking","split":"dev",
 "profile":"ALL_UNACCENTED",
 "text":"Ma OTP cua quy khach la 705214, hieu luc 5 phut...",
 "clean":"Mã OTP của quý khách là 705214, hiệu lực 5 phút...",
 "correction_pairs":[{"error":"Ma","correction":"Mã","start":0,"end":2}, ...],
 "expect":[{"ruleId":"POSSIBLE_MISSING_DIACRITIC","value":"Ma","suggestion":"Mã",
            "positionStart":0,"positionEnd":2}, ...]}
```

## Các lớp lỗi (profile)

| Profile | Mô tả | Lane chịu trách nhiệm |
|---|---|---|
| `CLEAN` | không lỗi — đo báo động giả | — |
| `ALL_UNACCENTED` | cả tin không dấu (rất phổ biến thực tế) | POSSIBLE_MISSING_DIACRITIC |
| `SOME_UNACCENTED` | 1–3 từ mất dấu | POSSIBLE_MISSING_DIACRITIC |
| `TELEX` | sót chữ tone kiểu Telex: `hàng → hangf` | POSSIBLE_SPELLING_ERROR |
| `WRONG_DIACRITIC` | sai dấu nhưng vẫn là từ có thật: `kỳ → ký` | POSSIBLE_SPELLING_ERROR |
| `TYPO` | đảo ký tự: `khách → khcáh` | POSSIBLE_SPELLING_ERROR |
| `BOUNDARY` | dính từ: `cảm ơn → cảmơn` | POSSIBLE_WORD_BOUNDARY_ERROR (lane SHADOW → chưa bắt) |
| `PUNCT` | khoảng trắng/dấu câu hỏng | rule deterministic |

Lỗi **chỉ được tiêm vào phần chữ viết tay của template**, không bao giờ vào giá trị slot
(URL, số tiền, ngày, mã đơn, số điện thoại) — những giá trị đó tồn tại để kiểm tra
protected ranges và phải giữ nguyên. Có test khẳng định điều này.

## Đánh giá

```bash
npm run sms:eval            # dev, profile lite
npm run sms:eval:full       # dev, artifact đầy đủ
node tools/eval_sms.mjs --split test    # held-out: đã dùng, khai báo trước khi chạy lại
```

Kết quả xem README chính, mục "Kết quả trên miền SMS".

## Giới hạn phải nói rõ

Đây là dữ liệu **tổng hợp từ template**, không phải SMS thật của khách hàng. Nó đo đúng
miền (câu, cấu trúc, kiểu lỗi) nhưng không đo được đuôi dài của thực tế: tên riêng lạ, viết
tắt tự phát, teencode, trộn tiếng Anh, emoji. Con số ở đây là **cận trên nhẹ** cho SMS thật
— vẫn trung thực hơn VSEC nhiều, nhưng bước tiếp theo vẫn phải là một tập SMS thật do team
gán nhãn.
