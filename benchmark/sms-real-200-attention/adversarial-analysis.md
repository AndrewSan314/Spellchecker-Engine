# Adversarial minimal-pair analysis

These are direct test-set observations. No candidate, model, or threshold tuning was performed.

## sms-real-191

- INPUT: Vui lòng xác nhận ban đã nhận được mã mới.
- EXPECTED: ban -> bạn

### OFF

- Generated pool: N/A
- Shortlist K=8: N/A
- Selected: N/A; final emitted: false; final correct: false
- Stage: PREFILTERED; verdict: **PREFILTERED**
- KEEP/candidate logits/probs: N/A

### SHADOW

- Generated pool: bản, bán, bàn, bạn, bắn, bẩn, băn, bận, bần, bân, bấn
- Shortlist K=8: bản, bán, bàn, bạn, bắn, bẩn, băn, bận
- Selected: bản; final emitted: false; final correct: false
- Stage: MODEL_WRONG_CANDIDATE; verdict: **MODEL_WRONG_CANDIDATE**
- KEEP/candidate logits/probs: ban (logit=-4.842100, p=0.098827) | bản (logit=-3.179599, p=0.521062) | bán (logit=-4.447559, p=0.146629) | bàn (logit=-4.996295, p=0.084705) | bạn (logit=-4.548631, p=0.132534) | bắn (logit=-7.827010, p=0.004995) | bẩn (logit=-7.466480, p=0.007163) | băn (logit=-8.691190, p=0.002105) | bận (logit=-8.752044, p=0.001981)

### ACTIVE

- Generated pool: bản, bán, bàn, bạn, bắn, bẩn, băn, bận, bần, bân, bấn
- Shortlist K=8: bản, bán, bàn, bạn, bắn, bẩn, băn, bận
- Selected: bản; final emitted: true; final correct: false
- Stage: MODEL_WRONG_CANDIDATE; verdict: **MODEL_WRONG_CANDIDATE**
- KEEP/candidate logits/probs: ban (logit=-4.842100, p=0.098827) | bản (logit=-3.179599, p=0.521062) | bán (logit=-4.447559, p=0.146629) | bàn (logit=-4.996295, p=0.084705) | bạn (logit=-4.548631, p=0.132534) | bắn (logit=-7.827010, p=0.004995) | bẩn (logit=-7.466480, p=0.007163) | băn (logit=-8.691190, p=0.002105) | bận (logit=-8.752044, p=0.001981)

## sms-real-192

- INPUT: Cửa hàng đang ban sản phẩm mới trong hôm nay.
- EXPECTED: ban -> bán

### OFF

- Generated pool: N/A
- Shortlist K=8: N/A
- Selected: N/A; final emitted: false; final correct: false
- Stage: PREFILTERED; verdict: **PREFILTERED**
- KEEP/candidate logits/probs: N/A

### SHADOW

- Generated pool: bản, bán, bàn, bạn, bắn, bẩn, băn, bận, bần, bân, bấn
- Shortlist K=8: bản, bán, bàn, bạn, bắn, bẩn, băn, bận
- Selected: bán; final emitted: false; final correct: false
- Stage: CORRECT_RANK_BUT_GATE_REJECT; verdict: **CORRECT_RANK_BUT_GATE_REJECT**
- KEEP/candidate logits/probs: ban (logit=-4.545002, p=0.004967) | bản (logit=-4.978662, p=0.003219) | bán (logit=0.711752, p=0.953019) | bàn (logit=-3.088504, p=0.021314) | bạn (logit=-4.855580, p=0.003641) | bắn (logit=-5.303493, p=0.002327) | bẩn (logit=-6.255603, p=0.000898) | băn (logit=-4.787245, p=0.003899) | bận (logit=-4.243493, p=0.006715)

### ACTIVE

- Generated pool: bản, bán, bàn, bạn, bắn, bẩn, băn, bận, bần, bân, bấn
- Shortlist K=8: bản, bán, bàn, bạn, bắn, bẩn, băn, bận
- Selected: bán; final emitted: false; final correct: false
- Stage: CORRECT_RANK_BUT_GATE_REJECT; verdict: **CORRECT_RANK_BUT_GATE_REJECT**
- KEEP/candidate logits/probs: ban (logit=-4.545002, p=0.004967) | bản (logit=-4.978662, p=0.003219) | bán (logit=0.711752, p=0.953019) | bàn (logit=-3.088504, p=0.021314) | bạn (logit=-4.855580, p=0.003641) | bắn (logit=-5.303493, p=0.002327) | bẩn (logit=-6.255603, p=0.000898) | băn (logit=-4.787245, p=0.003899) | bận (logit=-4.243493, p=0.006715)

## sms-real-193

- INPUT: Chiếc ban sẽ được giao vào chiều mai.
- EXPECTED: ban -> bàn

### OFF

- Generated pool: N/A
- Shortlist K=8: N/A
- Selected: N/A; final emitted: false; final correct: false
- Stage: PREFILTERED; verdict: **PREFILTERED**
- KEEP/candidate logits/probs: N/A

### SHADOW

- Generated pool: bản, bán, bàn, bạn, bắn, bẩn, băn, bận, bần, bân, bấn
- Shortlist K=8: bản, bán, bàn, bạn, bắn, bẩn, băn, bận
- Selected: bạn; final emitted: false; final correct: false
- Stage: MODEL_WRONG_CANDIDATE; verdict: **MODEL_WRONG_CANDIDATE**
- KEEP/candidate logits/probs: ban (logit=-4.828017, p=0.017793) | bản (logit=-2.440787, p=0.193650) | bán (logit=-2.878608, p=0.124990) | bàn (logit=-2.266938, p=0.230419) | bạn (logit=-1.725901, p=0.395811) | bắn (logit=-6.437246, p=0.003559) | bẩn (logit=-4.476367, p=0.025292) | băn (logit=-6.228928, p=0.004384) | bận (logit=-6.295181, p=0.004103)

### ACTIVE

- Generated pool: bản, bán, bàn, bạn, bắn, bẩn, băn, bận, bần, bân, bấn
- Shortlist K=8: bản, bán, bàn, bạn, bắn, bẩn, băn, bận
- Selected: bạn; final emitted: false; final correct: false
- Stage: MODEL_WRONG_CANDIDATE; verdict: **MODEL_WRONG_CANDIDATE**
- KEEP/candidate logits/probs: ban (logit=-4.828017, p=0.017793) | bản (logit=-2.440787, p=0.193650) | bán (logit=-2.878608, p=0.124990) | bàn (logit=-2.266938, p=0.230419) | bạn (logit=-1.725901, p=0.395811) | bắn (logit=-6.437246, p=0.003559) | bẩn (logit=-4.476367, p=0.025292) | băn (logit=-6.228928, p=0.004384) | bận (logit=-6.295181, p=0.004103)

## sms-real-194

- INPUT: Tôi nghi yêu cầu này cần được kiểm tra lại.
- EXPECTED: nghi -> nghĩ

### OFF

- Generated pool: N/A
- Shortlist K=8: N/A
- Selected: N/A; final emitted: true; final correct: false
- Stage: PREFILTERED; verdict: **PREFILTERED**
- KEEP/candidate logits/probs: N/A

### SHADOW

- Generated pool: nghị, nghỉ, nghĩ, nghì
- Shortlist K=8: nghị, nghỉ, nghĩ, nghì
- Selected: nghị; final emitted: true; final correct: false
- Stage: MODEL_WRONG_CANDIDATE; verdict: **MODEL_WRONG_CANDIDATE**
- KEEP/candidate logits/probs: nghi (logit=-4.282491, p=0.008678) | nghị (logit=-0.069981, p=0.586002) | nghỉ (logit=-3.863793, p=0.013191) | nghĩ (logit=-0.477680, p=0.389796) | nghì (logit=-5.596215, p=0.002333)

### ACTIVE

- Generated pool: nghị, nghỉ, nghĩ, nghì
- Shortlist K=8: nghị, nghỉ, nghĩ, nghì
- Selected: nghị; final emitted: true; final correct: true
- Stage: MODEL_WRONG_CANDIDATE; verdict: **CORRECT**
- KEEP/candidate logits/probs: nghi (logit=-4.282491, p=0.008678) | nghị (logit=-0.069981, p=0.586002) | nghỉ (logit=-3.863793, p=0.013191) | nghĩ (logit=-0.477680, p=0.389796) | nghì (logit=-5.596215, p=0.002333)

## sms-real-195

- INPUT: Bạn nên nghi ngơi sau khi uống thuốc.
- EXPECTED: nghi -> nghỉ

### OFF

- Generated pool: N/A
- Shortlist K=8: N/A
- Selected: N/A; final emitted: true; final correct: true
- Stage: CORRECT_EMIT; verdict: **CORRECT**
- KEEP/candidate logits/probs: N/A

### SHADOW

- Generated pool: nghị, nghỉ, nghĩ, nghì
- Shortlist K=8: nghị, nghỉ, nghĩ, nghì
- Selected: nghỉ; final emitted: true; final correct: true
- Stage: CORRECT_EMIT; verdict: **CORRECT**
- KEEP/candidate logits/probs: nghi (logit=-4.567386, p=0.000860) | nghị (logit=-3.916390, p=0.001649) | nghỉ (logit=2.475172, p=0.984340) | nghĩ (logit=-1.857621, p=0.012925) | nghì (logit=-5.907667, p=0.000225)

### ACTIVE

- Generated pool: nghị, nghỉ, nghĩ, nghì
- Shortlist K=8: nghị, nghỉ, nghĩ, nghì
- Selected: nghỉ; final emitted: false; final correct: false
- Stage: CORRECT_RANK_BUT_GATE_REJECT; verdict: **CORRECT_RANK_BUT_GATE_REJECT**
- KEEP/candidate logits/probs: nghi (logit=-4.567386, p=0.000860) | nghị (logit=-3.916390, p=0.001649) | nghỉ (logit=2.475172, p=0.984340) | nghĩ (logit=-1.857621, p=0.012925) | nghì (logit=-5.907667, p=0.000225)

## sms-real-196

- INPUT: Ma này chỉ dùng một lần để xác thực.
- EXPECTED: Ma -> Mã

### OFF

- Generated pool: N/A
- Shortlist K=8: N/A
- Selected: N/A; final emitted: false; final correct: false
- Stage: PREFILTERED; verdict: **PREFILTERED**
- KEEP/candidate logits/probs: N/A

### SHADOW

- Generated pool: mà, mã, mạ, má, mả, mặ
- Shortlist K=8: mà, mã, má, mạ, mả, mặ
- Selected: mã; final emitted: false; final correct: false
- Stage: CORRECT_RANK_BUT_GATE_REJECT; verdict: **CORRECT_RANK_BUT_GATE_REJECT**
- KEEP/candidate logits/probs: Ma (logit=-4.630843, p=0.118052) | mà (logit=-4.660726, p=0.114577) | mã (logit=-3.487911, p=0.370207) | má (logit=-3.814608, p=0.267031) | mạ (logit=-6.143295, p=0.026015) | mả (logit=-5.623988, p=0.043728) | mặ (logit=-5.301147, p=0.060390)

### ACTIVE

- Generated pool: mà, mã, mạ, má, mả, mặ
- Shortlist K=8: mà, mã, má, mạ, mả, mặ
- Selected: mã; final emitted: false; final correct: false
- Stage: CORRECT_RANK_BUT_GATE_REJECT; verdict: **CORRECT_RANK_BUT_GATE_REJECT**
- KEEP/candidate logits/probs: Ma (logit=-4.630843, p=0.118052) | mà (logit=-4.660726, p=0.114577) | mã (logit=-3.487911, p=0.370207) | má (logit=-3.814608, p=0.267031) | mạ (logit=-6.143295, p=0.026015) | mả (logit=-5.623988, p=0.043728) | mặ (logit=-5.301147, p=0.060390)

## sms-real-197

- INPUT: Ma bạn chưa xác nhận đơn hàng hôm nay.
- EXPECTED: Ma -> Mà

### OFF

- Generated pool: N/A
- Shortlist K=8: N/A
- Selected: N/A; final emitted: false; final correct: false
- Stage: PREFILTERED; verdict: **PREFILTERED**
- KEEP/candidate logits/probs: N/A

### SHADOW

- Generated pool: mà, mã, mạ, má, mả, mặ
- Shortlist K=8: mà, mã, má, mạ, mả, mặ
- Selected: mà; final emitted: false; final correct: false
- Stage: CORRECT_RANK_BUT_GATE_REJECT; verdict: **CORRECT_RANK_BUT_GATE_REJECT**
- KEEP/candidate logits/probs: Ma (logit=-4.874826, p=0.008781) | mà (logit=-0.277999, p=0.870781) | mã (logit=-4.254714, p=0.016325) | má (logit=-2.760883, p=0.072712) | mạ (logit=-5.222563, p=0.006202) | mả (logit=-4.708911, p=0.010365) | mặ (logit=-4.350449, p=0.014834)

### ACTIVE

- Generated pool: mà, mã, mạ, má, mả, mặ
- Shortlist K=8: mà, mã, má, mạ, mả, mặ
- Selected: mà; final emitted: false; final correct: false
- Stage: CORRECT_RANK_BUT_GATE_REJECT; verdict: **CORRECT_RANK_BUT_GATE_REJECT**
- KEEP/candidate logits/probs: Ma (logit=-4.874826, p=0.008781) | mà (logit=-0.277999, p=0.870781) | mã (logit=-4.254714, p=0.016325) | má (logit=-2.760883, p=0.072712) | mạ (logit=-5.222563, p=0.006202) | mả (logit=-4.708911, p=0.010365) | mặ (logit=-4.350449, p=0.014834)

## sms-real-198

- INPUT: Đơn hang đã đến điểm nhận gần nhất.
- EXPECTED: hang -> hàng

### OFF

- Generated pool: N/A
- Shortlist K=8: N/A
- Selected: N/A; final emitted: true; final correct: true
- Stage: CORRECT_EMIT; verdict: **CORRECT**
- KEEP/candidate logits/probs: N/A

### SHADOW

- Generated pool: hàng, hãng, hạng, hằng, hăng, hẫng, háng, hảng, hẵng, hắng, hẳng
- Shortlist K=8: hàng, hãng, hạng, hằng, hăng, hẫng, háng, hảng
- Selected: hàng; final emitted: true; final correct: true
- Stage: CORRECT_EMIT; verdict: **CORRECT**
- KEEP/candidate logits/probs: hang (logit=-4.984328, p=0.000166) | hàng (logit=3.715550, p=0.994739) | hãng (logit=-4.209802, p=0.000360) | hạng (logit=-4.086990, p=0.000407) | hằng (logit=-1.785836, p=0.004060) | hăng (logit=-5.687955, p=0.000082) | hẫng (logit=-5.726418, p=0.000079) | háng (logit=-5.931154, p=0.000064) | hảng (logit=-6.293972, p=0.000045)

### ACTIVE

- Generated pool: hàng, hãng, hạng, hằng, hăng, hẫng, háng, hảng, hẵng, hắng, hẳng
- Shortlist K=8: hàng, hãng, hạng, hằng, hăng, hẫng, háng, hảng
- Selected: hàng; final emitted: true; final correct: true
- Stage: CORRECT_EMIT; verdict: **CORRECT**
- KEEP/candidate logits/probs: hang (logit=-4.984328, p=0.000166) | hàng (logit=3.715550, p=0.994739) | hãng (logit=-4.209802, p=0.000360) | hạng (logit=-4.086990, p=0.000407) | hằng (logit=-1.785836, p=0.004060) | hăng (logit=-5.687955, p=0.000082) | hẫng (logit=-5.726418, p=0.000079) | háng (logit=-5.931154, p=0.000064) | hảng (logit=-6.293972, p=0.000045)

## sms-real-199

- INPUT: Sản phẩm này thuộc hang phân phối chính thức.
- EXPECTED: hang -> hãng

### OFF

- Generated pool: N/A
- Shortlist K=8: N/A
- Selected: N/A; final emitted: false; final correct: false
- Stage: PREFILTERED; verdict: **PREFILTERED**
- KEEP/candidate logits/probs: N/A

### SHADOW

- Generated pool: hàng, hãng, hạng, hằng, hăng, hẫng, háng, hảng, hẵng, hắng, hẳng
- Shortlist K=8: hàng, hãng, hạng, hằng, hăng, hẫng, háng, hảng
- Selected: hàng; final emitted: false; final correct: false
- Stage: MODEL_WRONG_CANDIDATE; verdict: **MODEL_WRONG_CANDIDATE**
- KEEP/candidate logits/probs: hang (logit=-4.779390, p=0.001496) | hàng (logit=1.439710, p=0.751355) | hãng (logit=0.077049, p=0.192331) | hạng (logit=-1.582733, p=0.036578) | hằng (logit=-3.110626, p=0.007937) | hăng (logit=-4.033197, p=0.003155) | hẫng (logit=-3.933205, p=0.003487) | háng (logit=-4.470561, p=0.002037) | hảng (logit=-4.696837, p=0.001625)

### ACTIVE

- Generated pool: hàng, hãng, hạng, hằng, hăng, hẫng, háng, hảng, hẵng, hắng, hẳng
- Shortlist K=8: hàng, hãng, hạng, hằng, hăng, hẫng, háng, hảng
- Selected: hàng; final emitted: true; final correct: true
- Stage: MODEL_WRONG_CANDIDATE; verdict: **CORRECT**
- KEEP/candidate logits/probs: hang (logit=-4.779390, p=0.001496) | hàng (logit=1.439710, p=0.751355) | hãng (logit=0.077049, p=0.192331) | hạng (logit=-1.582733, p=0.036578) | hằng (logit=-3.110626, p=0.007937) | hăng (logit=-4.033197, p=0.003155) | hẫng (logit=-3.933205, p=0.003487) | háng (logit=-4.470561, p=0.002037) | hảng (logit=-4.696837, p=0.001625)

## sms-real-200

- INPUT: Vui lòng ky tên trước khi gửi biểu mẫu.
- EXPECTED: ky -> ký

### OFF

- Generated pool: N/A
- Shortlist K=8: N/A
- Selected: N/A; final emitted: true; final correct: false
- Stage: PREFILTERED; verdict: **PREFILTERED**
- KEEP/candidate logits/probs: N/A

### SHADOW

- Generated pool: kỳ, ký, kỹ, kỷ, kỵ
- Shortlist K=8: kỳ, ký, kỷ, kỹ, kỵ
- Selected: ký; final emitted: true; final correct: false
- Stage: CORRECT_RANK_BUT_GATE_REJECT; verdict: **CORRECT_RANK_BUT_GATE_REJECT**
- KEEP/candidate logits/probs: ky (logit=-4.521327, p=0.030230) | kỳ (logit=-2.381865, p=0.256804) | ký (logit=-1.553330, p=0.588071) | kỷ (logit=-4.287702, p=0.038186) | kỹ (logit=-4.201922, p=0.041606) | kỵ (logit=-4.121232, p=0.045103)

### ACTIVE

- Generated pool: kỳ, ký, kỹ, kỷ, kỵ
- Shortlist K=8: kỳ, ký, kỷ, kỹ, kỵ
- Selected: ký; final emitted: false; final correct: false
- Stage: CORRECT_RANK_BUT_GATE_REJECT; verdict: **CORRECT_RANK_BUT_GATE_REJECT**
- KEEP/candidate logits/probs: ky (logit=-4.521327, p=0.030230) | kỳ (logit=-2.381865, p=0.256804) | ký (logit=-1.553330, p=0.588071) | kỷ (logit=-4.287702, p=0.038186) | kỹ (logit=-4.201922, p=0.041606) | kỵ (logit=-4.121232, p=0.045103)
