# Adversarial minimal-pair analysis

## sms-real-191
**INPUT:** Vui lòng xác nhận ban đã nhận được mã mới.
**EXPECTED:** Vui lòng xác nhận bạn đã nhận được mã mới.
**UPSTREAM:** []
**FORK_FULL:** []
**FORK_LITE:** []
**VERDICT:** UPSTREAM=MISSED; FORK_FULL=MISSED; FORK_LITE=MISSED

## sms-real-192
**INPUT:** Cửa hàng đang ban sản phẩm mới trong hôm nay.
**EXPECTED:** Cửa hàng đang bán sản phẩm mới trong hôm nay.
**UPSTREAM:** []
**FORK_FULL:** []
**FORK_LITE:** [{"ruleId":"POSSIBLE_MISSING_DIACRITIC","severity":"WARNING","start":14,"end":17,"value":"ban","suggestions":["bán"],"confidence":0.9909734243591639,"message":"Từ \"ban\" có thể đang thiếu dấu (gợi ý: \"bán\")."}]
**VERDICT:** UPSTREAM=MISSED; FORK_FULL=MISSED; FORK_LITE=CORRECT

## sms-real-193
**INPUT:** Chiếc ban sẽ được giao vào chiều mai.
**EXPECTED:** Chiếc bàn sẽ được giao vào chiều mai.
**UPSTREAM:** []
**FORK_FULL:** []
**FORK_LITE:** []
**VERDICT:** UPSTREAM=MISSED; FORK_FULL=MISSED; FORK_LITE=MISSED

## sms-real-194
**INPUT:** Tôi nghi yêu cầu này cần được kiểm tra lại.
**EXPECTED:** Tôi nghĩ yêu cầu này cần được kiểm tra lại.
**UPSTREAM:** [{"ruleId":"POSSIBLE_MISSING_DIACRITIC","severity":"WARNING","start":4,"end":8,"value":"nghi","suggestions":["nghị"],"confidence":0.91,"message":"Từ \"nghi\" có thể đang thiếu dấu (gợi ý: \"nghị\")."}]
**FORK_FULL:** [{"ruleId":"POSSIBLE_MISSING_DIACRITIC","severity":"WARNING","start":4,"end":8,"value":"nghi","suggestions":["nghị"],"confidence":0.9117068937117404,"message":"Từ \"nghi\" có thể đang thiếu dấu (gợi ý: \"nghị\")."}]
**FORK_LITE:** []
**VERDICT:** UPSTREAM=DETECTED_NOT_FULL; FORK_FULL=DETECTED_NOT_FULL; FORK_LITE=MISSED

## sms-real-195
**INPUT:** Bạn nên nghi ngơi sau khi uống thuốc.
**EXPECTED:** Bạn nên nghỉ ngơi sau khi uống thuốc.
**UPSTREAM:** [{"ruleId":"POSSIBLE_MISSING_DIACRITIC","severity":"WARNING","start":8,"end":12,"value":"nghi","suggestions":["nghỉ"],"confidence":0.98,"message":"Từ \"nghi\" có thể đang thiếu dấu (gợi ý: \"nghỉ\")."}]
**FORK_FULL:** [{"ruleId":"POSSIBLE_MISSING_DIACRITIC","severity":"WARNING","start":8,"end":12,"value":"nghi","suggestions":["nghỉ"],"confidence":0.9773369389885709,"message":"Từ \"nghi\" có thể đang thiếu dấu (gợi ý: \"nghỉ\")."}]
**FORK_LITE:** []
**VERDICT:** UPSTREAM=CORRECT; FORK_FULL=CORRECT; FORK_LITE=MISSED

## sms-real-196
**INPUT:** Ma này chỉ dùng một lần để xác thực.
**EXPECTED:** Mã này chỉ dùng một lần để xác thực.
**UPSTREAM:** []
**FORK_FULL:** []
**FORK_LITE:** []
**VERDICT:** UPSTREAM=MISSED; FORK_FULL=MISSED; FORK_LITE=MISSED

## sms-real-197
**INPUT:** Ma bạn chưa xác nhận đơn hàng hôm nay.
**EXPECTED:** Mà bạn chưa xác nhận đơn hàng hôm nay.
**UPSTREAM:** []
**FORK_FULL:** []
**FORK_LITE:** [{"ruleId":"POSSIBLE_MISSING_DIACRITIC","severity":"WARNING","start":0,"end":2,"value":"Ma","suggestions":["mà"],"confidence":0.9714183208685261,"message":"Từ \"Ma\" có thể đang thiếu dấu (gợi ý: \"mà\")."}]
**VERDICT:** UPSTREAM=MISSED; FORK_FULL=MISSED; FORK_LITE=CORRECT

## sms-real-198
**INPUT:** Đơn hang đã đến điểm nhận gần nhất.
**EXPECTED:** Đơn hàng đã đến điểm nhận gần nhất.
**UPSTREAM:** [{"ruleId":"POSSIBLE_MISSING_DIACRITIC","severity":"WARNING","start":4,"end":8,"value":"hang","suggestions":["hàng"],"confidence":0.9,"message":"Từ \"hang\" có thể đang thiếu dấu (gợi ý: \"hàng\")."}]
**FORK_FULL:** [{"ruleId":"POSSIBLE_MISSING_DIACRITIC","severity":"WARNING","start":4,"end":8,"value":"hang","suggestions":["hàng"],"confidence":0.9041843183188357,"message":"Từ \"hang\" có thể đang thiếu dấu (gợi ý: \"hàng\")."}]
**FORK_LITE:** [{"ruleId":"POSSIBLE_MISSING_DIACRITIC","severity":"WARNING","start":4,"end":8,"value":"hang","suggestions":["hàng"],"confidence":0.94168437636866,"message":"Từ \"hang\" có thể đang thiếu dấu (gợi ý: \"hàng\")."}]
**VERDICT:** UPSTREAM=CORRECT; FORK_FULL=CORRECT; FORK_LITE=CORRECT

## sms-real-199
**INPUT:** Sản phẩm này thuộc hang phân phối chính thức.
**EXPECTED:** Sản phẩm này thuộc hãng phân phối chính thức.
**UPSTREAM:** []
**FORK_FULL:** []
**FORK_LITE:** []
**VERDICT:** UPSTREAM=MISSED; FORK_FULL=MISSED; FORK_LITE=MISSED

## sms-real-200
**INPUT:** Vui lòng ky tên trước khi gửi biểu mẫu.
**EXPECTED:** Vui lòng ký tên trước khi gửi biểu mẫu.
**UPSTREAM:** [{"ruleId":"POSSIBLE_MISSING_DIACRITIC","severity":"WARNING","start":9,"end":11,"value":"ky","suggestions":["kỳ"],"confidence":0.82,"message":"Từ \"ky\" có thể đang thiếu dấu (gợi ý: \"kỳ\")."}]
**FORK_FULL:** [{"ruleId":"POSSIBLE_MISSING_DIACRITIC","severity":"WARNING","start":9,"end":11,"value":"ky","suggestions":["kỳ"],"confidence":0.8208339263055304,"message":"Từ \"ky\" có thể đang thiếu dấu (gợi ý: \"kỳ\")."}]
**FORK_LITE:** [{"ruleId":"POSSIBLE_MISSING_DIACRITIC","severity":"WARNING","start":9,"end":11,"value":"ky","suggestions":["kỳ"],"confidence":0.9120966818801269,"message":"Từ \"ky\" có thể đang thiếu dấu (gợi ý: \"kỳ\")."}]
**VERDICT:** UPSTREAM=DETECTED_NOT_FULL; FORK_FULL=DETECTED_NOT_FULL; FORK_LITE=DETECTED_NOT_FULL
