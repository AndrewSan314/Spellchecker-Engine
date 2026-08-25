// Probe run: candidate clean rows (traps included) — observe what actually fires.
import { show } from './probe.mjs';

const C = [
  ['P01', 'Tendoo xin thông báo: Flash Sale 12.12 bắt đầu 08:30 ngày 12/10/2026! Mua deal hot tại https://tendoo.vn/khuyenmai?utm_source=sms&id=DH123456 và nhận voucher 100.000đ cho 200 khách đầu tiên 🎉', 'ACCENTED', 'VT_TENDOO'],
  ['P02', 'ABC BANK kính gửi quý khách: GD009876 vừa nhận 500.000đ vào tài khoản lúc 14:05. Lưu ý: không ai được yêu cầu mã OTP của bạn.', 'ACCENTED', 'ABC_BANK'],
  ['P03', 'Chào Nguyễn Văn An! Tendoo giao đơn DH881235 đến địa chỉ Đà Nẵng trước 17:30. Nhân viên sẽ gọi số 0912 345 678 khi đến. Trân trọng!', 'ACCENTED', 'VT_TENDOO'],
  ['P04', 'TK {{stk}} da thay doi so du: +500.000d luc 21:00 ngay 12/10. SDT ho tro 1900 6936.', 'NON_ACCENTED', 'ABC_BANK'],
  ['P05', 'Mua ngay iPhone 15 Pro Max giá tốt tại Tendoo Store, bảo hành chính hãng 12 tháng, freeship mọi đơn hàng!', 'ACCENTED', 'VT_TENDOO'],
  ['P06', 'Ma xac minh OTP cua ban la 834902. Khong chuyen tiep ma nay. Truy cap: https://abcbank.vn/otp?id={{req_id}}', 'NON_ACCENTED', 'ABC_BANK'],
  ['P07', 'THONG BAO: GD +2.500.000d từ VIETTEL MONEY lúc 09:15. SD 7.850.000d. Hotline 1900 6936.', 'ACCENTED', 'ABC_BANK'],
  ['P08', 'Giảm đến 70% toàn bộ sản phẩm... Mua ngay tại cskh@tendoo.vn hoặc link facebook.com/tendoo.vn! Số lượng có hạn.', 'ACCENTED', 'VT_TENDOO'],
  ['P09', 'Kỳ thanh toán gần nhất: 12/10/2026. Quý khách vui lòng thanh toán trước 20:00 hôm nay.', 'ACCENTED', 'ABC_BANK'],
  ['P10', 'OTP 553821 dung cho GD 889132 tai ATM Vietcombank. Khong phai ban thao tac? Goi 1900 6936 ngay.', 'NON_ACCENTED', 'ABC_BANK'],
  ['P11', '{{ten_khach}} thân mến, đơn {order_id} của bạn đã được đóng gói và sẽ giao trong {{so_ngay}} ngày. Cảm ơn bạn đã tin dùng Tendoo!', 'ACCENTED', 'VT_TENDOO'],
  ['P12', 'Lưu ý: Mã A.B.C đã được gửi vào email của bạn trước 08:00.', 'ACCENTED', 'ABC_BANK'],
  ['P13', 'Giao hang FREE cho don tu 199.000d. Ap dung toi 31/12. Truy cap tendoo.vn/freeship de biet chi tiet.', 'NON_ACCENTED', 'VT_TENDOO'],
  ['P14', 'Chúc mừng sinh nhật Trần Thị Hoa! Nhận ngay voucher 200.000đ dành cho bạn tại tendoo.vn/sinh-nhat. Happy birthday 🎂', 'ACCENTED', 'VT_TENDOO'],
  ['P15', 'Thẻ của bạn vừa bị khóa do nhập sai mã PIN 3 lần. Gọi 1900 6936 để mở khóa.', 'ACCENTED', 'ABC_BANK'],
  ['P16', 'Bao cao tai khoan thang 10 da san sang. Xem ngay tren app ABC BANK hoac goi 1900 6936.', 'NON_ACCENTED', 'ABC_BANK'],
  ['P17', 'Tendoo xin thông báo:\n\n- Giờ mở cửa: 07:00 - 22:00 mỗi ngày\n- Hotline: 1900 6868\nTrân trọng,', 'ACCENTED', 'VT_TENDOO'],
  ['P18', 'Lãi suất kỳ 13 tuần chỉ 5,5%/năm. Gửi ngay hôm nay để nhận ưu đãi!', 'ACCENTED', 'ABC_BANK'],
  ['P19', 'Quét mã QR tại cửa hàng để tích điểm tự động.\tXin cảm ơn!', 'ACCENTED', 'VT_TENDOO'],
  ['P20', 'Don hang DH552010 cua ban da gui thanh cong. Thanh toan 1.250.000d khi nhan hang. Hen gap lai ban vao dip Tet 2026!', 'NON_ACCENTED', 'VT_TENDOO'],
  ['P21', 'Mua 1kg tặng 0,5kg. Chỉ từ 35.000đ/hộp. Áp dụng tại 63 tỉnh thành đến hết 30/04!', 'ACCENTED', 'VT_TENDOO'],
  ['P22', 'QUÝ KHÁCH LƯU Ý: ABC BANK không bao giờ yêu cầu khách hàng chuyển tiền vào tài khoản cá nhân. Nếu nhận được email lạ, gọi 1900 6936.', 'ACCENTED', 'ABC_BANK'],
  ['P23', 'Quy khach than men, TK {{tk}} vua rut thanh cong 2.000.000d tai ATM. Tran trong!', 'NON_ACCENTED', 'ABC_BANK'],
  ['P24', '[Tendoo] Chỉ còn 3 giờ! Giảm đến 50% toàn bộ iPhone, Samsung; freeship đơn từ 99.000đ. Đặt đơn ngay tại https://m.tendoo.vn/deal?utm=ny&id=8888 hoặc gọi 0987 654 321. Hẹn gặp lại!', 'ACCENTED', 'VT_TENDOO'],
];

for (const [id, text, mode, brand] of C) show(id, text, mode, brand);

console.log('\n\n########## EXTRA TRAPS ##########');
show('T-hochiminh', 'Giao hàng tận nơi tại TP. Hồ Chí Minh và Hà Nội.');
show('T-momo', 'Thanh toán qua Ví MoMo nhận giảm 20%.');
show('T-zalopay', 'Nạp tiền ZaloPay nhận ưu đãi ngay hôm nay!');
show('T-money', 'Đăng ký Viettel Money tặng 50.000đ.');
show('T-vietcombank', 'Tài khoản Vietcombank của bạn sẵn sàng.');
show('T-lowercase-initials', 'gặp nhau tại a.b.c của quán');
show('T-unknown-tld', 'Mua ngay tại shop.hanghieu.sale hoặc tendoo.store!');
show('T-qua-app', 'Đặt hàng qua app để nhận mã giảm giá.');
show('T-long-context', 'Quý khách có thể chọn dài dài thêm');
