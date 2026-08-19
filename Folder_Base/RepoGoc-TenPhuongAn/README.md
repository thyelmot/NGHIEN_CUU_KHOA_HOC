# <TenRepo>-<TenPhuongAn>

> **File mẫu.** Khi tạo bản fork thật, thay nội dung dưới đây bằng thông tin thật của phương án mới,
> theo đúng cấu trúc này (xem `Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md` mục "Bước 5").

Đây là bản fork của [<owner>/<repo>](https://github.com/<owner>/<repo>) (paper *<tên paper gốc>*)
với **<Tên Phương án>** trong kế hoạch tối ưu [`<Ten>_Optimization_Plan.md`](<link>) đã được áp dụng
trực tiếp vào code (không phải patch lúc chạy runtime).

## Thay đổi so với bản gốc

<Mô tả ngắn gọn ý tưởng: công thức/thuật toán nào bị thay, dựa trên cơ sở lý thuyết nào (trích dẫn
paper + số công thức).>

```
<công thức cũ, nếu có>
```

thay cho

```
<công thức mới>
```

| File | Thay đổi |
|---|---|
| `<file>` | <mô tả> |
| `<file>` | <mô tả> |

Toàn bộ phần còn lại (<liệt kê các thành phần không đổi>) **giữ nguyên 100%** so với bản gốc.

Đã kiểm chứng: patch biên dịch sạch (`py_compile`) và <đã kiểm tra số học trên CPU nếu áp dụng — mô tả
ngắn gọn điều kiện biên/quét dải tham số đã test>.

## Dữ liệu

<Ghi chú cách lấy dữ liệu — ví dụ: không chứa sẵn dữ liệu để giữ repo nhẹ, tải qua notebook Colab đi
kèm từ Google Drive. Liệt kê danh sách file dữ liệu cần thiết.>

## Chạy cục bộ (không qua Colab)

```bash
<lệnh chạy mẫu, lấy từ README gốc + thêm tham số của phương án mới>
```

(Yêu cầu GPU nếu code gốc dùng `.cuda()` trực tiếp.)

## Bản gốc

Xem các file source để đối chiếu — mọi phần không liên quan <Tên Phương án> giữ nguyên y hệt
[<owner>/<repo>](https://github.com/<owner>/<repo>). Trích dẫn paper gốc:

```
<bibtex trích dẫn paper gốc>
```
