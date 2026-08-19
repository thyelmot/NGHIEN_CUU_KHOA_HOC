# <Tên Phương án> — <mô tả 1 dòng>

> **Đây là README mẫu.** Khi tạo folder phương án mới, copy file này (và toàn bộ cấu trúc
> `Folder_Base/`) sang folder mới, rồi thay mọi chỗ trong dấu `<...>` bằng nội dung thật — xem
> `Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md` để biết quy trình đầy đủ.

Triển khai **<Tên Phương án>** trong [`<đường_dẫn>/..._Optimization_Plan.md`](<đường_dẫn>) (mục
<số>): <mô tả ngắn gọn thay đổi gì, dựa trên công thức/ý tưởng nào>.

## Nội dung folder

- **`<TenRepo>-<TenPhuongAn>/`** — bản fork đầy đủ của [<owner>/<repo>](https://github.com/<owner>/<repo>) đã có sẵn patch <Tên Phương án> áp trực tiếp vào code (`<file 1>`, `<file 2>`, ...). Đây là folder cần **đẩy (push) lên GitHub riêng của bạn** — xem hướng dẫn bên dưới.
- **`<Ten>_Colab.ipynb`** — notebook chạy trên Google Colab: clone code từ repo GitHub *của bạn*, tự tải dữ liệu từ Google Drive, chạy huấn luyện, xuất bảng chỉ số tốt nhất kèm tên dataset, và **tự động xuất + tải về file PDF** của bảng kết quả đó ngay khi chạy xong (không cần thao tác gì thêm).

## Bước 1 — Đẩy `<TenRepo>-<TenPhuongAn>/` lên GitHub riêng của bạn

⚠️ **Luôn tạo repo git ở vị trí NẰM NGOÀI mọi git repo khác đang có trên máy** (không tạo `.git` bên
trong 1 thư mục đã bị git repo cha theo dõi) — tránh lỗi "repo bị lồng thêm 1 cấp thư mục" khi push
(xem mục Troubleshooting). Repo mẫu cho phương án này (nếu đã được chuẩn bị sẵn) nằm tại:

```
<đường_dẫn_tuyệt_đối_ngoài_project>
```

1. Vào https://github.com/new, tạo một repo mới, **để trống, không tick "Initialize with README"**.
2. Chạy các lệnh sau (thay `<your-username>` và `<repo-name>`):

```bash
cd <đường_dẫn_tuyệt_đối_ngoài_project>
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

3. Copy URL repo — dán vào `GITHUB_REPO_URL` ở Cell 1 của notebook.

## Bước 2 — Chạy notebook trên Colab

1. Mở Google Colab → `File > Upload notebook` → chọn `<Ten>_Colab.ipynb`.
2. `Runtime > Change runtime type > Hardware accelerator > GPU`.
3. Chỉ chỉnh **Cell 1**: `GITHUB_REPO_URL`, link Google Drive chứa dữ liệu, các hyperparameter — rồi `Runtime > Run all`.

Dữ liệu trên Google Drive cần chứa (ở đâu đó bên trong file/thư mục chia sẻ, notebook tự dò tìm):
`<liệt kê file dữ liệu bắt buộc>`.

## Troubleshooting

Xem bảng "Bẫy môi trường đã biết" trong `Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md` — hầu hết lỗi gặp
phải khi chạy code cũ trên Colab hiện tại đều đã có sẵn cách chẩn đoán/sửa trong bảng đó.

**"Đã sửa lỗi trên GitHub rồi mà chạy lại vẫn lỗi y hệt"** — Cell clone code luôn xoá bản clone cũ
trong Colab rồi clone lại từ đầu mỗi lần chạy, nên chỉ cần chạy lại cell đó (không cần Restart
runtime). Nếu vẫn lỗi cũ, kiểm tra lại đã `git push` đúng repo mà `GITHUB_REPO_URL` đang trỏ tới chưa.

## Đã kiểm chứng trước khi bàn giao

- [ ] Patch biên dịch sạch (`py_compile`) trên bản mới nhất lấy từ repo gốc.
- [ ] <Nếu có công thức toán mới> Kiểm tra số học trên CPU: điều kiện biên đúng, không NaN/Inf ở mọi điểm quét.
- [ ] Notebook đã dry-run cell-theo-cell (nhánh thành công + nhánh lỗi).

Phần chưa/không thể kiểm chứng ở đây (do môi trường dev không có GPU/dữ liệu thật): chạy full training
thật trên GPU với dữ liệu thật. Nếu gặp lỗi khi chạy thật trên Colab, gửi lại log để debug tiếp.
