# Phương án 7 (TVS) — Tam giác hoá vận tốc (Triangle Velocities Synergy)

Triển khai **Phương án 7 (TVS)** trong [`DiffMM_OFM_Optimization_Plan.md`](../DiffMM_OFM_Optimization_Plan.md) (mục 5) và chi tiết tại [`Phuong_An_7_TVS_KeHoachChiTiet.md`](../Phuong_An_7_TVS_KeHoachChiTiet.md): Cải tiến quy trình hồi quy vận tốc (velocity-prediction) dựa trên quan hệ hình học tam giác trong bài báo Optical Flow Matching (OFM), xây dựng trực tiếp trên nền tảng của Phương án 6 (Learnable Coarse Anchor).

## Nội dung folder

- **`DiffMM-TVS/`** — bản fork đầy đủ của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) đã có sẵn patch Phương án 7 áp trực tiếp vào code (`Params.py`, `Model.py`, `Main.py`). Đây là folder cần **đẩy (push) lên GitHub riêng của bạn** — xem hướng dẫn bên dưới.
- **`DiffMM_PhuongAn7_TVS_Colab.ipynb`** — notebook chạy trên Google Colab: clone code từ repo GitHub *của bạn*, tự tải dữ liệu từ Google Drive, chạy huấn luyện, xuất bảng chỉ số tốt nhất kèm tên dataset, và **tự động xuất + tải về file PDF** của bảng kết quả đó ngay khi chạy xong (không cần thao tác gì thêm).

## Bước 1 — Đẩy `DiffMM-TVS/` lên GitHub riêng của bạn

⚠️ **Luôn tạo repo git ở vị trí NẰM NGOÀI mọi git repo khác đang có trên máy** (không tạo `.git` bên trong 1 thư mục đã bị git repo cha theo dõi) — tránh lỗi "repo bị lồng thêm 1 cấp thư mục" khi push. Repo mẫu cho phương án này đã được tôi chuẩn bị và đẩy sẵn lên GitHub tại:

```
https://github.com/thyelmot/DiffMM_7.git
```

Bạn cũng có thể tự tạo repo của riêng mình và đẩy code từ thư mục cục bộ (nằm ngoài project chính):
```
E:\NAM_BA\DiffMM-TVS
```

Lệnh đẩy lên GitHub (thay URL bằng repo riêng của bạn nếu muốn):
```bash
cd E:\NAM_BA\DiffMM-TVS
git remote add origin https://github.com/thyelmot/DiffMM_7.git
git branch -M main
git push -u origin main
```

## Bước 2 — Chạy notebook trên Colab

1. Mở Google Colab → `File > Upload notebook` → chọn `DiffMM_PhuongAn7_TVS_Colab.ipynb`.
2. `Runtime > Change runtime type > Hardware accelerator > GPU`.
3. Chỉ chỉnh **Cell 1**: `GITHUB_REPO_URL`, link Google Drive chứa dữ liệu, các hyperparameter — rồi `Runtime > Run all`.
   *(Mặc định URL repo và link Google Drive đã được tôi điền sẵn ở Cell 1, bạn chỉ cần chạy)*

Dữ liệu trên Google Drive cần chứa (ở đâu đó bên trong file/thư mục chia sẻ, notebook tự dò tìm):
`trnMat.pkl`, `tstMat.pkl`, `image_feat.npy`, `text_feat.npy` (và `audio_feat.npy` nếu chạy dataset `tiktok`).

## Troubleshooting

Xem bảng "Bẫy môi trường đã biết" trong `Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md` — hầu hết lỗi gặp phải khi chạy code cũ trên Colab hiện tại đều đã có sẵn cách chẩn đoán/sửa trong bảng đó.

**"Đã sửa lỗi trên GitHub rồi mà chạy lại vẫn lỗi y hệt"** — Cell clone code luôn xoá bản clone cũ trong Colab rồi clone lại từ đầu mỗi lần chạy, nên chỉ cần chạy lại cell đó (không cần Restart runtime). Nếu vẫn lỗi cũ, kiểm tra lại đã `git push` đúng repo mà `GITHUB_REPO_URL` đang trỏ tới chưa.

## Đã kiểm chứng trước khi bàn giao

- [x] Patch biên dịch sạch (`py_compile`) trên bản mới nhất lấy từ repo gốc.
- [x] Kiểm tra số học trên CPU qua `verify_cpu_tvs.py`: điều kiện biên đúng, không NaN/Inf ở mọi điểm quét, round-trip tái tạo dữ liệu hoàn hảo, fallback về PA6 chuẩn xác.
- [x] Notebook đã dry-run cell-theo-cell (nhánh thành công + nhánh lỗi).

Phần chưa/không thể kiểm chứng ở đây (do môi trường dev không có GPU/dữ liệu thật): chạy full training thật trên GPU với dữ liệu thật. Nếu gặp lỗi khi chạy thật trên Colab, gửi lại log để debug tiếp.
