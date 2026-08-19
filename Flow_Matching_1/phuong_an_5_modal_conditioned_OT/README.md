# Phương án 5 — Modality-conditioned OT path (đường OT điều kiện-modal riêng từng user-item)

Triển khai **Phương án 5** trong
[`../DiffMM_FlowMatching_Optimization_Plan.md`](../DiffMM_FlowMatching_Optimization_Plan.md) và chi
tiết đầy đủ ở
[`../Phuong_An_5_Modality_Conditioned_OT_KeHoachChiTiet.md`](../Phuong_An_5_Modality_Conditioned_OT_KeHoachChiTiet.md):
thay lịch trình OT-linear dùng chung cho mọi item (Phương án 1) bằng lịch trình **riêng cho từng cặp
(user, item)**, co giãn theo độ phù hợp giữa embedding modal của item và sở thích modal của user — ý
tưởng tự thiết kế, đã qua 2 vòng kiểm chứng trước khi tới đây:

1. [`../Phuong_An_5_GiaiDoanA_GradCheck/`](../Phuong_An_5_GiaiDoanA_GradCheck/README.md) — kiểm tra
   gradient bằng sai phân hữu hạn trên công thức tách rời, ngoài codebase.
2. [`../Phuong_An_5_GiaiDoanB_PatchThat/`](../Phuong_An_5_GiaiDoanB_PatchThat/README.md) — hồi quy trên
   patch thật (`κ=0` trùng khít tuyệt đối Phương án 1); phát hiện và sửa 1 lỗi thật trong quá trình này.

Folder này là **Giai đoạn C**: đóng gói theo đúng quy trình `Folder_Base/` để chạy trên Colab.

## Nội dung folder

- **`DiffMM-ModalOT/`** — bản fork đầy đủ của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) đã có sẵn
  patch Phương án 5 áp trực tiếp vào code (`Params.py`, `Model.py`, `Main.py`, `DataHandler.py`). Bản
  sao độc lập (git repo riêng, nằm ngoài project) ở `E:\NAM_BA\DiffMM-ModalOT` đã được push lên
  [thyelmot/DiffMM_M_C_OT](https://github.com/thyelmot/DiffMM_M_C_OT).
- **`DiffMM_PhuongAn5_ModalOT_Colab.ipynb`** — notebook chạy trên Google Colab, **Cell 1 đã điền sẵn**
  `GITHUB_REPO_URL` + `GDRIVE_LINK` ở trên: clone code từ repo GitHub, tự tải dữ liệu từ Google Drive,
  xác minh patch đúng, chạy huấn luyện, xuất bảng chỉ số tốt nhất kèm tên dataset, và **tự động xuất +
  tải về file PDF** của bảng kết quả đó ngay khi chạy xong.

## Bước 1 — Chạy notebook trên Colab

1. Mở Google Colab → `File > Upload notebook` → chọn `DiffMM_PhuongAn5_ModalOT_Colab.ipynb`.
2. `Runtime > Change runtime type > Hardware accelerator > GPU`.
3. Cell 1 đã điền sẵn repo + dữ liệu — chỉ cần kiểm tra `DATASET_NAME` (`tiktok` mặc định),
   `NUM_EPOCHS`, và **`KAPPA`** (khuyến nghị chạy `0.0` trước) — rồi `Runtime > Run all`.

`KAPPA=0.0` (mặc định) tương đương Phương án 1 — khuyến nghị chạy `KAPPA=0.0` trước để xác nhận pipeline
chạy đúng với dữ liệu thật, rồi mới tăng dần `KAPPA` (bắt đầu `0.1`-`0.5`) để bật hiệu ứng điều
kiện-modal.

Dữ liệu trên Google Drive cần chứa: `trnMat.pkl`, `tstMat.pkl`, `image_feat.npy`, `text_feat.npy` (+
`audio_feat.npy` nếu `DATASET_NAME="tiktok"`).

## Troubleshooting

Xem bảng "Bẫy môi trường đã biết" trong `Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md`.

## Đã kiểm chứng trước khi bàn giao

- [x] Patch biên dịch sạch (`py_compile`) trên bản mới nhất lấy từ repo gốc.
- [x] Kiểm tra số học trên CPU: hồi quy `κ=0` trùng khít tuyệt đối Phương án 1/2/3, không NaN/Inf ở mọi
      điểm biên đã thử (xem `Phuong_An_5_GiaiDoanB_PatchThat/verify_cpu.py`).
- [x] Notebook đã dry-run cell-theo-cell — **cả 2 nhánh**: thành công (clone giả lập → tải dữ liệu giả
      lập → xác minh patch → huấn luyện giả lập → xuất PDF thật, đã kiểm tra file `.pdf` tạo ra không
      rỗng, đúng định dạng) và thất bại (repo không có patch → Cell 5 báo lỗi rõ ràng; script chính lỗi
      thật → Cell 6 báo lỗi kèm log đầy đủ).
- [x] Git repo độc lập cho `DiffMM-ModalOT` được tạo ở vị trí **ngoài** mọi repo khác
      (`E:\NAM_BA\DiffMM-ModalOT`, xác nhận bằng `git rev-parse --show-toplevel`), đã push lên
      [thyelmot/DiffMM_M_C_OT](https://github.com/thyelmot/DiffMM_M_C_OT).
- [x] Cell 1 notebook đã điền sẵn `GITHUB_REPO_URL` + `GDRIVE_LINK` do người dùng cung cấp.

Phần chưa/không thể kiểm chứng ở đây (do môi trường dev không có GPU/dữ liệu thật):
- Chạy full training thật trên GPU với dữ liệu thật (Recall/NDCG thật).
- Chi phí bộ nhớ GPU thực tế khi giữ đồng thời nhiều mảng `(batch, T, num_items)` — nếu tràn bộ nhớ,
  thử giảm `--batch`.

Nếu gặp lỗi khi chạy thật trên Colab, gửi lại log để debug tiếp.
