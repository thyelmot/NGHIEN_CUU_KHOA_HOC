# Phương án 6 — Learnable Coarse Anchor (điểm neo thô học được)

Triển khai **Phương án 6** trong
[`../DiffMM_OFM_Optimization_Plan.md`](../DiffMM_OFM_Optimization_Plan.md) và chi tiết đầy đủ ở
[`../Phuong_An_6_Learnable_Anchor_KeHoachChiTiet.md`](../Phuong_An_6_Learnable_Anchor_KeHoachChiTiet.md):
lấy ý tưởng "điểm neo thô học được" từ Optical Flow Matching (OFM), dịch tâm nhiễu forward-diffusion từ
0 (cố định) sang `α_l` — ước lượng sơ bộ, phụ thuộc từng user, tính hoàn toàn từ embedding đã có sẵn
(không thêm tham số học mới).

Đã qua 2 giai đoạn kiểm chứng trước khi tới đây:

1. **Giai đoạn A** (derive + kiểm chứng công thức ngoài codebase): xem mục 3-4 của bản kế hoạch chi
   tiết — chứng minh điều kiện biên, và đặc biệt: chứng minh đại số + xác nhận số học rằng trọng số CFM
   (Phương án 2/3) **bất biến hoàn toàn** với điểm neo.
2. **Giai đoạn B** (patch thật, hồi quy): xem
   [`../Phuong_An_6_GiaiDoanB_PatchThat/`](../Phuong_An_6_GiaiDoanB_PatchThat/README.md) — `anchor_w=0`
   trùng khít tuyệt đối Phương án 3 trên `Model.py` thật, không phát hiện lỗi nào.

Folder này là **Giai đoạn C**: đóng gói theo đúng quy trình `Folder_Base/` để chạy trên Colab.

## Nội dung folder

- **`DiffMM-AnchorOT/`** — bản fork đầy đủ của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) đã có sẵn
  patch Phương án 6 áp trực tiếp vào code. Bản sao độc lập ở `E:\NAM_BA\DiffMM-AnchorOT` đã được push lên
  [thyelmot/DiffMM_6](https://github.com/thyelmot/DiffMM_6).
- **`DiffMM_PhuongAn6_AnchorOT_Colab.ipynb`** — notebook chạy trên Google Colab, **Cell 1 đã điền sẵn**
  `GITHUB_REPO_URL` + `GDRIVE_LINK` ở trên: clone code từ repo GitHub, tự tải dữ liệu từ Google Drive,
  xác minh patch đúng, chạy huấn luyện, xuất bảng chỉ số tốt nhất kèm tên dataset, và **tự động xuất +
  tải về file PDF** của bảng kết quả đó ngay khi chạy xong.

## Bước 1 — Chạy notebook trên Colab

1. Mở Google Colab → `File > Upload notebook` → chọn `DiffMM_PhuongAn6_AnchorOT_Colab.ipynb`.
2. `Runtime > Change runtime type > Hardware accelerator > GPU`.
3. Cell 1 đã điền sẵn repo + dữ liệu — chỉ cần kiểm tra `DATASET_NAME` (`tiktok` mặc định),
   `NUM_EPOCHS`, và **`ANCHOR_W`** (khuyến nghị chạy `0.0` trước) — rồi `Runtime > Run all`.

`ANCHOR_W=0.0` (mặc định) tương đương Phương án 3 — khuyến nghị chạy `ANCHOR_W=0.0` trước để xác nhận
pipeline chạy đúng với dữ liệu thật, rồi mới tăng dần `ANCHOR_W` (bắt đầu `1.0`-`5.0`) để bật hiệu ứng
điểm neo.

Dữ liệu trên Google Drive cần chứa: `trnMat.pkl`, `tstMat.pkl`, `image_feat.npy`, `text_feat.npy` (+
`audio_feat.npy` nếu `DATASET_NAME="tiktok"`).

## Troubleshooting

Xem bảng "Bẫy môi trường đã biết" trong `Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md`.

## Đã kiểm chứng trước khi bàn giao

- [x] Đã tải source gốc thật từ GitHub, không đoán tên biến/hàm.
- [x] Patch biên dịch sạch (`py_compile`) trên bản mới nhất lấy từ repo gốc.
- [x] Kiểm tra số học trên CPU: hồi quy `anchor_w=0` trùng khít tuyệt đối Phương án 3 (kể cả vòng lặp
      `p_sample` đầy đủ), `cfm_weight` bất biến với điểm neo, không NaN/Inf ở mọi điểm biên đã thử (xem
      `Phuong_An_6_GiaiDoanB_PatchThat/verify_cpu.py`).
- [x] Notebook đã dry-run cell-theo-cell — **cả 2 nhánh**: thành công (clone giả lập → tải dữ liệu giả
      lập → xác minh patch → huấn luyện giả lập → xuất PDF thật, đã kiểm tra file `.pdf` tạo ra không
      rỗng) và thất bại (repo không có patch → Cell 5 báo lỗi rõ ràng; script chính lỗi thật → Cell 6
      báo lỗi kèm log đầy đủ).
- [x] Git repo độc lập cho `DiffMM-AnchorOT` được tạo ở vị trí **ngoài** mọi repo khác
      (`E:\NAM_BA\DiffMM-AnchorOT`, xác nhận bằng `git rev-parse --show-toplevel`), đã push lên
      [thyelmot/DiffMM_6](https://github.com/thyelmot/DiffMM_6).
- [x] Cell 1 notebook đã điền sẵn `GITHUB_REPO_URL` + `GDRIVE_LINK` do người dùng cung cấp.
- [x] README bản fork có đối chiếu công thức gốc/mới, rõ file/hàm bị sửa.

Phần chưa/không thể kiểm chứng ở đây (do môi trường dev không có GPU/dữ liệu thật):
- Chạy full training thật trên GPU với dữ liệu thật (Recall/NDCG thật) — đặc biệt là liệu điểm neo thô
  `α_l` có thực sự cải thiện chất lượng gợi ý hay không, đây là câu hỏi mở duy nhất còn lại (rủi ro
  *thực nghiệm*, không còn là rủi ro toán học/kỹ thuật — xem mục 4 bản kế hoạch chi tiết).

Nếu gặp lỗi khi chạy thật trên Colab, gửi lại log để debug tiếp.
