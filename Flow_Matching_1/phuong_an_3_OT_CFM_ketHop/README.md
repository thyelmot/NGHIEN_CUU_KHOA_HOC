# Phương án 3 — OT path + CFM loss + rút gọn D4 cho DiffMM

Triển khai **Phương án 3** trong [`../DiffMM_FlowMatching_Optimization_Plan.md`](../DiffMM_FlowMatching_Optimization_Plan.md)
(mục 5) và bản kế hoạch chi tiết [`../Phuong_An_3_OT_CFM_KetHop_KeHoachChiTiet.md`](../Phuong_An_3_OT_CFM_KetHop_KeHoachChiTiet.md):
kết hợp **Phương án 1** (đường đi OT-linear cho D1) + **Phương án 2** (trọng số CFM cho D2) + phần
**hoàn toàn mới**: rút gọn số bước suy luận D4 từ `T` xuống `K < T` bước bằng công thức DDIM tổng
quát hóa.

## Nội dung folder

- **`DiffMM-OT-CFM/`** — bản fork đầy đủ của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) đã có sẵn
  patch Phương án 3 áp trực tiếp vào code (`Params.py`, `Model.py`, `Main.py`, `DataHandler.py`). Đây
  là folder đã được **đẩy (push) lên GitHub riêng** — xem Bước 1 bên dưới.
- **`DiffMM_PhuongAn3_OTCFM_Colab.ipynb`** — notebook chạy trên Google Colab: clone code từ repo
  GitHub *của bạn*, tự tải dữ liệu từ Google Drive, chạy huấn luyện, xuất bảng chỉ số tốt nhất.

## Bước 1 — Đẩy `DiffMM-OT-CFM/` lên GitHub riêng của bạn

✅ **Đã push sẵn** lên `https://github.com/thyelmot/DiffMM-OT-CFM.git` từ bản độc lập tại:

```
E:\NAM_BA\DiffMM-OT-CFM
```

(git init + commit sẵn, đã tự kiểm `git rev-parse --show-toplevel` trỏ đúng chính nó — không lồng
trong project). Nếu cần cập nhật thêm sau này:

```bash
cd /e/NAM_BA/DiffMM-OT-CFM
git add -A && git commit -m "..." && git push
```

## Bước 2 — Chạy notebook trên Colab

1. Mở Google Colab → `File > Upload notebook` → chọn `DiffMM_PhuongAn3_OTCFM_Colab.ipynb`.
2. `Runtime > Change runtime type > Hardware accelerator > GPU`.
3. Cell 1 **đã điền sẵn** `GITHUB_REPO_URL` và `GDRIVE_LINK` — chỉnh thêm `DATASET_NAME`, `NUM_EPOCHS`,
   `SIGMA_MIN`, `W_CLIP`, `NUM_SAMPLE_STEPS` nếu muốn — rồi `Runtime > Run all`.

Dữ liệu trên Google Drive cần chứa: `trnMat.pkl`, `tstMat.pkl`, `image_feat.npy`, `text_feat.npy`
(+ `audio_feat.npy` nếu dùng `tiktok`).

## Troubleshooting

Xem bảng "Bẫy môi trường đã biết" trong `../../Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md` — cùng các
lỗi đã gặp khi làm Phương án 1/2 (repo bị lồng thư mục, clone dùng nhầm code cũ, scipy `.A`...) đều đã
được xử lý sẵn trong notebook này.

## Đã kiểm chứng trước khi bàn giao

- [x] Tải source gốc thật từ GitHub (`Params.py`, `Model.py`, `Main.py`, `DataHandler.py`, `Utils/*`,
      `README.md`) — không đoán tên biến/hàm/class.
- [x] Patch biên dịch sạch (`py_compile`) trên bản mới nhất tải lại từ GitHub.
- [x] Kiểm tra số học D1+D2 kết hợp: quét `w_CFM(t)` trên đường OT với `T ∈ {5,10,50}` ×
      `σ_min ∈ {10⁻⁶,10⁻³,0.1}` — không NaN/Inf, max toàn cục luôn ≤ 1.0.
- [x] Kiểm tra số học D4 (phần mới): denoiser hoàn hảo → hội tụ đúng tuyệt đối (sai số 0.0) ở mọi
      lịch trình rút gọn, kể cả `K=1` (nhảy 1 bước); denoiser không hoàn hảo → đo được đường cong K vs
      độ chính xác thực (suy giảm nhẹ, không phải vách đá) — xem số liệu cụ thể trong
      `DiffMM-OT-CFM/README.md`.
- [x] Test hồi quy: `diff_loss = 0` với denoiser hoàn hảo, mọi cấu hình K.
- [x] Dry-run toàn bộ notebook cell-theo-cell (Cell 3/5/6/7) — cả nhánh thành công lẫn nhánh lỗi (repo
      chưa patch, script training crash, thiếu log) đều cho thông báo đúng như thiết kế.
- [x] Git repo độc lập tại `E:\NAM_BA\DiffMM-OT-CFM`, tự kiểm `git rev-parse --show-toplevel` trỏ
      đúng chính nó (không lồng trong project) — **đã push lên `thyelmot/DiffMM-OT-CFM`**.
- [x] Cell 1 notebook đã điền sẵn `GITHUB_REPO_URL` và `GDRIVE_LINK`.

Phần chưa/không thể kiểm chứng ở đây (do môi trường dev không có GPU và không có dữ liệu thật): chạy
full training thật trên GPU với dữ liệu TikTok/Baby/Sports — đặc biệt cần đo thật Recall/NDCG và thời
gian chạy ở các giá trị `NUM_SAMPLE_STEPS` khác nhau, vì mọi kiểm chứng D4 ở trên đều dùng denoiser giả
lập (không phải mạng `Denoise` thật đã huấn luyện).
