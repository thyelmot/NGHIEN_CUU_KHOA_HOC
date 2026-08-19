# Phương án 2 — CFM loss weighting cho DiffMM

Triển khai **Phương án 2** trong [`../DiffMM_FlowMatching_Optimization_Plan.md`](../DiffMM_FlowMatching_Optimization_Plan.md)
(mục 5) và bản kế hoạch chi tiết [`../Phuong_An_2_CFM_Loss_KeHoachChiTiet.md`](../Phuong_An_2_CFM_Loss_KeHoachChiTiet.md):
thay cách tính trọng số của loss huấn luyện — từ ELBO/KL divergence (`w_ELBO(t) = SNR(t−1) − SNR(t)`)
sang CFM loss của Flow Matching (`w_CFM(t) = [μₜ' − (σₜ'/σₜ)·μₜ]²`) — trong khi **giữ nguyên 100%**
đường đi diffusion VP-style hiện tại và mạng dự đoán α₀ trực tiếp.

## Nội dung folder

- **`DiffMM-CFM/`** — bản fork đầy đủ của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) đã có sẵn
  patch Phương án 2 áp trực tiếp vào code (`Params.py`, `Model.py`, `Main.py`, `DataHandler.py`). Đây
  là folder cần **đẩy (push) lên GitHub riêng của bạn** — xem hướng dẫn bên dưới.
- **`DiffMM_PhuongAn2_CFM_Colab.ipynb`** — notebook chạy trên Google Colab: clone code từ repo GitHub
  *của bạn*, tự tải dữ liệu từ Google Drive, chạy huấn luyện, xuất bảng chỉ số tốt nhất.

## Bước 1 — Đẩy `DiffMM-CFM/` lên GitHub riêng của bạn

⚠️ **Luôn tạo repo git ở vị trí NẰM NGOÀI mọi git repo khác đang có trên máy** — đã chuẩn bị sẵn 1 bản
độc lập (git init + commit sẵn, đã tự kiểm `git rev-parse --show-toplevel` trỏ đúng chính nó, không
lồng trong project) tại:

```
E:\NAM_BA\DiffMM-CFM
```

1. Vào https://github.com/new, tạo một repo mới (ví dụ tên `DiffMM-CFM`), **để trống, không tick
   "Initialize with README"**.
2. Chạy các lệnh sau (thay `<your-username>` và `<repo-name>`):

```bash
cd /e/NAM_BA/DiffMM-CFM
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

3. Copy URL repo — dán vào `GITHUB_REPO_URL` ở Cell 1 của notebook.

*(Chưa tự push giúp bạn vì bạn chưa cung cấp link GitHub repo trống cho Phương án 2 này — nếu có, gửi
lại link, tôi sẽ điền sẵn vào Cell 1 và push luôn.)*

## Bước 2 — Chạy notebook trên Colab

1. Mở Google Colab → `File > Upload notebook` → chọn `DiffMM_PhuongAn2_CFM_Colab.ipynb`.
2. `Runtime > Change runtime type > Hardware accelerator > GPU`.
3. Chỉ chỉnh **Cell 1**: `GITHUB_REPO_URL`, link Google Drive chứa dữ liệu, `DATASET_NAME`,
   `NUM_EPOCHS`, `W_CLIP` — rồi `Runtime > Run all`.

Dữ liệu trên Google Drive cần chứa: `trnMat.pkl`, `tstMat.pkl`, `image_feat.npy`, `text_feat.npy`
(+ `audio_feat.npy` nếu dùng `tiktok`).

## Troubleshooting

Xem bảng "Bẫy môi trường đã biết" trong `../../Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md` — cùng các
lỗi đã gặp khi làm Phương án 1 (repo bị lồng thư mục, clone dùng nhầm code cũ, scipy `.A`...) đều đã
được xử lý sẵn trong notebook này (đã dry-run kiểm chứng lại, xem mục dưới).

## Đã kiểm chứng trước khi bàn giao

- [x] Tải source gốc thật từ GitHub (`Params.py`, `Model.py`, `Main.py`, `DataHandler.py`, `Utils/*`,
      `README.md`) — không đoán tên biến/hàm/class.
- [x] Patch biên dịch sạch (`py_compile`) trên bản mới nhất tải lại từ GitHub.
- [x] Kiểm tra số học trên CPU: quét `w_CFM(t)` với `T ∈ {5, 10, 50}` — không NaN/Inf, `w_CFM(0)=1.0`
      đúng quy ước, giá trị bị clip đúng ở `w_clip`. So sánh hình dạng với `w_ELBO(t)` gốc (không clip)
      — phát hiện: `w_ELBO` có đỉnh nhọn tới hàng nghìn lần tại t=1, `w_CFM` giảm đều đặn hơn nhiều
      (xem `DiffMM-CFM/README.md` để có số liệu cụ thể).
- [x] Test hồi quy: denoiser hoàn hảo → `diff_loss = 0` với mọi `t`.
- [x] Xác nhận D4 (`p_sample`, kế thừa nguyên vẹn) vẫn hội tụ đúng — sai số `0.0` với denoiser hoàn
      hảo, giống hệt kết quả đã kiểm chứng ở Phương án 1 (đúng như dự đoán lý thuyết: D4 không đổi).
- [x] Dry-run toàn bộ notebook cell-theo-cell (Cell 3 clone tự dò, Cell 5 xác minh patch, Cell 6 chạy
      training qua `subprocess`, Cell 7 đọc kết quả) — **cả nhánh thành công lẫn nhánh lỗi** (repo
      chưa patch, script training crash, log chưa tồn tại) đều cho thông báo đúng như thiết kế.
- [x] Git repo độc lập tại `E:\NAM_BA\DiffMM-CFM`, đã tự kiểm `git rev-parse --show-toplevel` trỏ
      đúng chính nó (không lồng trong project).

Phần chưa/không thể kiểm chứng ở đây (do môi trường dev không có GPU và không có dữ liệu thật): chạy
full training thật trên GPU với dữ liệu TikTok/Baby/Sports, và bước push GitHub thật (chưa có link
repo trống cho phương án này nên chưa tự push).
