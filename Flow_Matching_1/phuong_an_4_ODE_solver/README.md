# Phương án 4 — Rút gọn D4 bằng ODE solver, không đổi huấn luyện

Triển khai **Phương án 4** trong [`../DiffMM_FlowMatching_Optimization_Plan.md`](../DiffMM_FlowMatching_Optimization_Plan.md)
(mục 5) và bản kế hoạch chi tiết [`../Phuong_An_4_ODE_Solver_KeHoachChiTiet.md`](../Phuong_An_4_ODE_Solver_KeHoachChiTiet.md):
**chỉ** rút gọn số bước suy luận D4 (từ `T` xuống `K < T` bước, bằng công thức DDIM tổng quát hóa —
lời giải đóng của probability flow ODE), **không đổi gì** về đường đi (D1) hay cách huấn luyện (D2).
Đây là patch tối giản nhất trong 4 phương án (chỉ 2 hàm bị override).

## Nội dung folder

- **`DiffMM-ODE/`** — bản fork đầy đủ của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) đã có sẵn
  patch Phương án 4 áp trực tiếp vào code (`Params.py`, `Model.py`, `Main.py`, `DataHandler.py`). Đã
  được **đẩy (push) lên GitHub riêng** — xem Bước 1 bên dưới.
- **`DiffMM_PhuongAn4_ODE_Colab.ipynb`** — notebook chạy trên Google Colab: clone code từ repo GitHub
  *của bạn*, tự tải dữ liệu từ Google Drive, chạy huấn luyện, xuất bảng chỉ số tốt nhất.

## Bước 1 — Đẩy `DiffMM-ODE/` lên GitHub riêng của bạn

✅ **Đã push sẵn** lên `https://github.com/thyelmot/DiffMM-ODE.git` từ bản độc lập tại:

```
E:\NAM_BA\DiffMM-ODE
```

(git init + commit sẵn, đã tự kiểm `git rev-parse --show-toplevel` trỏ đúng chính nó — không lồng
trong project). Nếu cần cập nhật thêm sau này:

```bash
cd /e/NAM_BA/DiffMM-ODE
git add -A && git commit -m "..." && git push
```

## Bước 2 — Chạy notebook trên Colab

1. Mở Google Colab → `File > Upload notebook` → chọn `DiffMM_PhuongAn4_ODE_Colab.ipynb`.
2. `Runtime > Change runtime type > Hardware accelerator > GPU`.
3. Cell 1 **đã điền sẵn** `GITHUB_REPO_URL` và `GDRIVE_LINK` — chỉnh thêm `DATASET_NAME`, `NUM_EPOCHS`,
   `NUM_SAMPLE_STEPS` nếu muốn — rồi `Runtime > Run all`.

Dữ liệu trên Google Drive cần chứa: `trnMat.pkl`, `tstMat.pkl`, `image_feat.npy`, `text_feat.npy`
(+ `audio_feat.npy` nếu dùng `tiktok`).

## Troubleshooting

Xem bảng "Bẫy môi trường đã biết" trong `../../Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md`.

## Đã kiểm chứng trước khi bàn giao

- [x] Tải source gốc thật từ GitHub (`Params.py`, `Model.py`, `Main.py`, `DataHandler.py`, `Utils/*`,
      `README.md`) — không đoán tên biến/hàm/class.
- [x] Patch biên dịch sạch (`py_compile`) trên bản mới nhất tải lại từ GitHub.
- [x] Kiểm tra số học: denoiser hoàn hảo → hội tụ đúng tuyệt đối (sai số 0.0) ở mọi `K` thử (kể cả
      `K=1`), nhiều `T`. Đã phát hiện và loại bỏ 1 cách làm SAI ("skip thô" dùng thẳng công thức
      Bayes gốc cho bước nhảy nhiều bước — xem chi tiết `DiffMM-ODE/README.md`).
- [x] Xác nhận D1 thực sự không đổi: so sánh trực tiếp các mảng hệ số giữa bản patch và
      `GaussianDiffusion` gốc — giống hệt tuyệt đối.
- [x] Dry-run toàn bộ notebook cell-theo-cell (Cell 3/5/6/7) — cả nhánh thành công lẫn nhánh lỗi (repo
      chưa patch, script training crash, thiếu log) đều cho thông báo đúng như thiết kế.
- [x] Git repo độc lập tại `E:\NAM_BA\DiffMM-ODE`, tự kiểm `git rev-parse --show-toplevel` trỏ đúng
      chính nó (không lồng trong project) — **đã push lên `thyelmot/DiffMM-ODE`**.
- [x] Cell 1 notebook đã điền sẵn `GITHUB_REPO_URL` và `GDRIVE_LINK`.

Phần chưa/không thể kiểm chứng ở đây (do môi trường dev không có GPU và không có dữ liệu thật): chạy
full training thật trên GPU với dữ liệu TikTok/Baby/Sports, đo Recall/NDCG và thời gian thật ở các giá
trị `NUM_SAMPLE_STEPS` khác nhau với mạng `Denoise` thật đã huấn luyện — và (nếu Phương án 3 cũng đã
chạy) so sánh trực tiếp ở cùng `K` giữa đường VP (Phương án 4) và đường OT (Phương án 3), đúng gợi ý
thực nghiệm trong bản kế hoạch chi tiết.
