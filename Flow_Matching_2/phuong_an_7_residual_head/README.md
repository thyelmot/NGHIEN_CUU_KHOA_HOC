# Phương án 7 v2 — Residual Head (mạng dự đoán phần dư so với điểm neo)

Triển khai **Residual Head** — mục 5 của
[`../Phuong_An_7_TVS_KeHoachChiTiet_v2/Phuong_An_7_TVS_KeHoachChiTiet_v2.md`](../Phuong_An_7_TVS_KeHoachChiTiet_v2/Phuong_An_7_TVS_KeHoachChiTiet_v2.md).

> **Phạm vi:** bản kế hoạch v2 có 2 phần. Phần "TVS đầy đủ" (đổi sang velocity-prediction, mục 4) **chưa
> triển khai** — chính tài liệu kết luận nên giữ ở trạng thái "ghi nhận, chưa triển khai" vì thiếu điều
> kiện kích hoạt K1-K3. Folder này chỉ triển khai phần **Residual Head** (mục 5) — được khuyến nghị làm
> ngay vì rẻ, an toàn, và capture đúng động lực gốc của TVS mà không cần đổi tham số hoá.

Ý tưởng: thay vì để mạng `Denoise` dự đoán `α₀` trực tiếp (như PA1-6), cho mạng dự đoán **phần dư**
(residual) so với điểm neo thô `α_l` (đã có từ Phương án 6): `α̂₀ = α_l + Denoise(αₜ,t)`. Đây không phải
1 công thức xác suất mới — chỉ là 1 skip-connection ở đầu ra mạng — nên `q_sample`, `p_mean_variance`,
`cfm_weight` giữ nguyên 100% công thức của Phương án 6.

## Nội dung folder

- **`DiffMM-ResidualOT/`** — bản fork đầy đủ của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) đã có
  sẵn patch Phương án 7 v2 áp trực tiếp vào code. Bản sao độc lập ở `E:\NAM_BA\DiffMM-ResidualOT` đã
  được push lên [thyelmot/DiffMM_7_v2](https://github.com/thyelmot/DiffMM_7_v2).
- **`verify_cpu.py`** — script kiểm chứng số học trên chính `Model.py` thật (không cần GPU).
- **`DiffMM_PhuongAn7_ResidualOT_Colab.ipynb`** — notebook chạy trên Google Colab, **Cell 1 đã điền
  sẵn** `GITHUB_REPO_URL` + `GDRIVE_LINK`.

## Bước 1 — Chạy notebook trên Colab

1. Mở Google Colab → `File > Upload notebook` → chọn `DiffMM_PhuongAn7_ResidualOT_Colab.ipynb`.
2. `Runtime > Change runtime type > Hardware accelerator > GPU`.
3. Cell 1 đã điền sẵn repo + dữ liệu — chỉ cần kiểm tra `DATASET_NAME` (`tiktok` mặc định),
   `NUM_EPOCHS`, `ANCHOR_W`, và **`RESIDUAL_HEAD`** (khuyến nghị chạy `False` trước) — rồi
   `Runtime > Run all`.

`RESIDUAL_HEAD=False` (mặc định) tương đương Phương án 6 ở mọi `ANCHOR_W` — khuyến nghị chạy vậy trước
để xác nhận pipeline đúng với dữ liệu thật, rồi mới bật `RESIDUAL_HEAD=True`.

Dữ liệu trên Google Drive cần chứa: `trnMat.pkl`, `tstMat.pkl`, `image_feat.npy`, `text_feat.npy` (+
`audio_feat.npy` nếu `DATASET_NAME="tiktok"`).

## Troubleshooting

Xem bảng "Bẫy môi trường đã biết" trong `Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md`. Không phát hiện bẫy
môi trường mới nào trong quá trình làm phương án này (patch chỉ thêm 1 class kế thừa, không đụng phần
hạ tầng đã ổn định từ PA1-6).

## Checklist bàn giao (theo mục 5, `Folder_Base/HUONG_DAN_XAY_DUNG_FOLDER.md`)

```
[x] Source gốc lấy từ GitHub thật (link: https://github.com/HKUDS/DiffMM, tải lại mới nhất, không dùng bản cache)
[x] Patch biên dịch sạch (py_compile)
[x] Đã dry-run số học — quét qua (anchor_w, residual_head) ở nhiều tổ hợp, không NaN/Inf ở mọi điểm biên
    (kể cả anchor_w cực đoan 1e6, user rỗng) — xem Phuong_An_7_TVS_KeHoachChiTiet_v2/ muc 7 va
    verify_cpu.py trong folder nay
[x] Đã dry-run notebook cell-theo-cell (thành công + lỗi)
[x] Cell 7 đã dry-run phần xuất PDF — file .pdf tạo ra thật (36.5KB), đúng tên phương án + tên dataset
[x] Git repo độc lập, nằm NGOÀI mọi repo khác (E:\NAM_BA\DiffMM-ResidualOT, xác nhận bằng
    git rev-parse --show-toplevel)
[x] README bản fork có đối chiếu công thức gốc/mới
[x] Cell 1 notebook đã điền sẵn link người dùng cung cấp
[x] Đã liệt kê rõ phần CHƯA kiểm chứng được (bên dưới)
```

**Kiểm chứng đặc biệt cho Phương án 7 (mạnh hơn PA6):** hồi quy `residual_head=False` được xác nhận
trùng khít Phương án 6 **ở cả `anchor_w=0` lẫn `anchor_w>0`** (không chỉ tại `anchor_w=0` như PA6 đã làm
với PA3) — vì `residual_head` là 1 công tắc hoàn toàn độc lập với `anchor_w`.

## Phần chưa/không thể kiểm chứng ở đây

- Chạy full training thật trên GPU với dữ liệu thật (Recall/NDCG thật) — đặc biệt là liệu residual head
  có thực sự giúp mạng hội tụ nhanh hơn/tốt hơn hay không. Đây là câu hỏi thực nghiệm thuần tuý (không
  có công thức toán nào cần chứng minh thêm, vì đây chỉ là 1 skip-connection kiến trúc tiêu chuẩn).
- Nếu gặp lỗi khi chạy thật trên Colab, gửi lại log để debug tiếp.
