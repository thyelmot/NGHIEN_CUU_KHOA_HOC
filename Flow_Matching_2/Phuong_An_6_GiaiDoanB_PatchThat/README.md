# Giai đoạn B (Phương án 6) — Patch thật vào fork DiffMM, có công tắc bật/tắt

Đây là kết quả của **Giai đoạn B** đề xuất trong
[`../Phuong_An_6_Learnable_Anchor_KeHoachChiTiet.md`](../Phuong_An_6_Learnable_Anchor_KeHoachChiTiet.md)
(mục 7): viết `GaussianDiffusionAnchorOT` — patch thật vào bản clone mới nhất của
[HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) — kế thừa trực tiếp `GaussianDiffusionOTCFM` (Phương án
3), với công tắc `anchor_w` (mặc định `0.0`, tương đương Phương án 3 tuyệt đối) để merge an toàn.

**Đây chưa phải Giai đoạn C** (chưa đóng gói theo `Folder_Base/`, chưa tạo repo GitHub riêng, chưa có
notebook Colab).

## Các file trong folder này

Bản fork đầy đủ của DiffMM, đã áp patch trực tiếp:

- **`Model.py`** — thêm 2 class ở cuối file: `GaussianDiffusionOTCFM` (Phương án 3, sao chép nguyên vẹn
  — cần làm lớp cha) và `GaussianDiffusionAnchorOT(GaussianDiffusionOTCFM)` (Phương án 6, mới). Không
  sửa `GaussianDiffusion` gốc.
- **`Main.py`** — 1 dòng import, 1 dòng khởi tạo (đọc thêm `args.anchor_w`), và 6 dòng gọi
  `training_losses`/`p_sample` (thêm đúng 1 tham số `uEmbeds_batch`/`uEmbeds[batch_index]` — tái sử dụng
  biến `uEmbeds` **đã có sẵn** trong `trainEpoch`, không cần tính embedding mới nào).
- **`Params.py`** — thêm `--sigma_min`, `--w_clip`, `--num_sample_steps`, `--anchor_w` (mặc định `0.0`).
- **`DataHandler.py`** — sửa sẵn lỗi môi trường đã biết (`self.trnMat.A` → `.toarray()`).
- **`verify_cpu.py`** — script kiểm chứng số học **trên chính `Model.py` thật**, chạy trên CPU.

## Kết quả kiểm chứng (chạy `python verify_cpu.py` để tái tạo, không cần GPU)

| Bài kiểm tra | Kết quả |
|---|---|
| **Hồi quy**: `anchor_w=0` → `mu_coef`, `sigma_coef`, `cfm_weight`, `q_sample`, `p_mean_variance`, **và cả vòng lặp `p_sample` đầy đủ** so với Phương án 3 | Trùng khít tuyệt đối (`atol=10⁻¹⁰`÷`10⁻¹²`) |
| `training_losses`: shape, không NaN/Inf, qua 4 giá trị `anchor_w` (0, 1, 5, −2) | Đạt; `cfm_weight` không đổi ở mọi `anchor_w` — xác nhận CT-6.7 trên code thật |
| `p_sample` tất định, model hoàn hảo, `anchor_w=0`, `steps=0` | Tái tạo chính xác `x_start` |
| `p_sample` `anchor_w=3` so với `anchor_w=0` (cùng seed) | Không NaN/Inf; **khác rõ** — xác nhận điểm neo thực sự có tác dụng |
| Round-trip với denoiser hoàn hảo, mọi tổ hợp `anchor_w∈{0,1,5}` × 3 seed khác nhau | Tái tạo đúng `x_start` (`atol=10⁻⁸`) — xác nhận CT-6.6 trên code thật |
| Biên: user rỗng (`x_start` toàn 0), `anchor_w=10⁶` | Không NaN/Inf |
| Quy mô gần thực tế (batch=1024, num_items=7050, giống Sports dataset) | `training_losses` (gồm cả tính điểm neo) ~0.4s, `p_sample` ~0.6s trên CPU/float64 — **rẻ hơn nhiều** so với ước tính ban đầu; xác nhận nhận định "chi phí tính toán thấp" ở bản kế hoạch chi tiết |

## Kết luận

Không phát hiện lỗi nào trong quá trình patch thật (khác với Phương án 5, nơi Giai đoạn B phát hiện 1
lỗi thật) — điều này phù hợp với việc bản kế hoạch chi tiết của Phương án 6 đã derive và kiểm chứng kỹ
CT-6.1→CT-6.7 **trước khi viết code**, và thiết kế bám rất sát cấu trúc đã kiểm chứng của Phương án 1/3
(chỉ thêm đúng 1 số hạng `σₜ·anchor_w·α_l` vào 2 công thức đã có sẵn).

`anchor_w=0` (giá trị mặc định trong `Params.py`) cho kết quả **trùng khít tuyệt đối** với Phương án 3 —
an toàn để merge. `anchor_w>0` cho kết quả khác biệt rõ ràng, không gây NaN/Inf, và chi phí tính toán
thêm cho việc tính điểm neo `α_l` là không đáng kể.

**Giới hạn của Giai đoạn B:** vẫn dùng `model` giả lập, **chưa** chạy qua mạng `Denoise` thật, **chưa**
chạy với dữ liệu thật trên GPU, **chưa** có số liệu Recall/NDCG. Bước tiếp theo (nếu xác nhận) là đóng
gói theo `Folder_Base/` (Giai đoạn C).
